use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use base64::Engine;
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};

use crate::audio_devices;
use crate::errors::{AppError, Result};

pub type CancelToken = Arc<AtomicBool>;

pub async fn play_pcm_base64(
    data: &str,
    sample_rate: u32,
    selected_device_id: Option<String>,
    cancel: CancelToken,
) -> Result<()> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(data)?;
    if bytes.len() % 4 != 0 {
        return Err(AppError::Audio(
            "invalid PCM payload: byte length is not aligned to f32".to_string(),
        ));
    }

    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunks")))
        .collect();

    play_samples(samples, sample_rate, selected_device_id, cancel).await
}

pub async fn play_file(path: &str, selected_device_id: Option<String>, cancel: CancelToken) -> Result<()> {
    let ext = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    if ext.as_deref() != Some("wav") {
        return Err(AppError::Audio(
            "only WAV files are currently supported for direct playback".to_string(),
        ));
    }

    let mut reader = hound::WavReader::open(path)
        .map_err(|error| AppError::Audio(format!("failed to open WAV file: {error}")))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));

    let raw_samples = if spec.sample_format == hound::SampleFormat::Float {
        reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()
            .map_err(|error| {
                AppError::Audio(format!("failed to decode WAV float samples: {error}"))
            })?
    } else {
        let bits = spec.bits_per_sample;
        if bits <= 16 {
            reader
                .samples::<i16>()
                .map(|sample| {
                    sample
                        .map(|value| f32::from(value) / f32::from(i16::MAX))
                        .map_err(|error| {
                            AppError::Audio(format!("failed to decode WAV i16 sample: {error}"))
                        })
                })
                .collect::<Result<Vec<f32>>>()?
        } else {
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|value| value as f32 / i32::MAX as f32)
                        .map_err(|error| {
                            AppError::Audio(format!("failed to decode WAV i32 sample: {error}"))
                        })
                })
                .collect::<Result<Vec<f32>>>()?
        }
    };

    let samples = if channels == 1 {
        raw_samples
    } else {
        raw_samples
            .chunks(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect::<Vec<f32>>()
    };

    play_samples(samples, spec.sample_rate, selected_device_id, cancel).await
}

async fn play_samples(
    samples: Vec<f32>,
    sample_rate: u32,
    selected_device_id: Option<String>,
    cancel: CancelToken,
) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        play_samples_blocking(samples, sample_rate, selected_device_id, cancel)
    })
    .await
    .map_err(|error| AppError::Audio(format!("audio playback task failed: {error}")))?
}

fn play_samples_blocking(
    samples: Vec<f32>,
    sample_rate: u32,
    selected_device_id: Option<String>,
    cancel: CancelToken,
) -> Result<()> {
    let device = audio_devices::find_output_device(selected_device_id.as_deref())?;
    let default_config = device.default_output_config().map_err(|error| {
        AppError::Audio(format!("failed to get default output config: {error}"))
    })?;

    let stream_config: cpal::StreamConfig = default_config.config();
    if stream_config.sample_rate.0 != sample_rate {
        log::warn!(
      "requested sample rate {} differs from output sample rate {}; playback will use output sample rate",
      sample_rate,
      stream_config.sample_rate.0
    );
    }

    let channel_count = usize::from(stream_config.channels.max(1));
    let cursor = Arc::new(AtomicUsize::new(0));
    let is_done = Arc::new(AtomicBool::new(false));
    let done_signal = Arc::new((Mutex::new(false), Condvar::new()));
    let shared_samples = Arc::new(samples);

    let stream = match default_config.sample_format() {
        SampleFormat::F32 => build_stream::<f32>(
            &device,
            &stream_config,
            channel_count,
            shared_samples,
            cursor,
            is_done,
            done_signal.clone(),
            cancel.clone(),
        )?,
        SampleFormat::I16 => build_stream::<i16>(
            &device,
            &stream_config,
            channel_count,
            shared_samples,
            cursor,
            is_done,
            done_signal.clone(),
            cancel.clone(),
        )?,
        SampleFormat::U16 => build_stream::<u16>(
            &device,
            &stream_config,
            channel_count,
            shared_samples,
            cursor,
            is_done,
            done_signal.clone(),
            cancel.clone(),
        )?,
        unsupported => {
            return Err(AppError::Audio(format!(
                "unsupported sample format: {unsupported:?}"
            )));
        }
    };

    stream
        .play()
        .map_err(|error| AppError::Audio(format!("failed to start output stream: {error}")))?;

    let (lock, cvar) = &*done_signal;
    let mut completed = lock.lock().expect("done mutex poisoned");
    while !*completed && !cancel.load(Ordering::Relaxed) {
        let (guard, _timeout) = cvar
            .wait_timeout(completed, std::time::Duration::from_millis(50))
            .expect("done mutex poisoned");
        completed = guard;
    }

    drop(stream);
    Ok(())
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    samples: Arc<Vec<f32>>,
    cursor: Arc<AtomicUsize>,
    is_done: Arc<AtomicBool>,
    done_signal: Arc<(Mutex<bool>, Condvar)>,
    cancel: CancelToken,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    let error_callback = |error| {
        log::error!("audio stream error: {error}");
    };

    device
        .build_output_stream(
            config,
            move |output: &mut [T], _| {
                if cancel.load(Ordering::Relaxed) {
                    for s in output.iter_mut() {
                        *s = T::from_sample(0.0f32);
                    }
                    if !is_done.swap(true, Ordering::SeqCst) {
                        let (lock, cvar) = &*done_signal;
                        if let Ok(mut completed) = lock.lock() {
                            *completed = true;
                            cvar.notify_all();
                        }
                    }
                    return;
                }

                let frame_count = output.len() / channels;
                let start_frame = cursor.fetch_add(frame_count, Ordering::Relaxed);

                for (frame_index, frame) in output.chunks_mut(channels).enumerate() {
                    let source_index = start_frame + frame_index;
                    let sample = samples.get(source_index).copied().unwrap_or(0.0);
                    let converted = T::from_sample(sample);
                    for channel_sample in frame.iter_mut() {
                        *channel_sample = converted;
                    }
                }

                if start_frame + frame_count >= samples.len()
                    && !is_done.swap(true, Ordering::SeqCst)
                {
                    let (lock, cvar) = &*done_signal;
                    if let Ok(mut completed) = lock.lock() {
                        *completed = true;
                        cvar.notify_all();
                    }
                }
            },
            error_callback,
            None,
        )
        .map_err(|error| AppError::Audio(format!("failed to build output stream: {error}")))
}

#[cfg(test)]
mod tests {
    use base64::Engine;

    #[test]
    fn pcm_base64_decoding_roundtrip() {
        let source = vec![0.0f32, 0.5f32, -0.5f32, 1.0f32];
        let mut bytes = Vec::new();
        for sample in &source {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("decode base64");

        let values: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk")))
            .collect();

        assert_eq!(values, source);
    }
}
