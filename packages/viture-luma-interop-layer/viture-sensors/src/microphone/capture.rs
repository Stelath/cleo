use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};

use crate::error::{Result, VitureError};

#[derive(Clone, Default)]
pub struct Microphone;

impl Microphone {
    pub fn new() -> Self {
        Self
    }

    pub fn record(&self, duration_ms: u64, sample_rate: Option<u32>) -> Result<Vec<f32>> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| VitureError::Audio("No input audio device available".to_string()))?;

        let default_supported_config = device
            .default_input_config()
            .map_err(|e| VitureError::Audio(format!("Failed to read default input config: {e}")))?;
        let sample_format = default_supported_config.sample_format();
        let mut config = default_supported_config.config();
        if let Some(sr) = sample_rate {
            config.sample_rate = cpal::SampleRate(sr);
        }

        let shared = Arc::new(Mutex::new(Vec::<f32>::new()));
        let mut stream = Self::build_stream(&device, &config, sample_format, Arc::clone(&shared));
        if stream.is_err() && sample_rate.is_some() {
            let fallback = default_supported_config.config();
            stream = Self::build_stream(&device, &fallback, sample_format, Arc::clone(&shared));
        }
        let stream = stream?;

        stream
            .play()
            .map_err(|e| VitureError::Audio(format!("Failed to start input stream: {e}")))?;
        std::thread::sleep(Duration::from_millis(duration_ms));
        drop(stream);

        shared
            .lock()
            .map(|v| v.clone())
            .map_err(|_| VitureError::Audio("Audio buffer lock poisoned".to_string()))
    }

    fn build_stream(
        device: &cpal::Device,
        config: &StreamConfig,
        sample_format: SampleFormat,
        shared: Arc<Mutex<Vec<f32>>>,
    ) -> Result<Stream> {
        let err_fn = |err| log::error!("audio stream error: {err}");
        match sample_format {
            SampleFormat::F32 => {
                let shared_for_callback = Arc::clone(&shared);
                device
                    .build_input_stream(
                        config,
                        move |data: &[f32], _| {
                            if let Ok(mut out) = shared_for_callback.lock() {
                                out.extend_from_slice(data);
                            }
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| VitureError::Audio(format!("Failed to create f32 stream: {e}")))
            }
            SampleFormat::I16 => {
                let shared_for_callback = Arc::clone(&shared);
                device
                    .build_input_stream(
                        config,
                        move |data: &[i16], _| {
                            if let Ok(mut out) = shared_for_callback.lock() {
                                out.extend(data.iter().map(|s| *s as f32 / i16::MAX as f32));
                            }
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| VitureError::Audio(format!("Failed to create i16 stream: {e}")))
            }
            SampleFormat::U16 => {
                let shared_for_callback = Arc::clone(&shared);
                device
                    .build_input_stream(
                        config,
                        move |data: &[u16], _| {
                            if let Ok(mut out) = shared_for_callback.lock() {
                                out.extend(
                                    data.iter()
                                        .map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0),
                                );
                            }
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| VitureError::Audio(format!("Failed to create u16 stream: {e}")))
            }
            _ => Err(VitureError::Audio(format!(
                "Unsupported input sample format: {sample_format:?}"
            ))),
        }
    }
}
