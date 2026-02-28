use hound::{SampleFormat, WavSpec, WavWriter};
use viture_sensors::microphone::Microphone;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mic = Microphone::new();
    let sample_rate = 16_000;
    let samples = mic.record(1_000, Some(sample_rate))?;

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create("mic_record.wav", spec)?;
    for sample in samples {
        let s = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    println!("Saved mic_record.wav");
    Ok(())
}
