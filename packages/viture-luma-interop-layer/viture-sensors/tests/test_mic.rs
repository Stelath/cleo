use viture_sensors::microphone::Microphone;

fn hardware_enabled() -> bool {
    std::env::var("VITURE_HARDWARE").ok().as_deref() == Some("1")
}

#[test]
fn mic_record_has_samples() {
    if !hardware_enabled() {
        return;
    }
    let mic = Microphone::new();
    let sample_rate = 16_000u32;
    let duration_ms = 500u64;
    let samples = mic
        .record(duration_ms, Some(sample_rate))
        .expect("record failed");

    assert!(!samples.is_empty(), "no audio samples returned");
    let minimum_expected = (sample_rate as usize * duration_ms as usize) / 10 / 1000;
    assert!(
        samples.len() >= minimum_expected,
        "too few samples: {} < {}",
        samples.len(),
        minimum_expected
    );

    for sample in &samples {
        assert!(sample.is_finite(), "non-finite mic sample");
    }
    let peak = samples
        .iter()
        .fold(0.0f32, |acc, s| if s.abs() > acc { s.abs() } else { acc });
    assert!(
        peak <= 1.5,
        "unexpectedly large normalized sample peak: {peak}"
    );
}
