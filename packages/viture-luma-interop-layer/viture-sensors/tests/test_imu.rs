use std::thread;
use std::time::{Duration, Instant};

use viture_sensors::sdk::VitureDevice;

fn hardware_enabled() -> bool {
    std::env::var("VITURE_HARDWARE").ok().as_deref() == Some("1")
}

#[test]
fn imu_stream_returns_shape() {
    if !hardware_enabled() {
        return;
    }
    let dev = VitureDevice::new(None).expect("device creation failed");
    dev.connect().expect("connect failed");
    let reading = dev
        .imu()
        .wait_for_first_sample(2_000)
        .expect("no imu sample");
    assert_eq!(reading.accel.len(), 3);
    assert_eq!(reading.gyro.len(), 3);
    assert_eq!(reading.quaternion.len(), 4);

    for value in reading
        .accel
        .iter()
        .chain(reading.gyro.iter())
        .chain(reading.quaternion.iter())
    {
        assert!(value.is_finite(), "IMU contains non-finite value: {value}");
    }

    let start_ts = reading.timestamp_ns;
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut observed_new_sample = false;
    while Instant::now() < deadline {
        if let Ok(next) = dev.read_imu() {
            if next.timestamp_ns > start_ts {
                observed_new_sample = true;
                break;
            }
        }
        thread::sleep(Duration::from_millis(10));
    }
    assert!(
        observed_new_sample,
        "IMU timestamp did not advance within 3 seconds"
    );

    let q = reading.quaternion;
    let q_norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    assert!(
        (q_norm - 1.0).abs() < 0.25,
        "Quaternion norm should be near 1.0, got {q_norm}"
    );

    dev.disconnect().expect("disconnect failed");
}
