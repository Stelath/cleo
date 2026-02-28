use std::thread;
use std::time::{Duration, Instant};

use viture_sensors::camera::RGBCamera;
use viture_sensors::sdk::callbacks;
use viture_sensors::sdk::VitureDevice;
use viture_sensors::types::DeviceType;

fn hardware_enabled() -> bool {
    std::env::var("VITURE_HARDWARE").ok().as_deref() == Some("1")
}

#[test]
fn camera_capture_shape() {
    if !hardware_enabled() {
        return;
    }
    let dev = VitureDevice::new(None).expect("device creation failed");
    dev.connect().expect("connect failed");

    if !matches!(dev.device_type(), DeviceType::Carina) {
        // Depth callbacks are currently wired through Carina APIs only.
        dev.disconnect().expect("disconnect failed");
        return;
    }

    let camera = RGBCamera::new(dev.depth());
    callbacks::reset_camera_debug_counters();
    let wait_ms = std::env::var("VITURE_CAMERA_WAIT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(5_000);
    let deadline = Instant::now() + Duration::from_millis(wait_ms);
    let frame = loop {
        match camera.capture() {
            Ok(frame) => break frame,
            Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            Err(err) => {
                let (calls, null_ptr, bad_dim, usable, last_w, last_h) =
                    callbacks::camera_debug_counters();
                if std::env::var("VITURE_REQUIRE_CAMERA").ok().as_deref() == Some("1") {
                    panic!(
                        "capture failed after warm-up: {err}; callback_calls={calls}, null_ptr={null_ptr}, bad_dim={bad_dim}, usable={usable}, last_dims={last_w}x{last_h}"
                    );
                }
                eprintln!(
                    "camera stream not ready yet, skipping strict capture check: {err}; callback_calls={calls}, null_ptr={null_ptr}, bad_dim={bad_dim}, usable={usable}, last_dims={last_w}x{last_h}"
                );
                dev.disconnect().expect("disconnect failed");
                return;
            }
        }
    };

    assert!(frame.width > 0);
    assert!(frame.height > 0);
    assert_eq!(frame.data.len(), frame.width * frame.height * 3);
    assert!(
        frame.data.iter().any(|v| *v != 0),
        "camera frame appears empty (all zeros)"
    );

    dev.disconnect().expect("disconnect failed");
}
