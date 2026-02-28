use std::thread;
use std::time::Duration;

use viture_sensors::sdk::VitureDevice;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = VitureDevice::new(None)?;
    device.connect()?;
    println!("Connected. Reading IMU...");

    for _ in 0..200 {
        match device.read_imu() {
            Ok(reading) => {
                println!(
                    "ts={} accel={:?} gyro={:?} quat={:?}",
                    reading.timestamp_ns, reading.accel, reading.gyro, reading.quaternion
                );
            }
            Err(err) => eprintln!("IMU read not ready: {err}"),
        }
        thread::sleep(Duration::from_millis(20));
    }

    device.disconnect()?;
    Ok(())
}
