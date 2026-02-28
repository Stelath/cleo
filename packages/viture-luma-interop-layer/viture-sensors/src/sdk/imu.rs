use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::Receiver;
use parking_lot::RwLock;

use crate::error::{Result, VitureError};
use crate::types::IMUReading;

#[derive(Clone)]
pub struct IMUStream {
    latest: Arc<RwLock<Option<IMUReading>>>,
}

impl IMUStream {
    pub fn new(rx: Receiver<IMUReading>) -> Self {
        let latest = Arc::new(RwLock::new(None));
        let latest_for_thread = latest.clone();
        thread::spawn(move || {
            while let Ok(reading) = rx.recv() {
                *latest_for_thread.write() = Some(reading);
            }
        });
        Self { latest }
    }

    pub fn read(&self) -> Result<IMUReading> {
        self.latest
            .read()
            .clone()
            .ok_or(VitureError::StreamEmpty("IMU stream has no samples yet"))
    }

    pub fn wait_for_first_sample(&self, timeout_ms: u64) -> Result<IMUReading> {
        let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            if let Ok(reading) = self.read() {
                return Ok(reading);
            }
            if std::time::Instant::now() >= deadline {
                return Err(VitureError::StreamEmpty("Timed out waiting for IMU sample"));
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}
