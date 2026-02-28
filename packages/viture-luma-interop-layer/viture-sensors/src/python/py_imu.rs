use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use parking_lot::Mutex;
use pyo3::prelude::*;

use crate::python::py_device::Shared;
use crate::python::py_types::PyIMUReading;

struct WorkerState {
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

#[pyclass(name = "IMU")]
pub struct PyIMU {
    shared: Shared,
    worker: Arc<Mutex<Option<WorkerState>>>,
}

impl PyIMU {
    pub(crate) fn from_shared(shared: Shared) -> Self {
        Self {
            shared,
            worker: Arc::new(Mutex::new(None)),
        }
    }
}

#[pymethods]
impl PyIMU {
    fn read(&self) -> PyResult<PyIMUReading> {
        let guard = self.shared.device.lock();
        guard.read_imu().map(Into::into).map_err(Into::into)
    }

    #[pyo3(signature = (callback, interval_ms=16))]
    fn start_stream(&self, callback: PyObject, interval_ms: u64) -> PyResult<()> {
        self.stop_stream()?;

        let shared = self.shared.clone();
        let running = Arc::new(AtomicBool::new(true));
        let running_for_thread = running.clone();
        let handle = thread::spawn(move || {
            while running_for_thread.load(Ordering::SeqCst) {
                let reading = {
                    let guard = shared.device.lock();
                    guard.read_imu()
                };
                if let Ok(reading) = reading {
                    Python::with_gil(|py| {
                        let _ = callback.call1(py, (PyIMUReading::from(reading),));
                    });
                }
                thread::sleep(Duration::from_millis(interval_ms));
            }
        });

        *self.worker.lock() = Some(WorkerState {
            running,
            handle: Some(handle),
        });
        Ok(())
    }

    fn stop_stream(&self) -> PyResult<()> {
        if let Some(mut state) = self.worker.lock().take() {
            state.running.store(false, Ordering::SeqCst);
            if let Some(handle) = state.handle.take() {
                let _ = handle.join();
            }
        }
        Ok(())
    }
}
