use std::sync::Arc;

use parking_lot::Mutex;
use pyo3::prelude::*;

use crate::camera::RGBCamera;
use crate::error::Result;
use crate::microphone::Microphone;
use crate::python::py_types::PyDeviceInfo;
use crate::python::{PyCamera, PyDepthCamera, PyIMU, PyMic};
use crate::sdk::VitureDevice;

pub(crate) struct SharedDevice {
    pub(crate) device: Arc<Mutex<VitureDevice>>,
    pub(crate) mic: Microphone,
}

impl SharedDevice {
    pub(crate) fn rgb_camera(&self) -> RGBCamera {
        let depth = self.device.lock().depth();
        RGBCamera::new(depth)
    }
}

pub(crate) type Shared = Arc<SharedDevice>;

#[pyclass(name = "Device")]
pub struct PyDevice {
    pub(crate) shared: Shared,
}

impl PyDevice {
    fn with_device<T>(&self, f: impl FnOnce(&VitureDevice) -> Result<T>) -> PyResult<T> {
        let guard = self.shared.device.lock();
        f(&guard).map_err(Into::into)
    }
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (product_id=None))]
    fn new(product_id: Option<i32>) -> PyResult<Self> {
        let dev = VitureDevice::new(product_id)?;
        let shared = Arc::new(SharedDevice {
            device: Arc::new(Mutex::new(dev)),
            mic: Microphone::new(),
        });
        Ok(Self { shared })
    }

    fn connect(&self) -> PyResult<()> {
        self.with_device(|d| d.connect())
    }

    fn disconnect(&self) -> PyResult<()> {
        self.with_device(|d| d.disconnect())
    }

    fn is_connected(&self) -> bool {
        self.shared.device.lock().is_connected()
    }

    fn info(&self) -> PyResult<PyDeviceInfo> {
        self.with_device(|d| d.info().map(Into::into))
    }

    fn set_brightness(&self, value: u8) -> PyResult<()> {
        self.with_device(|d| d.set_brightness(value))
    }

    fn get_brightness(&self) -> PyResult<u8> {
        self.with_device(|d| d.get_brightness())
    }

    fn set_display_mode(&self, mode: i32) -> PyResult<()> {
        self.with_device(|d| d.set_display_mode(mode))
    }

    fn set_volume(&self, value: u8) -> PyResult<()> {
        self.with_device(|d| d.set_volume(value))
    }

    #[getter]
    fn imu(&self) -> PyIMU {
        PyIMU::from_shared(self.shared.clone())
    }

    #[getter]
    fn camera(&self) -> PyCamera {
        PyCamera::from_shared(self.shared.clone())
    }

    #[getter]
    fn depth(&self) -> PyDepthCamera {
        PyDepthCamera::from_shared(self.shared.clone())
    }

    #[getter]
    fn mic(&self) -> PyMic {
        PyMic::from_shared(self.shared.clone())
    }
}
