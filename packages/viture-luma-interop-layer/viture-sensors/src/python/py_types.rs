use pyo3::prelude::*;

use crate::types::{DeviceInfo, IMUReading};

#[pyclass(name = "IMUReading")]
#[derive(Clone)]
pub struct PyIMUReading {
    #[pyo3(get)]
    pub accel: Vec<f32>,
    #[pyo3(get)]
    pub gyro: Vec<f32>,
    #[pyo3(get)]
    pub quaternion: Vec<f32>,
    #[pyo3(get)]
    pub timestamp_ns: u64,
}

impl From<IMUReading> for PyIMUReading {
    fn from(value: IMUReading) -> Self {
        Self {
            accel: value.accel.to_vec(),
            gyro: value.gyro.to_vec(),
            quaternion: value.quaternion.to_vec(),
            timestamp_ns: value.timestamp_ns,
        }
    }
}

#[pyclass(name = "DeviceInfo")]
#[derive(Clone)]
pub struct PyDeviceInfo {
    #[pyo3(get)]
    pub device_type: String,
    #[pyo3(get)]
    pub firmware_version: String,
    #[pyo3(get)]
    pub product_id: i32,
}

impl From<DeviceInfo> for PyDeviceInfo {
    fn from(value: DeviceInfo) -> Self {
        Self {
            device_type: format!("{:?}", value.device_type),
            firmware_version: value.firmware_version,
            product_id: value.product_id,
        }
    }
}
