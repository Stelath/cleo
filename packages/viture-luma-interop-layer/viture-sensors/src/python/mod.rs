mod py_camera;
mod py_depth;
mod py_device;
mod py_imu;
mod py_mic;
mod py_types;

use pyo3::prelude::*;

pub use py_camera::PyCamera;
pub use py_depth::PyDepthCamera;
pub use py_device::PyDevice;
pub use py_imu::PyIMU;
pub use py_mic::PyMic;
pub use py_types::{PyDeviceInfo, PyIMUReading};

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDevice>()?;
    module.add_class::<PyIMU>()?;
    module.add_class::<PyCamera>()?;
    module.add_class::<PyDepthCamera>()?;
    module.add_class::<PyMic>()?;
    module.add_class::<PyIMUReading>()?;
    module.add_class::<PyDeviceInfo>()?;
    Ok(())
}
