use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;

use crate::python::py_device::Shared;

#[pyclass(name = "Camera")]
pub struct PyCamera {
    shared: Shared,
}

impl PyCamera {
    pub(crate) fn from_shared(shared: Shared) -> Self {
        Self { shared }
    }
}

#[pymethods]
impl PyCamera {
    fn capture<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let camera = self.shared.rgb_camera();
        let frame = camera.capture()?;
        let array = Array3::from_shape_vec((frame.height, frame.width, 3), frame.data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("shape error: {e}")))?;
        Ok(array.into_pyarray_bound(py))
    }
}
