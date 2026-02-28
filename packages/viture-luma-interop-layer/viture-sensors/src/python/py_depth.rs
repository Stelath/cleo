use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use crate::python::py_device::Shared;

#[pyclass(name = "DepthCamera")]
pub struct PyDepthCamera {
    shared: Shared,
}

impl PyDepthCamera {
    pub(crate) fn from_shared(shared: Shared) -> Self {
        Self { shared }
    }
}

#[pymethods]
impl PyDepthCamera {
    #[pyo3(signature = (side="left"))]
    fn capture<'py>(&self, py: Python<'py>, side: &str) -> PyResult<Bound<'py, PyArray2<u16>>> {
        if side != "left" && side != "right" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "side must be 'left' or 'right'",
            ));
        }
        let depth = self.shared.device.lock().depth().capture(side)?;
        let array = Array2::from_shape_vec((depth.height, depth.width), depth.data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("shape error: {e}")))?;
        Ok(array.into_pyarray_bound(py))
    }
}
