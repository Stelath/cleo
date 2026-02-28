use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::python::py_device::Shared;

#[pyclass(name = "Mic")]
pub struct PyMic {
    shared: Shared,
}

impl PyMic {
    pub(crate) fn from_shared(shared: Shared) -> Self {
        Self { shared }
    }
}

#[pymethods]
impl PyMic {
    #[pyo3(signature = (duration_ms, sample_rate=Some(48000)))]
    fn record<'py>(
        &self,
        py: Python<'py>,
        duration_ms: u64,
        sample_rate: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let samples = self.shared.mic.record(duration_ms, sample_rate)?;
        let array = Array1::from_vec(samples);
        Ok(array.into_pyarray_bound(py))
    }
}
