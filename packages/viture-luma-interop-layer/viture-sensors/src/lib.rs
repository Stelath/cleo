pub mod camera;
pub mod error;
pub mod microphone;
pub mod python;
pub mod sdk;
pub mod types;

use pyo3::prelude::*;

#[pymodule]
fn _native(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::builder().is_test(true).try_init();
    python::register(py, module)
}
