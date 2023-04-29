#![allow(dead_code)]
#![allow(unused_imports)]

use bincode::{deserialize, serialize};
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, ToPyArray,IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use crate::lr::RLDAClusteredModel;
use std::sync::Arc;

#[pyclass(module="scalib._scalib_ext")]
pub(crate) struct ItEstimator {
    inner: scalib::information::ItEstimator,
}

#[pymethods]
impl ItEstimator {
    #[new]
    #[args(args = "*")]
    fn new(py: Python, model: &mut crate::lr::RLDAClusteredModel, max_popped_classes: usize) -> PyResult<Self> {
            //let (model,max_popped_classes): (crate::lr::RLDAClusteredModel, usize)= args.extract()?;
            let inner = py.allow_threads(|| {
                scalib::information::ItEstimator::new(model.inner.as_ref().unwrap().clone(),max_popped_classes)
            });
            Ok(Self { inner })
        
    }

    
    fn fit_u<'py>(
        &mut self,
        py: Python<'py>,
        traces: PyReadonlyArray2<i16>,
        label: PyReadonlyArray1<u64>,
        config: crate::ConfigWrapper,
    ) -> PyResult<(f64,f64)> {
        let traces = traces.as_array();
        let label = label.as_array();
        let result = config.on_worker(py, |_| self.inner.fit_u(traces,label));
        Ok(result)
    }
}