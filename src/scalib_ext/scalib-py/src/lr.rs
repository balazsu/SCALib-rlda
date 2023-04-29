//! Python binding of SCALib's LR implementation.

use bincode::{deserialize, serialize};
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, ToPyArray,IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use std::sync::Arc;

#[pyclass(module = "scalib._scalib_ext")]
pub(crate) struct RLDA {
    inner: Option<scalib::lr::RLDA>,
}
#[pymethods]
impl RLDA {
    /// Init an empty RLDA model
    #[new]
    #[pyo3(signature = (*args))]
    fn new(py: Python, args: &PyTuple) -> PyResult<Self> {
        if args.len() == 0 {
            Ok(Self { inner: None })
        } else {
            let (nb,ns,nv,p): (
                usize,
                usize,
                usize,
                usize
            ) = args.extract()?;
            let inner = py.allow_threads(|| {
                Some(scalib::lr::RLDA::new(nb, ns, nv, p))
            });
            Ok(Self { inner })
        }
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.inner = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }

    /// Add n measurements to the model
    // x: traces with shape (n,ns)
    // y: random value realization (nv,n)
    // gemm_algo is 0 for ndarray gemm, x>0 for BLIS gemm with x threads.
    fn update(
        &mut self,
        py: Python,
        x: PyReadonlyArray2<i16>,
        y: PyReadonlyArray2<u64>,
        gemm_algo: u32,
        config: crate::ConfigWrapper,
    ) {
        let x = x.as_array();
        let y = y.as_array();    
        config.on_worker(py, |_| self.inner.as_mut().unwrap().update(x, y, gemm_algo));
    }
    
    fn solve<'py>(&mut self, py: Python<'py>,config: crate::ConfigWrapper) -> &'py PyArray3<f64>{
        config.on_worker(py, |_| self.inner.as_mut().unwrap().solve()).to_pyarray(py)
    }

    fn get_state<'py>(
        &self,
        py: Python<'py>
    ) -> (
        usize,
        usize,
        usize,
        usize,
        usize,
        &'py PyArray1<f64>,
        &'py PyArray3<f64>,
        &'py PyArray3<f64>,
        &'py PyArray2<f64>,
        &'py PyArray3<f64>,
        &'py PyArray3<f64>,
        &'py PyArray4<f64>,
    ) {
        (
        self.inner.as_ref().unwrap().ns,
        self.inner.as_ref().unwrap().nb,
        self.inner.as_ref().unwrap().n,
        self.inner.as_ref().unwrap().nv,
        self.inner.as_ref().unwrap().p,
        self.inner.as_ref().unwrap().traces_sum.to_pyarray(py),
        self.inner.as_ref().unwrap().xtx.to_pyarray(py),
        self.inner.as_ref().unwrap().xty.to_pyarray(py),
        self.inner.as_ref().unwrap().scatter.to_pyarray(py),
        self.inner.as_ref().unwrap().norm_proj.to_pyarray(py),
        self.inner.as_ref().unwrap().proj_coefs.to_pyarray(py),
        self.inner.as_ref().unwrap().mu_chunks.to_pyarray(py),
        )
    }

    /// Set the accumulator state
    #[staticmethod]
    fn from_state(
        ns: usize,
        nb: usize,
        n: usize,
        nv: usize,
        p: usize,
        traces_sum: PyReadonlyArray1<f64>,
        xtx: PyReadonlyArray3<f64>,
        xty: PyReadonlyArray3<f64>,
        scatter: PyReadonlyArray2<f64>,
        norm_proj: PyReadonlyArray3<f64>,
        proj_coefs: PyReadonlyArray3<f64>,
        mu_chunks: PyReadonlyArray4<f64>,
    ) -> Self {
        let inner = scalib::lr::RLDA{
            ns,
            nb,
            nv,
            p,
            n,
            traces_sum: traces_sum.as_array().to_owned(),
            xtx: xtx.as_array().to_owned(),
            xty: xty.as_array().to_owned(),
            scatter: scatter.as_array().to_owned(),
            norm_proj: norm_proj.as_array().to_owned(),
            proj_coefs: proj_coefs.as_array().to_owned(),
            mu_chunks: mu_chunks.as_array().to_owned(),
        };
        Self { inner: Some(inner) }
    }

    fn predict_proba_with_nparray<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
        v: usize,
        config: crate::ConfigWrapper,
    ) -> PyResult<&'py PyArray2<f64>> {
        let x = x.as_array();
        let prs = config.on_worker(py, |_| self.inner.as_ref().unwrap().predict_proba(x,v));
        Ok(prs.into_pyarray(py))
    }

    fn get_proj_coefs<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<&'py PyArray3<f64>> {
        Ok(self.inner.as_ref().unwrap().proj_coefs.to_pyarray(py))
    }

    fn get_norm_proj<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<&'py PyArray3<f64>> {
        Ok(self.inner.as_ref().unwrap().norm_proj.to_pyarray(py))
    }

    fn get_clustered_model<'py>(
        &self,
        py: Python<'py>,
        var_id: usize,
        store_associated_classes: bool, //TODO
        store_marginalized_weights: bool,
        max_squared_distance: f64,
        max_cluster_number: u32,
    ) -> PyResult<RLDAClusteredModel> {
        Ok(RLDAClusteredModel{
            inner: Some(Arc::new(self.inner.as_ref().unwrap().get_clustered_model(
                var_id,store_associated_classes,store_marginalized_weights,max_squared_distance,max_cluster_number
            )))
        })
    } 
}




#[pyclass(module="scalib._scalib_ext")]
pub(crate) struct RLDAClusteredModel {
    pub inner: Option<Arc<scalib::lr::RLDAClusteredModel>>,
}

#[pymethods]
impl RLDAClusteredModel {
    #[new]
    #[args(args = "*")]
    fn new(py: Python, args: &PyTuple) -> PyResult<Self> {
        Ok(Self { inner: None })
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.inner = deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
    }

    fn nearest<'py>(&mut self, py: Python<'py>, point: PyReadonlyArray1<f64>) -> (usize, f64) {
        self.inner
            .as_ref()
            .unwrap()
            .nearest(point.as_slice().unwrap())
    }

    fn get_max_sqdist<'py>(&mut self, py: Python<'py>) -> f64 {
        self.inner.as_ref().unwrap().get_max_dist()
    }

    fn get_size<'py>(&mut self, py: Python<'py>) -> u32 {
        self.inner.as_ref().unwrap().get_size()
    }

    fn get_weights<'py>(&mut self, py: Python<'py>) -> (&'py PyArray1<u64>, &'py PyArray1<u32>, &'py PyArray1<f64>) {
        let (ids, weights,weights_and) = self.inner.as_ref().unwrap().get_weights();
        (ids.to_pyarray(py), weights.to_pyarray(py),weights_and.to_pyarray(py))
    }

    fn get_tree<'py>(&mut self, py: Python<'py>) -> &'py PyArray1<u64> {
        self.inner.as_ref().unwrap().get_tree().to_pyarray(py)
        //let (a,b) = self.inner.get_tree();
        //(a.to_pyarray(py),b.to_pyarray(py))
    }

    fn get_marginalized_weights<'py>(&mut self, py: Python<'py>) -> &'py PyArray2<u32> {
        self.inner
            .as_ref()
            .unwrap()
            .get_marginalized_weights()
            .unwrap()
            .to_pyarray(py)
    }

    fn get_close_clusters<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<f64>,max_n_points:usize) ->  &'py PyArray1<usize> {
        self.inner
            .as_ref()
            .unwrap()
            .get_close_clusters(point
                .as_slice()
                .unwrap(),
                max_n_points)
            .map(|(c_id, _n_associated)| c_id)
            .collect::<Vec<usize>>()
            .to_pyarray(py)

    }

    fn get_clustered_prs<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<i16>,
        label: PyReadonlyArray1<u64>,
        max_popped_classes: usize,
        config: crate::ConfigWrapper,
    ) -> PyResult<(&'py PyArray1<f64>,&'py PyArray1<f64>)> {
        let x = x.as_array();
        let label = label.as_array();
        let prs = config.on_worker(py, |_| self.inner.as_ref().unwrap().clustered_prs(x,label,max_popped_classes));
        Ok((prs.0.to_pyarray(py),prs.1.to_pyarray(py)))
    }
}