#![allow(dead_code)]
#![allow(unused_imports)]

use std::ops::{AddAssign, SubAssign};
use ndarray::{linalg::Dot, s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis, NewAxis,Zip};
use nshare::{ToNalgebra, ToNdarray1, ToNdarray2};
use geigen::Geigen;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::cmp::{min,max};
use rayon::prelude::*;
use crate::lr::RLDAClusteredModel;
use std::sync::Arc;

pub struct ItEstimator {
    pub model: Arc<RLDAClusteredModel>,
    pub sum_prs_l: f64,
    pub sum_prs_h: f64,
    pub sum_prs2_l: f64,
    pub sum_prs2_h: f64,
    pub n: usize,
    pub max_popped_classes: usize
}

impl ItEstimator {
    pub fn new(model: Arc<RLDAClusteredModel>, max_popped_classes: usize) -> Self {
        let nbits = model.coefs.shape()[1];
        Self {
            model: model.clone(),
            sum_prs_l:  0.0f64,
            sum_prs_h:  0.0f64,
            sum_prs2_l: 0.0f64,
            sum_prs2_h: 0.0f64,
            n: 0 as usize,
            max_popped_classes: max_popped_classes,
        }
    }

    pub fn fit_u(
        &mut self,
        traces: ArrayView2<i16>,
        labels: ArrayView1<u64>,
    ) -> (f64,f64) {
        let (prs_l,prs_h) = self.model.clustered_prs(traces,labels,self.max_popped_classes);
        self.sum_prs_l += prs_l.fold(0.0,|mut x,y| {x+=y.log2(); return x});
        self.sum_prs_h += prs_h.fold(0.0,|mut x,y| {x+=y.log2(); return x});
        self.sum_prs2_l += prs_l.fold(0.0,|mut x,y| {x+=y.log2()*y.log2(); return x});
        self.sum_prs2_h += prs_h.fold(0.0,|mut x,y| {x+=y.log2()*y.log2(); return x});
        self.n += prs_l.shape()[0];

        return self.get_information();
    }

    pub fn get_information(
        &mut self,
    ) -> (f64,f64) {
        let nbits = self.model.coefs.shape()[1] as f64 - 1.0 ;
        let pi_l = nbits + self.sum_prs_l/self.n as f64;
        let pi_h = nbits + self.sum_prs_h/self.n as f64;
        return (pi_l,pi_h)
    }


}
