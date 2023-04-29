#![allow(dead_code)]
#![allow(unused_imports)]

use std::ops::{AddAssign, SubAssign};
use ndarray::{linalg::Dot, s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis, NewAxis,Zip};
use nshare::{ToNalgebra, ToNdarray1, ToNdarray2};
use geigen::Geigen;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::cmp::{min,max};
use rayon::prelude::*;

const NBITS_CHUNK :usize = 8;
const SIZE_CHUNK :usize = 1 << NBITS_CHUNK;

#[derive(Serialize, Deserialize)]
pub struct RLDA {
    /// Number of samples in trace
    pub ns: usize,
    /// Number of bits
    pub nb: usize,
    /// Total number of traces
    pub n: usize,
    /// Number of variables
    pub nv: usize,
    /// Number of dimensions after dimensionality reduction
    pub p: usize,
    /// Sum traces Shape (ns,).
    pub traces_sum: Array1<f64>,
    /// X^T*X (shape nv*(nb+1)*(nb+1)), +1 is for intercept
    pub xtx: Array3<f64>,
    /// X^T*trace (shape nv*(nb+1)*ns), +1 is for intercept
    pub xty: Array3<f64>,
    /// trace^T*trace (ns*ns)
    pub scatter: Array2<f64>,
    /// Normalized Projection matrix to the subspace. shape of (nv,ns,p)
    pub norm_proj: Array3<f64>,
    /// Regression coefficients in the projected subspace. Shape of (nv,p,nb+1)  +1 is for intercept
    pub proj_coefs: Array3<f64>,
    /// Precomputed partial mus. Shape of (nv,n_chunks,size_chunk,p)
    pub mu_chunks: Array4<f64>,
}

/// Coefficient bit from a class. 0th bit is always 1, ith bit is i-1 th bit of c
fn ext_bit(x: u64, i: usize) -> u64 {
    if i == 0 {
        1
    } else {
        (x >> (i - 1)) & 0x1
    }
}

fn int2mul(x: u64) -> f64 {
    if x == 0 {
        -1.0
    } else {
        1.0
    }
}


impl RLDA {
    pub fn new(nb: usize, ns: usize, nv: usize, p: usize) -> Self {
        let n_chunks: usize = (nb + NBITS_CHUNK-1)/NBITS_CHUNK; 
        Self {
            ns,
            nb,
            nv,
            p,
            n: 0,
            traces_sum: Array1::zeros((ns,)),
            xtx: Array3::zeros((nv, nb + 1, nb + 1)),
            xty: Array3::zeros((nv, nb + 1, ns)),
            scatter: Array2::zeros((ns, ns)),
            norm_proj: Array3::zeros((nv, p, ns)),
            proj_coefs: Array3::zeros((nv, p, nb + 1)),
            mu_chunks: Array4::zeros((nv, n_chunks, SIZE_CHUNK, p)),
        }
    }

    pub fn update(&mut self, traces: ArrayView2<i16>, classes: ArrayView2<u64>, gemm_algo: u32) {
        assert_eq!(classes.shape()[0], self.nv);
        assert_eq!(traces.shape()[1], self.ns);
        let nt = traces.shape()[0];
        assert_eq!(classes.shape()[1], nt);

        self.n += nt;
        let traces_buf = traces.mapv(|x| x as f64);
        self.traces_sum.add_assign(&traces_buf.sum_axis(Axis(0)));

        crate::matrixmul::opt_dgemm(
            traces_buf.t(),
            traces_buf.view(),
            self.scatter.view_mut(),
            1.0,
            1.0,
            gemm_algo,
        );


        Zip::indexed(self.xtx.outer_iter_mut()).and(self.xty.outer_iter_mut()).into_par_iter().for_each(
            |(k,mut xtx,mut xty)| {
                let classes = classes.slice(s![k, ..]);
                for i in 0..(self.nb + 1) {
                    for j in 0..(self.nb + 1) {
                        let s: f64 = classes
                            .iter()
                            .map(|c| int2mul(ext_bit(*c, i) ^ ext_bit(*c, j)))
                            .sum();
                        *xtx.get_mut((i, j)).unwrap() -= s;
                    }
                }
                for i in 0..(self.nb + 1) {
                    for (c, t) in classes.iter().zip(traces_buf.outer_iter()) {
                        if ext_bit(*c, i) == 0 {
                            xty.slice_mut(s![i, ..]).sub_assign(&t);
                        } else {
                            xty.slice_mut(s![i, ..]).add_assign(&t);
                        }
                    }
                }
            }
        );
    }

    pub fn solve(&mut self) -> Array3<f64> {
        // result of the linear regression
        
        //let mut reg_coefs = Array2::<f64>::zeros((self.nb + 1, self.ns));
        
        let mut projections = Array3::zeros((self.nv, self.p, self.ns));
        Zip::indexed(self.xty.outer_iter_mut())
            .and(self.norm_proj.outer_iter_mut())
            .and(self.proj_coefs.outer_iter_mut())
            .and(self.mu_chunks.outer_iter_mut())
            .and(projections.outer_iter_mut())
            .into_par_iter().
            for_each_init(
                || {return Array2::zeros((self.nb + 1, self.ns))},
                |reg_coefs,(k,xty,mut norm_proj, mut proj_coefs, mut mu_chunks,mut proj)| {
                    // Compute linear regression
                    reg_coefs.view_mut().assign(&xty);
                    let xtx = self.xtx.slice(s![k,..,..]);
                    let xtx_nalgebra = xtx.into_nalgebra();
                    let cholesky = xtx_nalgebra
                        .cholesky()
                        .expect("Failed Cholesky decomposition.");
                    cholesky.solve_mut(&mut reg_coefs.view_mut().into_nalgebra());
                    // Between class scatter for LDA
                    // Original LDA: sb = sum_{traces} (trace-mu)*(trace-mu)^T
                    // here, we replace trace with the model coefs^T*b and we get
                    //     mu = 1/ntraces * sum_{b} coefs^T*b
                    //        = 1/ntraces * coefs^T * sum_{b} b
                    //        = 1/ntraces * coefs^T * xtx[0,..] (since b[0] = 1.0 always)
                    // Therefore, the scatter is
                    //     s_b = sum_{b} (coef^T*b)*(coef^T*b)^T - ntraces*mu*mu^T
                    //         = s_m - ntraces*mu*mu^T
                    // where we define the model scatter as
                    //     s_m = sum_{b} (coef^T*b)*(coef^T*b)^T
                    //         = coef^T * [sum_{b} b*b^T] * coef
                    //         = coef^T * (self.xtx) * coef
                    let nt_mu: Array1<f64> = xtx.slice(s![0usize, ..]).dot(reg_coefs);
                    let mu = nt_mu / self.n as f64;
                    let s_m = reg_coefs
                        .t()
                        .dot(&xtx)
                        .dot(reg_coefs);
                    let s_b =
                        &s_m - (self.n as f64) * mu.slice(s![.., NewAxis,]).dot(&mu.slice(s![NewAxis, ..]));
                    // Dimentionality reduction (LDA part)
                    // The idea is to solve the generalized eigenproblem (l,w)
                    //     s_b*w = l*s_w*w
                    // where s_b is the between-classes scatter matrix computed above
                    // and s_w is the within-class scatter matrix, in our case it is the scatter of the
                    // residual trace-model, where model=coefs^T*b.
                    //     sw
                    //     = sum_{trace} (trace-coefs^T*b)*(trace-coefs^T*b)^T
                    //     = sum_{trace} trace*trace^T - (coefs^T*b)*(coefs^T*b)^T
                    //     = s_t - s_m
                    //     (s_t is self.scatter)
                    let s_w = &self.scatter + s_m - &xty.t().dot(reg_coefs) - &reg_coefs.t().dot(&xty);

                    let solver = geigen::GEigenSolverP::new(&s_b.view(), &s_w.view(), self.p)
                        .expect("failed to solve");
                    let projection = solver.vecs().t().into_owned();

                    proj.assign(&projection);

                    // Now we can project traces, and projecting the coefs gives us a
                    // reduced-dimensionality model.
                    // The projection does not guarantee that the scatter of the new residual is unitary
                    // (we'd like it to be for later simplicity), hence a apply a rotation.
                    // The new residual is projection*(trace-coefs^T*b), hence its scatter is
                    // projection*s_w*projection^T
                    let cov_proj_res = projection.view().dot(&s_w).dot(&projection.t()) / (self.n as f64);
                    // We decompose cov_proj_res N as N = V*W*V^T where V is orthonormal and W diagonal
                    // then if we re-project with W^-1/2*V^T, we get an identity covariance.
                    let nalgebra::linalg::SymmetricEigen {
                        eigenvectors,
                        eigenvalues,
                    } = nalgebra::linalg::SymmetricEigen::new(cov_proj_res.into_nalgebra());
                    let mut evals = eigenvalues.into_ndarray1();
                    let evecs = eigenvectors.into_ndarray2();
                    evals.mapv_inplace(|v| 1.0 / v.sqrt());
                    let normalizing_proj_t = evecs * evals.slice(s![.., NewAxis]);
                    // Storing projections and projected coefficients
                    norm_proj.assign(&normalizing_proj_t.t().dot(&projection));
                    proj_coefs.assign(&norm_proj.dot(&reg_coefs.t()));

                    //Precomputing lower parts the mus and storing them.
                    for data_l in 0..min(SIZE_CHUNK,1<<self.nb as usize) {
                        for chunk in 0..(self.nb + NBITS_CHUNK-1)/NBITS_CHUNK as usize {
                            let mut mu_l = mu_chunks.slice_mut(s![chunk,data_l,..]);
                            mu_l.fill(0.0); // Clear chunk in case multiple calls to solve
                            if chunk==0 {
                                for j in 0..self.p {
                                    mu_l[j] += int2mul(ext_bit(data_l as u64,0))*proj_coefs[[j,0]];    
                                }
                            }
                            for i in (chunk*NBITS_CHUNK)..(min(self.nb,(chunk+1)*NBITS_CHUNK)) {
                                for j in 0..self.p {
                                    mu_l[j] += int2mul(ext_bit((data_l<<(chunk*NBITS_CHUNK)) as u64,i + 1))*proj_coefs[[j,i+1]];
                                }
                            }
                        }
                    }
                }
            );
        return projections;
    }

    pub fn get_clustered_model(
        &self,
        var_id: usize,
        store_associated_classes: bool, //TODO
        store_marginalized_weights: bool,
        max_squared_distance: f64,
        max_cluster_number: u32,
    ) -> RLDAClusteredModel {
        let marginalized_weights = store_marginalized_weights.then(|| Array2::zeros((0, self.nb)));
        let mut clustered_model = RLDAClusteredModel {
            kdtree: KdTree::new(self.p),
            coefs: self.proj_coefs.slice(s![var_id,..,..]).into_owned(),
            norm_proj: self.norm_proj.slice(s![var_id,..,..]).into_owned(),
            mu_chunks: self.mu_chunks.slice(s![var_id,..,..,..]).into_owned(),
            max_cluster_number: max_cluster_number,
            ncentroids: 0 as u32,
            centroid_ids: Vec::new(),
            centroid_weights: Vec::new(),
            centroid_weights_and: Vec::new(),
            associated_centroids: Vec::new(),
            marginalized_weights,
            max_squared_distance,
            store_associated_classes,
        };
        clustered_model.initialize();
        return clustered_model;
    }
  
    /// return the probability of each of the possible value for leakage samples
    /// x : traces with shape (n,ns)
    /// v : index of variable that we want to get the probabilities
    /// return prs with shape (n,2**nb). Every row corresponds to one probability distribution
    pub fn predict_proba(&self, x: ArrayView2<i16>, v: usize) -> Array2<f64> {

        fn softmax(mut v: ndarray::ArrayViewMut1<f64>) {
            //let max = v.fold(f64::NEG_INFINITY, |x, y| f64::max(x, *y));
            v.par_mapv_inplace(|x| f64::exp(x));
            
            let tot: f64 = Zip::from(v.view()).par_fold(|| 0.0,
                |acc,s| acc+*s,
                |sum, other_sum| sum + other_sum,);
            v.into_par_iter().for_each(|s| *s/=tot);
        }
        
        let x = x.mapv(|x| x as f64).dot(&self.norm_proj.slice(s![v,..,..]).t());
        let mut scores : Array2<f64>= Array2::zeros((x.len_of(Axis(0)), 1<<self.nb));

        /*for t in 0..x.len_of(Axis(0)) {
            Zip::from(scores.index_axis_mut(Axis(0), t))
                .par_for_each(| x:&mut f64| *x = 0.0 as f64);
        }*/

        for t in 0..x.len_of(Axis(0)) {
            Zip::from(scores.index_axis_mut(Axis(0), t))
                .par_for_each(| x:&mut f64| *x = 0.0 as f64);
        }

        Zip::from(scores.outer_iter_mut()).and(x.outer_iter()).for_each(|mut scores_trace,trace| {
            if self.nb < 8 {
                Zip::indexed(scores_trace)
                .into_par_iter()
                .for_each_init( || { return Array1::zeros(self.p) },
                    |tmp_mu, (i, score)| {
                        // iter over MSBs
                        for j in 0..self.p {
                            tmp_mu[j] = trace[[j]];
                        }
                        let mut acc: f64;
                        for j in 0..self.p{
                            acc = tmp_mu[j] - self.mu_chunks[[v,0,i,j]];
                            *score += -0.5*acc*acc;
                        }
                    }
                );
            } else {
            Zip::indexed(scores_trace
                    .exact_chunks_mut(SIZE_CHUNK))
                    .into_par_iter()
                    .for_each_init( || { return Array1::zeros(self.p) },
                    |tmp_mu, (i, mut score)| {
                        // iter over MSBs
                        for j in 0..self.p{
                            tmp_mu[j] = trace[[j]];
                            for chunk in 0..((self.nb + NBITS_CHUNK-1)/NBITS_CHUNK - 1) as usize {
                                //iterate over chunks except smallest
                                let i_chunk = (i>>(chunk*NBITS_CHUNK)) & (SIZE_CHUNK-1);
                                tmp_mu[j] -= self.mu_chunks[[v,chunk+1,i_chunk,j]];
                            }
                        }
                        for i_lsb in 0..SIZE_CHUNK {
                            let mut acc: f64;
                            for j in 0..self.p{
                                acc = tmp_mu[j] - self.mu_chunks[[v,0,i_lsb,j]];
                                score[[i_lsb]] += -0.5*acc*acc;
                            }
                        }
                    }
                );
            }
        });

        for score_distr in scores.outer_iter_mut() {
            softmax(score_distr);
        }
        return scores;
    }
}



#[derive(Serialize, Deserialize)]
pub struct RLDAClusteredModel {
    pub kdtree: KdTree<f64, usize, Vec<f64>>,
    pub coefs: Array2<f64>,
    pub norm_proj: Array2<f64>,
    pub mu_chunks: Array3<f64>,
    pub max_cluster_number: u32,
    pub ncentroids: u32,
    pub centroid_ids: Vec<u64>,
    pub centroid_weights: Vec<u32>,
    pub centroid_weights_and: Vec<f64>,
    pub associated_centroids: Vec<Vec<u32>>,
    /// Shape (ncentroids, nbits)
    pub marginalized_weights: Option<Array2<u32>>,
    pub max_squared_distance: f64,
    pub store_associated_classes: bool,
}
impl RLDAClusteredModel {
    pub fn nearest(&self, point: &[f64]) -> (usize, f64) {
        let res = self.kdtree.nearest(point, 1, &squared_euclidean).unwrap();
        if res.len() == 0 {
            (0usize, f64::INFINITY)
        } else {
            let (sq_distance, centroid_id) = res[0];
            (*centroid_id, sq_distance)
        }
    }

    fn initialize(&mut self) {
        let ndims = self.coefs.shape()[0];
        let nbits = self.coefs.shape()[1] as u32;
        let mut centroid = vec![0f64; ndims];
        let mut p_hw_and = vec![0f64; nbits as usize];
        for i in 0..nbits {
            p_hw_and[i as usize] =
                0.25_f64.powf(i.into()) * 0.75_f64.powf((nbits - 1 - i).into());
        }
        let mut hw: f64;
        for i in 0..2u64.pow(nbits - 1) {
            centroid.fill(0.0);
            hw = 0.0;
            for b in 0..nbits {
                //let sign:f64 = if (i>>b)&1==1 || b==nbits {1.0} else {-1.0};
                //let sign: f64 = if (i >> b) & 1 == 1 { 1.0 } else { -1.0 };
                let sign = int2mul(ext_bit(i,b as usize));
                if b>0 { //Don't do this for intercept
                    hw = hw + (sign + 1.0) / 2.0;
                }
                for (centroid, coef) in centroid.iter_mut().zip(
                    self.coefs
                        .slice(s!(.., b as usize))
                        .iter(),
                ) {
                    *centroid += sign * *coef;
                }
            }
            let nearest = self.nearest(centroid.as_slice());
            let c_id = if nearest.1 > self.max_squared_distance {
                self.kdtree
                    .add(centroid.clone(), self.centroid_ids.len())
                    .unwrap();
                self.centroid_ids.push(i);
                self.centroid_weights.push(0);
                self.centroid_weights_and.push(0.0);
                if self.store_associated_classes {
                    self.associated_centroids.push(Vec::new());
                }
                if let Some(mw) = self.marginalized_weights.as_mut() {
                    mw.push_row(Array1::zeros((nbits as usize,)).view()); //TODO READ CAREFULLY THIS CODE 
                }
                self.ncentroids += 1;
                (self.ncentroids - 1) as usize
            } else {
                nearest.0
            };
            self.centroid_weights[c_id] += 1;
            self.centroid_weights_and[c_id] += p_hw_and[hw as usize];
            if self.store_associated_classes {
                self.associated_centroids[c_id].push(i as u32);
            }
            if let Some(mw) = self.marginalized_weights.as_mut() {
                for j in 0..(nbits as usize) {
                    if (i >> j) & 0x1 == 1 {
                        mw[(c_id, j)] += 1;
                    }
                }
            }
            if self.ncentroids > self.max_cluster_number {
                break; // TODO :: break more properly : raise error and reset state 
            }
        }
    }

    pub fn get_size(&self) -> u32 {
        self.kdtree.size().try_into().unwrap()
    }

    pub fn get_max_dist(&self) -> f64 {
        self.max_squared_distance.try_into().unwrap()
    }

    pub fn get_tree(&self) -> Array1<u64> {
        let ndims = self.coefs.shape()[0];
        let point = vec![0f64; ndims];
        let res = self
            .kdtree
            .nearest(&point, self.centroid_ids.len(), &squared_euclidean)
            .unwrap();
        let mut centroid_ids = Array1::<u64>::zeros(self.centroid_ids.len());
        for i in 0..self.centroid_ids.len() {
            centroid_ids[i] = *res[i].1 as u64;
        }
        return centroid_ids;
    }

    pub fn get_weights(&self) -> (Array1<u64>, Array1<u32>, Array1<f64>) {
        (
            Array1::from_vec(self.centroid_ids.clone()),
            Array1::from_vec(self.centroid_weights.clone()),
            Array1::from_vec(self.centroid_weights_and.clone()),
        )
    }

    pub fn get_marginalized_weights(&self) -> Option<ArrayView2<u32>> {
        self.marginalized_weights.as_ref().map(|mw| mw.view())
    }

    pub fn get_close_clusters<'a, 'b, 's>(&'s self, point: &'a [f64], max_popped_classes:usize) -> impl Iterator<Item=(usize,usize)> + 'b where 'a: 'b, 's: 'b {//Vec<(usize,usize)>  {
        let mut n:usize = 0;
        self.kdtree
            .iter_nearest(point, &squared_euclidean)
            .unwrap()
            .map(| (_d, &c_id)| (c_id,self.associated_centroids[c_id].len()) )
            .take_while(move |(_c_id, n_associated) | {
                n+=n_associated;
                return n<max_popped_classes;
            })
    }

    /// return the probability of the correct class for leakage samples based on a set of centroids and their weights
    /// x : traces with shape (n,ns)
    /// v : index of variable that we want to get the probabilities
    /// values : array of the correct class for each trace. Has length n.
    /// return prs with shape (n,2**nb). Every row corresponds to one probability distribution
    pub fn clustered_prs(
        &self, x: ArrayView2<i16>,
        values: ArrayView1<u64>,
        max_popped_classes:usize,
    ) -> (
        Array1<f64>,
        Array1<f64>
    ) {
        let ndims = self.coefs.shape()[0];
        let nbits = self.coefs.shape()[1]-1;
        let x = x.mapv(|x| x as f64).dot(&self.norm_proj.t());
        let mut clustered_prs_lower : Array1<f64>= Array1::zeros(x.len_of(Axis(0)));
        let mut clustered_prs_upper : Array1<f64>= Array1::zeros(x.len_of(Axis(0)));
        let n_chunks: usize = (nbits + NBITS_CHUNK-1)/NBITS_CHUNK; 
        
        Zip::from(clustered_prs_lower.view_mut())
            .and(clustered_prs_upper.view_mut())
            .and(x.outer_iter())
            .par_for_each(|clustered_pr_lower, clustered_pr_upper,trace| {
                let mut close_clusters: Vec<usize> = self.get_close_clusters(trace.as_slice().unwrap(), max_popped_classes)
                                                            .map(|(c_id, _n_associated)| c_id)
                                                            .collect();
                close_clusters.sort();

                let denoms = Zip::indexed(&self.centroid_ids)
                    .and(&self.centroid_weights)
                    .par_fold(|| (0.0,0.0),
                    |mut denom,c_id, centroid,weight| {                        
                        
                        // Check if actual centroid_id is in close_clusters -> Enumerate on centroid ids and close_cluster.contains()
                        // If true : do calc prs on all elements of cluster
                        // Else : calc prs on centroid and *weight as before
                        //if close_clusters.contains(&c_id) {
                         if self.store_associated_classes && close_clusters.binary_search(&c_id).is_ok() {
                            let exact_denom:f64 = self.associated_centroids[c_id].iter().fold(0.0, 
                            |denom,&val| {
                                let mut exponent = 0.0;
                                for d in 0..ndims{
                                    let mut tmp = trace[[d]];
                                    for chunk in 0..n_chunks {
                                        let i_chunk = (val as usize>>(chunk*NBITS_CHUNK)) & (SIZE_CHUNK-1);
                                        tmp -=  self.mu_chunks[[chunk,i_chunk,d]]
                                    }
                                    exponent += tmp*tmp;    
                                }
                                denom + f64::exp(-0.5*exponent)
                            });
                            denom.0 += exact_denom;
                            denom.1 += exact_denom;
                        } else {
                            let mut exponent = 0.0;
                            for d in 0..ndims{
                                let mut tmp = trace[[d]];
                                for chunk in 0..n_chunks {
                                    let i_chunk = (*centroid as usize>>(chunk*NBITS_CHUNK)) & (SIZE_CHUNK-1);
                                    tmp -= self.mu_chunks[[chunk,i_chunk,d]]
                                }
                                exponent += tmp*tmp;    
                            }
                            let lower_bound = (f64::sqrt(exponent) + f64::sqrt(self.max_squared_distance)).powi(2);
                            let upper_bound = (f64::sqrt(exponent) - f64::sqrt(self.max_squared_distance)).max(0.0).powi(2);
                            denom.0 += f64::exp(-0.5*(lower_bound)) * (*weight as f64);
                            denom.1 += f64::exp(-0.5*(upper_bound)) * (*weight as f64);
                        }
                        return denom;
                    },
                |sum, other_sum| (sum.0+other_sum.0,sum.1+other_sum.1)
                );
                *clustered_pr_upper = denoms.0;
                *clustered_pr_lower = denoms.1;
            }
            );
        
        
        Zip::from(clustered_prs_lower.view_mut())
            .and(clustered_prs_upper.view_mut())
            .and(values)
            .and(x.outer_iter())
            .into_par_iter()
            .for_each_init(|| {return Array1::zeros(ndims)},
            |tmp_mu,(prs_l,prs_u,value,trace)| {
                for d in 0..ndims{
                    tmp_mu[[d]] = 0.0;
                }
                for chunk in 0..n_chunks {
                    let i_chunk = (*value as usize>>(chunk*NBITS_CHUNK)) & (SIZE_CHUNK-1);
                    for d in 0..ndims{
                        tmp_mu[[d]] += self.mu_chunks[[chunk,i_chunk,d]];
                    }
                }
                let associated_centroid = self.nearest(&tmp_mu.to_vec()).0;
                let mut pr_num = 0.0;
                
                for d in 0..ndims{
                    let tmp = trace[[d]] - tmp_mu[[d]];
                    pr_num += tmp*tmp;
                }
                pr_num = f64::exp(-0.5*pr_num);
                
                let mut pr_associated = 0.0;
                for d in 0..ndims{
                    tmp_mu[[d]] = 0.0;
                }
                for chunk in 0..n_chunks {
                    let i_chunk = ( associated_centroid>>(chunk*NBITS_CHUNK)) & (SIZE_CHUNK-1);
                    for d in 0..ndims{
                        tmp_mu[[d]] += self.mu_chunks[[chunk,i_chunk,d]];
                    }
                }
                for d in 0..ndims{
                    let tmp = trace[[d]] - tmp_mu[[d]];
                    pr_associated += tmp*tmp;
                }
                pr_associated = f64::exp(-0.5*pr_associated); //TODO : READ carefully this code again
                
                //prs contains clustered denominator for now.
                //*prs = pr_num/(*prs+pr_num-pr_associated);
                *prs_l = pr_num/ *prs_l;
                *prs_u = pr_num/ *prs_u;
            });
        return (clustered_prs_lower,clustered_prs_upper);
    }
}