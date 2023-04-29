import numpy as np
from scalib import _scalib_ext
from scalib.config import get_config

class RLDAClassifier:
    def __init__(self,nb,ns,nv,p):
        self._nb = nb
        self._ns = ns
        self._nv = nv
        self._p = p
       
        self._nt = 0
        self._inner = _scalib_ext.RLDA(nb,ns,nv,p)
        self._solved = False

    def fit_u(self,l,x,gemm_mode=1):
        r"""Update statistical model estimates with fresh data.

        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        x : array_like, uint64
            Labels for each trace. Must be of shape `(nv,n)` and
            must be `uint64`.
        """
        assert l.shape[0]==x.shape[1]
        assert l.shape[1]==self._ns
        assert x.shape[0]==self._nv

        self._nt += l.shape[0]
        self._inner.update(l,x,gemm_mode,get_config())

    def solve(self):
        self._inner.solve(get_config())
        self._solved = True

    def get_proj(self):
        return self._inner.get_norm_proj()

    def get_proj_coefs(self):
        return self._inner.get_proj_coefs()

    def get_mu_chunks(self):
        return self._inner.get_mu_chunks()

    def predict_proba(self,traces,var):
        return self._inner.predict_proba_with_nparray(traces,var,get_config())

    def get_clustered_model(self,var,t,max_cluster_number,store_associated_classes=True,store_marginalized_weights=False):
        return self._inner.get_clustered_model(var,store_associated_classes,store_marginalized_weights,t,max_cluster_number)