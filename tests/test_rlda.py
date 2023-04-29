import numpy as np
from scalib.modeling import RLDAClassifier
from scipy.linalg import eigh
from scalib.config import get_config

from scalib.metrics import Information

def get_bits(data,dtype=None,nbits = None,intercept = True):
        if dtype==None:
            dtype = data.dtype
        assert data.ndim == 1 or data.ndim == 2
        if data.ndim == 1:
            ntraces = len(data)
            nwords = 1
        elif data.ndim == 2:
            ntraces,nwords = data.shape
        #Not necessary but to be coherent with MCUs that have different endianness
        data = data.astype(dtype)
        data = data.byteswap()
        data = np.frombuffer(data,np.uint8)
        bytes_per_trace = len(data)//ntraces
        data = np.reshape(data,(ntraces,bytes_per_trace))
        data = np.unpackbits(data,axis=1)
        dtype = np.int8 # for -1 1 coefs
        data = data.astype(dtype)
        coefs=[-1,1]
        where_0 = data==0
        where_1 = data==1
        data[where_0] = coefs[0]
        data[where_1] = coefs[1]
        if nbits==None:
            nbits = bytes_per_trace*8
        assert nbits <= bytes_per_trace*8
        if intercept:
            data = np.hstack((data[:,bytes_per_trace*8-nbits:],np.ones((ntraces,1),dtype)))
        else:
            data = data[:,bytes_per_trace*8-nbits:]
        return np.flip(data,axis=1)


if True:
    ns = 5
    n_components = 3
    nb=12
    nc = 2**nb
    n = 2000
    nv=1
    noise=30

    m = np.ones((nc,ns),dtype=np.int16)
    m[:,0]= np.arange(nc)
    traces = np.random.randint(-noise, noise, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, (nv,n), dtype=np.uint64)
    traces += m[labels][0].astype(np.int16)

    #Python ref
    
    label_bits = get_bits(labels.T,nbits=nb).astype(np.int64)
    ref_coefs,_,_,_ = np.linalg.lstsq(label_bits,traces,rcond=None)
    ref_mus = get_bits(np.arange(nc,dtype=np.uint64),nbits=nb)@ref_coefs

    B = label_bits.T@label_bits
    C = label_bits.T@traces
    S_L = traces.astype(np.int64).T@traces.astype(np.int64)

    A = np.linalg.solve(B,C)
    mu =  C[0]/n
    
    SB = A.T@B@A - n*np.outer(mu,mu)
    SW = S_L  + A.T@B@A - C.T@A - A.T@C
    
    #eigenvectors are different than in scalib... :/
    ev,W = eigh(SB,SW,eigvals=(ns-n_components,ns-1))
    W = W.T

    #Process data with scalib and compare projection
    rlda = RLDAClassifier(nb, ns,nv,n_components)
    rlda.fit_u(traces, labels, 1)
    
    proj = rlda._inner.solve(get_config())
    #check projections are close up to a sign
    assert np.allclose(np.abs(np.flip(W,0)),np.abs(proj),1e-7)
    #Continue test by using projection from scalib
    W = proj[0]
    cov = W@SW@W.T * 1/n

    evals,evecs = np.linalg.eig(cov)
    Wnorm = evecs*(evals[:,np.newaxis]**-0.5)

    Weff = W.T@Wnorm
    Aeff = A@Weff

    solved_state = rlda._inner.get_state()

    traces = np.random.randint(-noise, noise, (n, ns), dtype=np.int16)
    labels = np.random.randint(0, nc, n, dtype=np.uint16)
    traces += m[labels]

    #Use Rust coefficients for prs verification
    Aeff = solved_state[-2][0].T
    Weff = solved_state[-3][0].T

    prs_ref = (traces@Weff)[:,:,np.newaxis]-(get_bits(np.arange(2**nb,dtype=np.int64),nbits=nb)@Aeff).T[np.newaxis]
    prs_ref = np.exp(-0.5*(prs_ref**2).sum(axis=1))
    prs_ref = prs_ref/prs_ref.sum(axis=1,keepdims=True)

    prs = rlda.predict_proba(traces,0)
    
    assert np.allclose(prs,prs_ref)

    
    #test clustering
    cl = rlda._inner.get_clustered_model(0,True,False,0,2**16)
    prscl = cl.get_clustered_prs(traces,labels.astype(np.uint64),0,get_config())
    it = Information(cl,0)
    pi = it.fit_u(traces,labels.astype(np.uint64))
