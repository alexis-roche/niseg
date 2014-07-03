# -*- Mode: Python -*-  

"""
Markov random field utils. 

Author: Alexis Roche, 2010.
"""

__version__ = '0.2'

# Includes
import numpy as np
cimport numpy as np
from scipy.linalg.lapack import flapack

# Externals
cdef extern from "Python.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef extern from "mrf.h":
    void mrf_import_array()
    void ve_step(np.ndarray ppm, 
                 np.ndarray ref,
                 np.ndarray XYZ, 
                 np.ndarray U,
                 int ngb_size, 
                 double beta)
    np.ndarray make_edges(np.ndarray mask,
                       int ngb_size)
    double interaction_energy(np.ndarray ppm, 
                              np.ndarray XYZ,
                              np.ndarray U,
                              int ngb_size)
cdef extern from "pve.h":
    void pve_import_array()
    int __linsolve(double* A, double* b, int* tmp, int n, void* dgesv_ptr)
    void __locsum(double* res, double* degree, double* pvm,
                  int x, int y, int z,
                  int dimx, int dimy, int dimz, int dimk,
                  int ngb_size)
    int* __generate_bmaps(int n, int* nmaps)
    void __quadsimplex(double* A, double* b, int n,
                       double* x, int* bmaps, int nbmaps,
                       int* I, double* AI, double* bI, double* cI,
                       double* AI_cp, int* tmp, void* dgesv_ptr)
    void _quadsimplex(np.ndarray A, np.ndarray b, void* dgesv_ptr)
cdef extern from "stdlib.h":
    void* calloc(int nmemb, int size)
    void* free(void* ptr) 
cdef extern from "math.h":
    double HUGE_VAL
    void* memcpy(void* dest, void* src, int n)
    void* memset(void* s, int c, int n)

# Initialize numpy
mrf_import_array()
pve_import_array()
np.import_array()



def _ve_step(ppm, ref, XYZ, U, int ngb_size, double beta):

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not ref.flags['C_CONTIGUOUS'] or not ref.dtype=='double':
        raise ValueError('ref array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='intp':
        raise ValueError('XYZ array should be intp C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')
    if not U.flags['C_CONTIGUOUS'] or not U.dtype=='double':
        raise ValueError('U array should be double C-contiguous')
    if not ppm.shape[-1] == ref.shape[-1]:
        raise ValueError('Inconsistent shapes for ppm and ref arrays')

    ve_step(<np.ndarray>ppm, <np.ndarray>ref, <np.ndarray>XYZ, <np.ndarray>U, 
             ngb_size, beta)
    return ppm 


def _make_edges(mask, int ngb_size):
    
    if not mask.flags['C_CONTIGUOUS'] or not mask.dtype=='intp':
        raise ValueError('mask array should be intp and C-contiguous')

    return make_edges(mask, ngb_size)


def _interaction_energy(ppm, XYZ, U, int ngb_size):

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='intp':
        raise ValueError('XYZ array should be intp C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')
    if not U.flags['C_CONTIGUOUS'] or not U.dtype=='double':
        raise ValueError('U array should be double C-contiguous')

    return interaction_energy(<np.ndarray>ppm, <np.ndarray>XYZ, <np.ndarray>U,
                               ngb_size)


cdef _simple_fitting_one_voxel(double y, double* q0, double* m, double* c,
                               np.npy_intp* idx, double* mat, np.npy_intp size_idx,
                               double* b, double* qe, double* aux):
    """
    Compute a feasible solution to the simplex fitting problem
    assuming that the positivity constraints on the elements given by
    `idx` are inactive.
    """
    cdef np.npy_intp i, j, idx_i, K = 3
    cdef double lda, mt_qe, summ, tmp
    cdef double *buf_mat

    # Compute qe = mat * b, aux = mat * 1, and the Lagrange multiplier
    # lda that matches the unit sum constraint
    memset(<void*>qe, 0, K * sizeof(double))
    memset(<void*>aux, 0, K * sizeof(double))
    lda = 0.0
    summ = 0.0
    buf_mat = mat
    for i from 0 <= i < size_idx:
        idx_i = idx[i]
        for j from 0 <= j < size_idx:
            tmp = buf_mat[0]
            qe[idx_i] += tmp * b[idx[j]]
            aux[idx_i] += tmp
            buf_mat += 1
        lda += qe[idx_i]
        summ += aux[idx_i]
    lda = (lda - 1)/ summ

    # Compute the candidate solution (replace negative components by
    # zero)
    summ = 0.0
    for i from 0 <= i < size_idx:
        idx_i = idx[i]
        tmp = qe[idx_i] - lda * aux[idx_i]
        if tmp < 0:
            tmp = 0
        qe[idx_i] = tmp
        summ += tmp

    # Replace qe with a uniform distribution for safety if all
    # components are small
    if summ < 1e-20:
        for i from 0 <= i < K:
            qe[i] = 1.0 / <double>K
            summ = 1.0

    # Renormalize qe and compute mt_qe = m' * qe and aux = qe - q0
    mt_qe = 0.0
    for i from 0 <= i < K:
        qe[i] /= summ
        tmp = qe[i]
        mt_qe += tmp * m[i]
        aux[i] = tmp - q0[i]

    # Compute (qe - q0)^t C (qe - q0)
    buf_mat = c
    summ = 0.0
    for i from 0 <= i < K:
        tmp = 0.0
        for j from 0 <= j < K:
            tmp += buf_mat[0] * aux[j] 
            buf_mat += 1
        summ += aux[i] * tmp 

    # Return the objective value
    tmp = y - mt_qe
    return tmp * tmp + summ


cdef simple_fitting_one_voxel(double* q, double y, double* q0, double* m, double* c,
                              inactives, solver, double* b, double* qe, double* aux):
    cdef np.npy_intp i, j, K = 3, Kbytes = K * sizeof(double), size_idx
    cdef np.flatiter itInactives, itSolver
    cdef np.npy_intp* idx
    cdef double *mat, *buf_mat
    cdef np.PyArrayObject *idx_ptr, *mat_ptr
    cdef double tmp, best = HUGE_VAL

    # Compute b = y * m + c * q0
    buf_mat = c
    for i from 0 <= i < K:
        tmp = y * m[i]
        for j from 0 <= j < K:
            tmp += buf_mat[0] * q0[j]
            buf_mat += 1
        b[i] = tmp

    # Evaluate each hypothesis regarding active inequality constraints
    itInactives = inactives.flat
    itSolver = solver.flat
    while np.PyArray_ITER_NOTDONE(itInactives):
        idx_ptr = (<np.PyArrayObject**> np.PyArray_ITER_DATA(itInactives))[0]
        idx = <np.npy_intp*>np.PyArray_DATA(<object> idx_ptr)
        mat_ptr = (<np.PyArrayObject**> np.PyArray_ITER_DATA(itSolver))[0]
        mat = <double*>np.PyArray_DATA(<object> mat_ptr)
        size_idx = np.PyArray_DIM(<object>idx_ptr, 0)
        tmp = _simple_fitting_one_voxel(y, q0, m, c, idx, mat, size_idx, b, qe, aux)
        if tmp < best:
            memcpy(<void*>q, <void*>qe, Kbytes)
            best = tmp
        np.PyArray_ITER_NEXT(itInactives)
        np.PyArray_ITER_NEXT(itSolver)


def simplex_fitting(Y, M, Q0, C):
    """
    simplex_fitting(y, m, q0, C)

    Find the vector on the standard simplex in dimension 3 that
    minimizes:
    
    (y - m^t * q)^2 + (q - q0)^t C (q - q0)
    
    where dimensions are:
      y : (N,)
      M : (3,)
      Q0: (N,3)
      C: (3,3)
    
    Parameter `q0` is modified in place.
    """
    cdef np.flatiter itY, itQ0
    cdef double *y, *q0
    cdef int axis = 1
    cdef double *m, *c, *b, *qe, *aux
    cdef K = 3, Kbytes = K * sizeof(double)

    if not Q0.flags['C_CONTIGUOUS'] or not Q0.dtype=='double':
        raise ValueError('q0 should be double C-contiguous')

    Y = np.asarray(Y, dtype='double')
    M = np.asarray(M, dtype='double', order='C')
    C = np.asarray(C, dtype='double', order='C')
    m = <double*>np.PyArray_DATA(M)
    c = <double*>np.PyArray_DATA(C)
    itY = Y.flat
    itQ0 = np.PyArray_IterAllButAxis(Q0, &axis)

    # Pre-compute matrices needed to solve for the Lagrangian
    # derivative's root for each possible set of inactive constraints.
    # Make sure the matrices are stored in row-major order as it's
    # assumed in sub-routines.
    inactives = [0,1,2], [0,1], [0,2], [1,2], [0], [1], [2]
    inactives = np.array([np.array(idx, dtype='intp') for idx in inactives])
    A = np.dot(M.reshape((K, 1)), M.reshape((1, K))) + C
    solver = []
    for idx in inactives:
        solver.append(np.asarray(np.linalg.inv(A[idx][:, idx]), order='C'))
    solver = np.array(solver)

    # Allocate auxiliary arrays
    b = <double*>calloc(K, sizeof(double))
    q = <double*>calloc(K, sizeof(double))
    qe = <double*>calloc(K, sizeof(double))
    aux = <double*>calloc(K, sizeof(double))

    while np.PyArray_ITER_NOTDONE(itY):
        y = <double*>(np.PyArray_ITER_DATA(itY))
        q0 = <double*>(np.PyArray_ITER_DATA(itQ0))
        simple_fitting_one_voxel(q, y[0], q0, m, c, inactives, solver, b, qe, aux)
        memcpy(<void*>q0, <void*>q, Kbytes)
        np.PyArray_ITER_NEXT(itY)
        np.PyArray_ITER_NEXT(itQ0)

    # Free auxiliary arrays
    free(b)
    free(q)
    free(qe)
    free(aux)


def linsolve(A, b):
    cdef int ok, n = np.PyArray_DIM(A, 0)
    cdef void* dgesv_ptr = PyCObject_AsVoidPtr(flapack.dgesv._cpointer)
    Ac = np.array(A, order='F')
    bc = np.array(b, order='F')
    tmp = <int*>calloc(n, sizeof(int))
    ok = __linsolve(<double*>np.PyArray_DATA(Ac),
                     <double*>np.PyArray_DATA(bc), 
                     tmp, n, dgesv_ptr)
    free(tmp)
    return bc


def quadsimplex(A, b):
    cdef void* dgesv_ptr = PyCObject_AsVoidPtr(flapack.dgesv._cpointer)
    _quadsimplex(A, b, dgesv_ptr)
    return b


def update_cmap(CM, DATA, XYZ, MU, s2, alpha, double beta, int ngb_size):
    """
    update_cmap(cm, data, XYZ, mu, s2, beta, V, ngb_size)

    where dimensions are:
      cm: (dimx,dimy,dimz,nclasses)
      data: (N,)
      XYZ: (N,3)
      mu: (nclasses,)
      s2: (nchannels,)
      alpha: (nclasses,nclasses)
      beta: float

    Parameter cm is modified in place.
    """
    cdef double *data, *cm, *_cm, *q, *buf
    cdef double *A, *b, *precA, *precB, *AI, *bI, *cI, *AI_cp
    cdef int N, dimx, dimy, dimz, stx, sty, i, j, k
    cdef int nchannels = 1, nclasses, nclasses_bytes, sqr_nclasses_bytes, nbmaps, axis = 1
    cdef int *bmaps, *I, *tmp
    cdef np.npy_intp* xyz
    cdef double aux, degree, two_beta = 2*beta
    cdef void* dgesv_ptr = PyCObject_AsVoidPtr(flapack.dgesv._cpointer)

    # Test input compliance and reformat if needed
    DATA = np.asarray(DATA, dtype='double', order='C')
    N = DATA.shape[0]
    if len(DATA.shape) > 1:
        nchannels = DATA.shape[1]
    if not CM.ndim == 4 or not CM.flags['C_CONTIGUOUS']\
            or not CM.dtype=='double':
        raise ValueError('cm should be 4D, double and C-contiguous')
    if not XYZ.ndim == 2 or not XYZ.shape[1] == 3\
            or not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='intp':
        raise ValueError('XYZ array should be intp C-contiguous')

    # Compute dimensional parameters
    dimx, dimy, dimz, nclasses = CM.shape
    sty = dimz*nclasses
    stx = dimy*sty
    nclasses_bytes = nclasses*sizeof(double)
    sqr_nclasses_bytes = nclasses*nclasses*sizeof(double)

    # Reshape arrays
    DATA = np.reshape(DATA, (N, nchannels))
    MU = np.reshape(np.asarray(MU), (nclasses, nchannels))

    # Pre-compute auxiliary arrays
    # precA: nclasses x nclasses matrix 
    # precB: nclasses x nchannels matrix 
    PrecB = -MU/s2
    PrecA = np.dot(-PrecB, MU.T) + np.asarray(alpha)
    precA = <double*>np.PyArray_DATA(PrecA)  
    precB = <double*>np.PyArray_DATA(PrecB)

    # Generate mappings from [1,2,...,n] to {0,1} for the active set
    # quadratic programming method
    bmaps = __generate_bmaps(nclasses, &nbmaps)

    # Allocate auxiliary arrays
    q = <double*>calloc(nclasses, sizeof(double))
    A = <double*>calloc(nclasses*nclasses, sizeof(double))
    b = <double*>calloc(nclasses, sizeof(double))
    I = <int*>calloc(nclasses, sizeof(int))
    AI = <double*>calloc(nclasses*nclasses, sizeof(double))
    bI = <double*>calloc(nclasses, sizeof(double))
    cI = <double*>calloc(nclasses, sizeof(double))
    AI_cp = <double*>calloc(nclasses*nclasses, sizeof(double))
    tmp = <int*>calloc(nclasses, sizeof(int))

    # Loop over the image array
    itXYZ = np.PyArray_IterAllButAxis(XYZ, &axis)
    itDATA = np.PyArray_IterAllButAxis(DATA, &axis)
    cm = <double*>np.PyArray_DATA(CM)
    while np.PyArray_ITER_NOTDONE(itDATA):
        
        # Get voxel coordinates
        xyz = <np.npy_intp*>(np.PyArray_ITER_DATA(itXYZ))
        
        # Compute local sum of concentration vectors --> q
        if beta > 0:
            __locsum(q, &degree, cm, xyz[0], xyz[1], xyz[2],
                     dimx, dimy, dimz, nclasses, ngb_size)

        # Get image intensity
        data = <double*>(np.PyArray_ITER_DATA(itDATA)) 

        # Assemble matrix A and vector b at current voxel
        # A is obtained by adding 2*beta*degree to the diagonal of precA
        # b is obtained by computing the dot product precB*Y and
        # subtracting 2*beta times q
        memcpy(<void*>A, <void*>precA, sqr_nclasses_bytes)
        buf = precB
        for i from 0 <= i < nclasses:
            aux = 0.0
            for j from 0 <= j < nchannels:
                aux += buf[0] * data[j]
                buf += 1
            b[i] = aux - two_beta*q[i]
            k = i * (nclasses + 1)
            A[k] = A[k] + two_beta*degree

        # Solve quadratic simplex programming --> q
        __quadsimplex(A, b, nclasses, q, bmaps, nbmaps, I, AI, bI, cI, AI_cp, tmp, dgesv_ptr)

        # Copy result back into cm
        _cm = cm + xyz[0]*stx + xyz[1]*sty + xyz[2]*nclasses
        memcpy(<void*>_cm, <void*>q, nclasses_bytes)

        # Update iterators
        np.PyArray_ITER_NEXT(itDATA)
        np.PyArray_ITER_NEXT(itXYZ)

    # Free auxiliary arrays
    free(q)
    free(A)
    free(b)
    free(I)
    free(AI)
    free(bI)
    free(cI)
    free(AI_cp)
    free(tmp)
    free(bmaps)


def generate_bmaps(int n):
    cdef int i, nmaps
    cdef int* bmaps
    cdef np.npy_intp* data
    bmaps = __generate_bmaps(n, &nmaps)
    ret = np.zeros((nmaps, n), dtype='intp')
    data = <np.npy_intp*>np.PyArray_DATA(ret)
    for i from 0 <= i < n*nmaps:
        data[i] = bmaps[i]
    free(bmaps)
    return ret
