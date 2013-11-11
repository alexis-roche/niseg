#include "pve.h"

#include <math.h>
#include <stdlib.h>

#ifdef _MSC_VER
#define inline __inline
#endif

#define TINY 1e-50
#define ABS(a) ( (a) > 0.0 ? (a) : (-(a)) )

/* Lapack routine to solve a linear system using LU decomposition */
extern int FNAME(dgesv)(int* n, int* m, double* a, int* lda,
			int* ipiv, double* b, int* ldb, int* info);

/* Numpy import */
void pve_import_array(void) { 
  import_array(); 
  return;
}


/* Encode neighborhood systems using static arrays */
int N6 [] = {1,0,0,
	     -1,0,0,
	     0,1,0,
	     0,-1,0,
	     0,0,1,
	     0,0,-1}; 

int N26 [] = {1,0,0,
	      -1,0,0,
	      0,1,0,
	      0,-1,0,
	      1,1,0,
	      -1,-1,0,
	      1,-1,0,
	      -1,1,0, 
	      1,0,1,
	      -1,0,1,
	      0,1,1,
	      0,-1,1, 
	      1,1,1,
	      -1,-1,1,
	      1,-1,1,
	      -1,1,1, 
	      1,0,-1,
	      -1,0,-1,
	      0,1,-1,
	      0,-1,-1, 
	      1,1,-1,
	      -1,-1,-1,
	      1,-1,-1,
	      -1,1,-1, 
	      0,0,1,
	      0,0,-1}; 


/*
  Given a 3D concentration map cm represented by a 4D array with
  tissue class as last axis, compute the vector representing the sum
  of concentrations in a neighborhood around location x, y, z - where
  the neighborhood is defined by ngb_size (either 6 or 26).

  Store the result in pre-allocated vector res. The sum of weights
  over the neighborhood is also returned in weight.
 */
void __locsum(double* res,
	      double* degree,
	      const double* cm,
	      int x,
	      int y, 
	      int z,
	      int dimx,
	      int dimy,
	      int dimz,
	      int dimk,
	      int ngb_size)
{
  int xn, yn, zn, pos, ngb_idx, k, count;
  const int* ngb;
  double *_res, *_cm;
  size_t sty = dimz*dimk;
  size_t stx = dimy*sty;
  size_t posmax = dimx*stx - dimk;

  /* Neighborhood type */
  if (ngb_size == 6)
    ngb = N6;
  else if (ngb_size == 26)
    ngb = N26;
  else
    return;

  /*  Re-initialize output array */
  memset((void*)res, 0, dimk*sizeof(double));
  count = 0;

  /* Loop over neighbors */
  for (ngb_idx=0; ngb_idx<ngb_size; ngb_idx++) {
    xn = x + *ngb; ngb++; 
    yn = y + *ngb; ngb++;
    zn = z + *ngb; ngb++;
    pos = xn*stx + yn*sty + zn*dimk;

    /* Ignore neighbor if outside the grid boundaries */
    if ((pos < 0) || (pos > posmax))
      continue; 
    count ++;

    /* Update sum of concentrations */
    _res = res;
    _cm = (double*)cm + pos;
    for (k=0; k<dimk; k++, _res++, _cm++)
      *_res += *_cm;
  }

  *degree = (double)count;

  return; 
}



/*
  Solve the linear system:  Ax = b

  for a square F-contiguous matrix A with dimension (n, n).

  b is a contiguous one-dimensional arrays with dimension n, where
  the result is overwritten.

  tmp is an auxiliary array of int with length n used by Lapack to
  store intermediate calculations.  */
int __linsolve(double* A, double* b, int* tmp, int n)
{
  int info;
  int m = 1;

  FNAME(dgesv)(&n, &m, A, &n, tmp, b, &n, &info);

  return info;
}


/*
  Generate all mappings from {1, 2, ..., n} to {0, 1} excluding the
  constant assignment to zero. The number of such mappings is 2^n-1.

  This function dynamically allocates and returns an array of size
  (2^n-1)*n.
  
 */
int* __generate_bmaps(int n, int* nmaps)
{
  int i, j, magic;
  int *buf, *maps = NULL;

  if (n < 1) {
    *nmaps = 0;
    return NULL;
  }
  *nmaps = 1;
  for (j=0; j<n; j++)
    *nmaps *= 2;
  *nmaps -= 1;
  maps = (int*)calloc(n*(*nmaps), sizeof(int));

  for (i=0, buf=maps; i<(*nmaps); i++) {
    magic = i+1;
    for(j=0; j<n; j++, buf++) {
      *buf = magic % 2;
      magic /= 2;
    }
  }

  return maps;
}

/*
  Solve the quadratic programming problem:

  min_x [(1/2) x^t A x + b^t x]

  where A is a nxn symmetric positive-definite matrix, b is a nx1
  vector, and x is searched on the simplex, i.e. x >= 0 and sum_xi =
  1.  
*/

void __quadsimplex(const double* A, 
		   const double* b,
		   int n,
		   double* x,
		   int* bmaps,
		   int nbmaps,
		   int* I,
		   double* AI,
		   double* bI,
		   double* cI,
		   double* AI_cp,
		   int* tmp)
{
  double *_AI, *_bI, *_cI;
  double aux, alpha, dot_bbI, dot_cbI, dot_ccI, cost, best = HUGE_VAL;
  int m, i, j, nI, offset, unfeasible;
  int *_bmaps, *_I, *__I;

  /* Loop over binary maps */
  for (m=0, _bmaps=bmaps; m<nbmaps; m++) {

    /* Fill the array I of indices corresponding to assumed inactive
       constraints */
    nI = 0;
    for (i=0, _I=I; i<n; i++, _bmaps++) 
      if (*_bmaps > 0) {
	*_I = i;
	_I ++;
	nI ++;
      }

    /* Fill the reduced arrays AI and bI of respective sizes nIxnI and
       nI */
    for (i=0, _I=I, _bI=bI, _AI=AI; i<nI; i++, _I++, _bI++) {
      *_bI = b[*_I];
      offset = n*(*_I);
      for (j=0, __I=I; j<nI; j++, __I++, _AI++) {
	*_AI = A[offset + *__I];
      }
    }
    
    /* Solve AI*x = bI and store the result in bI */
    memcpy((void*)AI_cp, (void*)AI, nI*nI*sizeof(double));
    memcpy((void*)cI, (void*)bI, nI*sizeof(double));
    __linsolve(AI_cp, bI, tmp, nI);

    /* Compute the dot product bI^T AI^-1 bI */
    for (i=0, dot_bbI=0, _bI=bI, _cI=cI; i<nI; i++, _bI++, _cI++)
      dot_bbI += (*_bI)*(*_cI);

    /* Solve AI*x = 1 and store the result in cI */
    for (i=0, _cI=cI; i<nI; i++, _cI++)
      *_cI = 1;
    __linsolve(AI, cI, tmp, nI);

    /* Compute the dot products: cI^T AI^-1 bI and cI^T AI^-1 cI */
    for (i=0, _bI=bI, _cI=cI, dot_cbI=0, dot_ccI=0; i<nI; i++, _bI++, _cI++) {
      dot_cbI += *_bI;
      dot_ccI += *_cI;
    }

    /* Compute tentative solution: ((1+dot_cbI)/dot_ccI)*cI - bI and
       store result in bI. If some components are negative, the
       solution is unfeasible and is assigned an infinite cost for
       immediate rejection. Note that we are testing for primal
       feasibility, but not dual feasibility (positiveness of the
       Lagrange multipliers associated with active inequality
       constraints). Tentative solutions that are primal feasible but
       not dual feasible will be rejected later on as they yield
       larger cost than the unique primal/dual feasible solution. */
    if (ABS(dot_ccI) < TINY)
      unfeasible = 1;
    else {
      unfeasible = 0;
      alpha = (1 + dot_cbI)/dot_ccI; 
      for (i=0, _bI=bI, _cI=cI; i<nI; i++, _bI++, _cI++) {
	aux = alpha*(*_cI) - *_bI;
	if (aux < 0) {
	  unfeasible = 1;
	  break;
	}
	*_bI = aux;
      }
    }
    
    /* Criterion value at proposal */
    if (unfeasible)
      cost = HUGE_VAL;
    else
      cost = .5*(alpha*alpha*dot_ccI - dot_bbI);

    /* Update current solution if proposal is better */
    if (cost < best) {
      best = cost;
      memset((void*)x, 0, n*sizeof(double));
      for (i=0, _bI=bI, _I=I; i<nI; i++, _bI++, _I++)
	x[*_I] = *_bI;
    }
  }

  return;
}

void _quadsimplex(const PyArrayObject* PyA, const PyArrayObject* Pyb)
{
  int nbmaps, n = PyA->dimensions[0];
  int *bmaps, *I, *tmp;
  double *x, *AI, *bI, *cI, *AI_cp;
  double *A = (double*)PyArray_DATA(PyA);
  double *b = (double*)PyArray_DATA(Pyb);

  /* Allocate auxiliary arrays */
  I = (int*)calloc(n, sizeof(int));
  x = (double*)calloc(n, sizeof(double));
  AI = (double*)calloc(n*n, sizeof(double));
  bI = (double*)calloc(n, sizeof(double));
  cI = (double*)calloc(n, sizeof(double));
  AI_cp = (double*)calloc(n*n, sizeof(double));
  tmp = (int*)calloc(n, sizeof(int));

  /* Generate all mappings from [1,2,...,n] to {0,1} */
  bmaps = __generate_bmaps(n, &nbmaps);

  /* Solve quadratic simplex programming */
  __quadsimplex(A, b, n, x, bmaps, nbmaps, I, AI, bI, cI, AI_cp, tmp);
  memcpy((void*)b, (void*)x, n*sizeof(double));

  /* Free memory */
  free(I);
  free(x);
  free(AI);
  free(bI);
  free(cI);
  free(AI_cp);
  free(tmp);
  free(bmaps);

  return;
}
