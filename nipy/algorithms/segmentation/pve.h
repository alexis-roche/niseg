#ifndef PVE
#define PVE

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef NO_APPEND_FORTRAN
# define FNAME(x) x
#else
# define FNAME(x) x##_
#endif 

  extern void pve_import_array(void);

  extern int __linsolve(double* A, double* b, int* tmp, int n, void* dgesv_ptr);
  extern void __locsum(double* res, double* degree, const double* pvm,
		       int x, int y, int z,
		       int dimx, int dimy, int dimz, int dimk,
		       int ngb_size);
  extern int* __generate_bmaps(int n, int* nmaps);
  extern void __quadsimplex(const double* A, const double* b, int n,
			    double* x, int* bmaps, int nbmaps,
			    int* I, double* AI, double* bI, double* cI,
			    double* AI_cp, int* tmp, void* dgesv_ptr);

  extern void _quadsimplex(const PyArrayObject* A, const PyArrayObject* b, void* dgesv_ptr);


#ifdef __cplusplus
}
#endif

#endif
