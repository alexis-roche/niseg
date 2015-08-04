# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

from ._segmentation import _ve_step, _interaction_energy
from .moment_matching import moment_matching, matching_params

nonzero = lambda x: np.maximum(x, 1e-50)
log = lambda x: np.log(nonzero(x))


class VEM(object):

    def __init__(self, img, mask=None, mu=None, s2=1e-5, prop=None,
                 prior=None, U=None, ngb_size=6, beta=None,
                 usecase='brainT1'):
        """
        Class for multichannel Markov random field image segmentation
        using the variational EM algorithm. For details regarding the
        underlying algorithm, see:

        Roche et al, 2011. On the convergence of EM-like algorithms
        for image segmentation using Markov random fields. Medical
        Image Analysis (DOI: 10.1016/j.media.2011.05.002).

        Parameters
        ----------
        img : image-like
          Input image

        mask : array-like or tuple of arrays
          Input mask to restrict the segmentation. By default, the
          mask excludes voxels with zero intensities.

        beta : float
          Markov regularization parameter

        mu : array-like
          Initial class-specific means

        s2 : array-like
          Initial class-specific variances

        prop : array-like
          Initial class-specific proportions (uniform if None)
        """
        data = img.get_data().squeeze()
        if not len(data.shape) in (3, 4):
            raise ValueError('Invalid input image')
        if len(data.shape) == 3:
            nchannels = 1
            space_shape = data.shape
        else:
            nchannels = data.shape[-1]
            space_shape = data.shape[0:-1]

        self.nchannels = nchannels

        # Make default mask that removes zero intensities and fills
        # holes. This wil be passed to the _ve_step C-routine, which
        # assumes a contiguous int array and raise an error
        # otherwise. Voxels on the image borders are further rejected
        # to avoid segmentation faults.
        if mask == None:
            mask = binary_fill_holes(data > 0)
        X, Y, Z = np.where(mask)
        XYZ = np.zeros((X.shape[0], 3), dtype='intp')
        XYZ[:, 0], XYZ[:, 1], XYZ[:, 2] = X, Y, Z
        self.XYZ = XYZ
        self.mask = mask
        self.data = data[mask]
        if nchannels == 1:
            self.data = np.reshape(self.data, (self.data.shape[0], 1))

        # Initialize ppm as uniform distributions
        try:
            self.classes, self.matching_params = matching_params[usecase]
            self.usecase = usecase
            _mu, _s2 = moment_matching(self.data, *self.matching_params)
            if mu == None:
                mu = _mu
            if s2 == None:
                s2 = _s2
        except:
            self.usecase = 'unknown'
            self.matching_params = None

        nclasses = len(mu)
        self.nclasses = nclasses
        self.ppm = np.zeros(list(space_shape) + [nclasses])
        self.ppm[mask] = 1. / nclasses
        self.mu = np.array(mu, dtype=float).reshape(\
            (nclasses, nchannels))
        try:
            self.s2 = np.array(s2, dtype=float).reshape(\
                (nclasses, nchannels, nchannels))
        except:
            self.s2 = s2 * np.ones((nclasses, nchannels, nchannels))

        if prop == None:
            self.prop = np.ones(nclasses)
        else:
            self.prop = np.asarray(prop, dtype=float)

        if not prior == None:
            self.prior = np.asarray(prior)[self.mask].reshape(\
                [self.data.shape[0], nclasses])
        else:
            self.prior = None

        self.ngb_size = int(ngb_size)
        self.set_markov_prior(beta, U=U)

    def set_markov_prior(self, beta, U=None):
        if not U == None:  # make sure it's C-contiguous
            self.U = np.asarray(U).copy()
        else:  # Potts model
            U = np.ones((self.nclasses, self.nclasses))
            U[_diag_indices(self.nclasses)] = 0
            self.U = U
        if beta == None:
            beta = 0.5
        self.beta = float(beta)

    def vm_step(self, freeze=(), update_s2=True, update_prop=False):

        # don't update frozen class parameters 
        classes = range(self.nclasses)
        for i in freeze:
            classes.remove(i)
        
        for i in classes:
            P = self.ppm[..., i][self.mask].ravel()
            Z = nonzero(P.sum())
            tmp = self.data.T * P.T
            mu = tmp.sum(1) / Z
            mu_ = mu.reshape((len(mu), 1))
            self.mu[i] = mu
            if update_s2:
                s2 = np.dot(tmp, self.data) / Z - np.dot(mu_, mu_.T)
                self.s2[i] = s2
            if update_prop:
                self.prop[i] = Z
        
        if update_prop:
            self.prop /= np.sum(self.prop)


    def log_external_field(self):
        """
        Compute the logarithm of the external field, where the
        external field is defined as the likelihood times the
        first-order component of the prior.
        """
        lef = np.zeros([self.data.shape[0], self.nclasses])

        for i in range(self.nclasses):
            centered_data = self.data - self.mu[i]
            if self.nchannels == 1:
                inv_s2 = 1. / nonzero(self.s2[i])
                norm_factor = np.sqrt(inv_s2.squeeze())
            else:
                inv_s2 = np.linalg.inv(self.s2[i])
                norm_factor = 1. / np.sqrt(\
                    nonzero(np.linalg.det(self.s2[i])))
            maha_dist = np.sum(centered_data * np.dot(inv_s2,
                                                      centered_data.T).T, 1)
            lef[:, i] = -.5 * maha_dist + log(norm_factor) + log(self.prop[i])
            #lef[:, i] += log(norm_factor)
            #lef[:, i] += log(self.prop[i])

        if self.prior != None:
            lef += log(self.prior)

        return lef

    def normalized_external_field(self):
        f = self.log_external_field().T
        f -= np.max(f, 0)
        np.exp(f, f)
        f /= f.sum(0)
        return f.T

    def ve_step(self):
        nef = self.normalized_external_field()
        if self.beta == 0:
            self.ppm[self.mask] = np.reshape(\
                nef, self.ppm[self.mask].shape)
        else:
            self.ppm = _ve_step(self.ppm, nef, self.XYZ,
                                self.U, self.ngb_size, self.beta)

    def run(self, niters=1, freeze=(), update_s2=True, update_prop=False):
        for i in range(niters):
            self.ve_step()
            self.vm_step(freeze=freeze, update_s2=update_s2, update_prop=update_prop)

    def map(self):
        """
        Return the maximum a posterior label map
        """
        return map_from_ppm(self.ppm, self.mask)

    def free_energy(self, ppm=None):
        """
        Compute the free energy defined as:

        F(q, theta) = int q(x) log q(x)/p(x,y/theta) dx

        associated with input parameters mu,
        s2 and beta (up to an ignored constant).
        """
        if ppm == None:
            ppm = self.ppm
        q = ppm[self.mask]
        # Entropy term
        lef = self.log_external_field()
        f1 = np.sum(q * (log(q) - lef))
        # Interaction term
        if self.beta > 0.0:
            f2 = self.beta * _interaction_energy(ppm, self.XYZ,
                                                 self.U, self.ngb_size)
        else:
            f2 = 0.0
        return f1 + f2


def _diag_indices(n, ndim=2):
    # diag_indices function present in numpy 1.4 and later.  This for
    # compatibility with numpy < 1.4
    idx = np.arange(n)
    return (idx,) * ndim


def map_from_ppm(ppm, mask=None):
    x = np.zeros(ppm.shape[0:-1], dtype='uint8')
    if mask == None:
        mask = ppm == 0
    x[mask] = ppm[mask].argmax(-1) + 1
    return x


def binarize_ppm(q):
    """
    Assume input ppm is masked (ndim==2)
    """
    bin_q = np.zeros(q.shape)
    bin_q[(range(q.shape[0]), np.argmax(q, axis=1))] = 1.
    return bin_q
