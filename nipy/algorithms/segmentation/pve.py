from os.path import join, splitext
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

from ...core.image.image_spaces import (make_xyz_image,
                                        as_xyz_image,
                                        xyz_affine)
from ... import save_image
from .moment_matching import moment_matching, matching_params
from ._segmentation import update_cmap


def hyperparameters(usecase, ngb_size):
    """
    Return suitable hyperparameters as a dictionary for a particular
    combination of tissues, modality and neighborhood size. 
    """
    p = {}
    if usecase == 'brainT1':
        if ngb_size == 6:
            p['alpha'] = np.array([10.398556972836019, 29485.917556006651, 7.030430723881528])
            p['beta'] = 1.1851546491201328
            p['gamma'] =  0.0048004191376676006
    return p


class PartialVolumeEstimation(object):

    def __init__(self, img, mask=None, mu=None, s2=1e-5,
                 alpha=None, beta=None, gamma=None,
                 ngb_size=6, usecase='brainT1'):
        try:
            self.tissues, self.matching_params = matching_params[usecase]
            self.usecase = usecase
        except:
            self.tissues = np.arange(len(mu)).astype(str)
            self.matching_params = None
            self.usecase = 'unknown'
        ntissues = len(self.tissues)
        self.ngb_size = int(ngb_size)
        # set hyperparameters
        self._hyperparams = hyperparameters(self.usecase, self.ngb_size)
        self.set_alpha(alpha)
        self.set_beta(beta)
        self.set_gamma(gamma)
        # data and masking
        self._init_data(img, mask)
        # initialize intensity parameters
        if self.matching_params != None:
            _mu, _s2 = self.guess_theta()
            if mu == None:
                self.mu = _mu
            else:
                self.mu = np.asarray(mu, dtype=float).squeeze()
            if s2 == None:
                self.s2 = _s2
            else:
                self.s2 = float(s2)
        self.mmu = np.mean(self.mu)
        # initialize with uniform concentrations
        self.cmap = np.zeros(list(self.shape) + [ntissues])
        self.cmap[self.mask] = 1. / ntissues
        # sequence of intensity parameters
        self._mu = [self.mu]
        self._s2 = [self.s2]

    def set_alpha(self, alpha):
        ntissues = len(self.tissues)
        if alpha == None:
            alpha = self.hyperparam('alpha')
        try:
            a = float(alpha)
            self.alpha = np.zeros(ntissues)
            self.alpha.fill(a)
        except:
            self.alpha = np.asarray(alpha)
        if self.alpha.size == (ntissues ** 2 - ntissues) / 2:
            self.alpha_mat = np.zeros((ntissues, ntissues))
            self.alpha_mat[np.triu_indices(ntissues, 1)] = self.alpha
            self.alpha_mat[np.tril_indices(ntissues, -1)] = self.alpha
        else:
            self.alpha_mat = self.alpha
        if not self.alpha_mat.shape == (ntissues, ntissues):
            raise ValueError('Wrong alpha matrix shape, expected (%d, %d)'\
                                 % (ntissues, ntissues))

    def set_beta(self, beta):
        if beta == None:
            beta = self.hyperparam('beta')
        self.beta = float(beta)

    def set_gamma(self, gamma):
        if gamma == None:
            gamma = self.hyperparam('gamma')
        self.gamma = float(gamma)
    
    def _init_data(self, img, mask):
        # get image data
        img = as_xyz_image(img)
        data = img.get_data()
        self.affine = xyz_affine(img)
        self.shape = data.shape
        # masking -- by default, mask out zero intensities
        if mask == None:
            mask = binary_fill_holes(data > 0)
        self.data = data[mask]
        X, Y, Z = np.where(mask)
        XYZ = np.zeros((X.shape[0], len(self.tissues)), dtype='intp')
        XYZ[:, 0], XYZ[:, 1], XYZ[:, 2] = X, Y, Z
        self.mask = mask
        self.XYZ = XYZ

    def hyperparam(self, key):
        p = self._hyperparams[key]
        if p == None:
            raise NotImplementedError('Unknown hyperparameter')
        return p

    def guess_theta(self):
        mu, s2 = moment_matching(self.data, *self.matching_params)
        return mu, s2

    def masked_cmap(self):
        return self.cmap[self.mask]

    def update_theta(self, fcmean=False):
        if fcmean:
            self._update_theta_fcmean()
            return
        npts = len(self.data)
        mcmap = self.masked_cmap()
        b = np.sum(np.reshape(self.data, (npts, 1)) * mcmap, 0)
        A = np.dot(mcmap.T, mcmap)
        if self.gamma > 0:
            A += self.gamma * npts * np.eye(len(self.tissues))
            b += self.gamma * npts * self.mmu
        self.mu = np.asarray(np.dot(np.linalg.pinv(A), b))
        self.s2 = float(np.mean((self.data - np.dot(mcmap, self.mu)) ** 2))
        if self.gamma > 0:
            self.s2 += float(self.gamma * np.sum((self.mu - self.mmu) ** 2))
        self.mmu = np.mean(self.mu)
        self._mu.append(self.mu)
        self._s2.append(self.s2)

    def _update_theta_fcmean(self):
        npts = len(self.data)
        w = self.cmap[self.mask] ** 2
        self.mu = np.asarray(np.sum(w * np.reshape(self.data, (npts, 1)), 0)\
                                 / np.sum(w, 0))
        tmp = self.data - np.reshape(self.mu, (len(self.tissues), 1))
        self.s2 = float(np.sum((tmp.T * self.masked_cmap()) ** 2) / npts)

    def update(self, fcmean=False):
        """
        Update tissue concentration map
        """
        if fcmean:
            self._update_fcmean()
            return
        update_cmap(self.cmap, self.data, self.XYZ, self.mu, self.s2,
                    self.alpha_mat, self.beta, self.ngb_size)

    def _update_fcmean(self):
        npts = len(self.data)
        data = np.reshape(self.data, (npts, 1))
        #d2 = (data - self.mu) ** -2
        d2 = np.maximum(np.abs(data - self.mu), 1e-50) ** -2
        self.cmap[self.mask] = (d2.T / d2.sum(1)).T

    def save(self, fname, path='.'):
        self._save(fname, path, 'cmap')

    def _save(self, fname, path, tag):
        """
        Save concentration map to disk 
        """
        for i in range(len(self.tissues)):
            img = make_xyz_image(self.cmap[..., i], self.affine, 'scanner')
            save_image(img, join(path, tag + '_' + self.tissues[i] +\
                                     '_' + splitext(fname)[0] + '.nii.gz'))

    def _print_theta(self):
        print('mu = %s' % self.mu)
        print('s2 = %s' % self.s2)

    def run(self, niters=1, fcmean_niters=0, update_theta=True, print_theta=False):
        niters = niters + fcmean_niters
        fcmean_count = 0
        if print_theta:
            self._print_theta()
        for i in range(niters):
            fcmean = fcmean_count < fcmean_niters
            print('Iter %d/%d' % (i + 1, niters))
            self.update(fcmean=fcmean)
            if update_theta:
                self.update_theta(fcmean=fcmean)
            fcmean_count += 1
            if print_theta:
                self._print_theta()

    def check_convexity(self):
        A = self.alpha_mat +\
            2 * self.beta * self.ngb_size * np.eye(len(self.tissues))
        w, _ = np.linalg.eigh(A)
        if np.min(w) < 1e-5:
            return False
        return True



class FuzzyCMean(PartialVolumeEstimation):

    def run(self, niters=1, update_theta=True, print_theta=False):
        if print_theta:
            self._print_theta()
        for i in range(niters):
            print('Iter %d/%d' % (i + 1, niters))
            self.update(fcmean=True)
            if update_theta:
                self.update_theta(fcmean=True)
            if print_theta:
                self._print_theta()

    def check_convexity(self):
        return True

    def save(self, fname, path='.'):
        self._save(fname, path, 'fmap')

    

