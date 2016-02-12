from os.path import join, splitext
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

import nibabel as nb
from .moment_matching import moment_matching, matching_params
from ._segmentation import update_cmap


def update_parameters(data, mcmap, stab, refmu, stab_s2=False):
    npts = len(data)
    ntissues = mcmap.shape[1]
    b = np.mean(np.reshape(data, (npts, 1)) * mcmap, 0)
    A = np.dot(mcmap.T, mcmap) / npts
    if stab > 0:
        A += stab * np.eye(ntissues)
        b += stab * refmu
    mu = np.asarray(np.dot(np.linalg.pinv(A), b))
    s2 = float(np.mean((data - np.dot(mcmap, mu)) ** 2))
    if stab_s2:
        s2 += float(stab * np.sum((mu - refmu) ** 2))
    return mu, s2


def update_parameters_fcmean(data, mcmap):
    npts = len(data)
    ntissues = mcmap.shape[1]
    w = mcmap ** 2
    mu = np.asarray(np.sum(w * np.reshape(data, (npts, 1)), 0)\
                        / np.sum(w, 0))
    tmp = data - np.reshape(mu, (ntissues, 1))
    s2 = float(np.sum((tmp.T * mcmap) ** 2) / npts)
    return mu, s2


def name_tissues(mu, tissues=None):
    ntissues = np.asarray(mu).shape[0]
    if tissues == None:
        if ntissues != 3:
            tissues = ['tissue' + str(k) for k in range(ntissues)]
        else:
            tissues = ('csf', 'gm', 'wm')
    else:
        if len(tissues) != ntissues:
            raise ValueError('mu and tissues arguments should have same length')
        tissues = [str(t) for t in tissues]
    return tissues


class PVE(object):

    def __init__(self, img, mu, s2=1e-5, mask=None, 
                 alpha=0, beta=0, gamma=0, update_refmu=False,
                 ngb_size=6, tissues=None):

        """
        alpha : a sequence with size K(K+1)/2 of real numbers or images.
        """
        self.tissues = name_tissues(mu, tissues)
        self._init_data(img, mask)
        self._finit(mu, s2, alpha, beta, gamma, update_refmu, ngb_size)

    def _init_data(self, img, mask):
        # get image data
        data = img.get_data()
        self.affine = img.get_affine()
        self.shape = data.shape
        # masking -- by default, mask out zero intensities
        if mask == None:
            mask = binary_fill_holes(data > 0)
        self.data = data[mask]
        X, Y, Z = np.where(mask)
        XYZ = np.zeros((X.shape[0], 3), dtype='intp')
        XYZ[:, 0], XYZ[:, 1], XYZ[:, 2] = X, Y, Z
        self.mask = mask
        self.XYZ = XYZ

    def _finit(self, mu, s2, alpha, beta, gamma, update_refmu, ngb_size):
        ntissues = len(self.tissues)
        self.ngb_size = int(ngb_size)
        # set hyperparameters
        self.set_alpha(alpha)
        self.set_beta(beta)
        self.set_gamma(gamma)
        self.update_refmu = bool(update_refmu)
        # initialize intensity parameters
        self._init_parameters(mu, s2)
        # initialize with uniform concentrations
        self.cmap = np.zeros(list(self.shape) + [ntissues])
        self.cmap[self.mask] = 1. / ntissues
        # sequence of intensity parameters
        self._mu = [self.mu]
        self._s2 = [self.s2]

    def _init_parameters(self, mu, s2):
        self.mu = np.asarray(mu, dtype=float)
        self.s2 = float(s2)
        self.refmu = self.mu.copy()
        self._update_refmu()

    def set_alpha(self, alpha):
        """
        Either alpha is a float, a sequence of floats, or a sequence of float/images
        """
        ntissues = len(self.tissues)
        alpha_size = (ntissues * (ntissues + 1)) / 2
        try:
            a = float(alpha)
            self.alpha = np.zeros(alpha_size)
            self.alpha.fill(a)
            return
        except:
            if len(alpha) != alpha_size:
                raise ValueError('Wrong length for alpha parameter, should be %d' % alpha_size)
        try:
            self.alpha = np.array(alpha, dtype=float)
            return
        except:
            self.alpha = np.zeros((self.data.shape[0], alpha_size))
        for i in range(alpha_size):
            try:
                self.alpha[:, i] = alpha[i]
            except:
                self.alpha[:, i] = alpha[i].get_data()[self.mask]

    def set_beta(self, beta):
        self.beta = float(beta)

    def set_gamma(self, gamma):
        self.gamma = float(gamma)
        
    def masked_cmap(self):
        return self.cmap[self.mask]

    def simulate_data(self):
        data = np.zeros(self.shape)
        data[self.mask] = np.dot(self.masked_cmap(), self.mu)
        return data

    def _update_refmu(self):
        if self.update_refmu:
            self.refmu.fill(np.mean(self.mu))

    def update_parameters(self, fcmean=False, freeze_mu=False, freeze_s2=False):
        if fcmean:
            mu, s2 = update_parameters_fcmean(self.data, self.masked_cmap())
        else:
            if self.update_refmu:
                stab = self.gamma
                stab_s2 = True
            else:
                stab = self.gamma * self.s2 / len(self.data)
                stab_s2 = False
            mu, s2 = update_parameters(self.data, self.masked_cmap(),
                                       stab, self.refmu, stab_s2)
        if not freeze_mu:
            self.mu = mu
        if not freeze_s2:
            self.s2 = s2
        self._update_refmu()
        self._mu.append(self.mu)
        self._s2.append(self.s2)

    def update(self, fcmean=False):
        """
        Update tissue concentration map
        """
        if fcmean:
            self._update_fcmean()
            return
        update_cmap(self.cmap, self.data, self.XYZ, self.mu, self.s2,
                    self.alpha, self.beta, self.ngb_size)

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
            img = nb.Nifti1Image(self.cmap[..., i], self.affine)
            nb.save(img, join(path, tag + '_' + self.tissues[i] +\
                              '_' + splitext(fname)[0] + '.nii.gz'))

    def _print_parameters(self):
        print('mu = %s' % self.mu)
        print('s2 = %s' % self.s2)

    def run(self, niters=1, fcmean_niters=0, freeze_mu=False, freeze_s2=False, print_parameters=False):
        niters = niters + fcmean_niters
        fcmean_count = 0
        if print_parameters:
            self._print_parameters()
        for i in range(niters):
            fcmean = fcmean_count < fcmean_niters
            print('Iter %d/%d' % (i + 1, niters))
            self.update(fcmean=fcmean)
            self.update_parameters(fcmean=fcmean, freeze_mu=freeze_mu, freeze_s2=freeze_s2)
            fcmean_count += 1
            if print_parameters:
                self._print_parameters()



class FuzzyCMean(PVE):

    def run(self, niters=1, update_parameters=True, print_parameters=False):
        if print_parameters:
            self._print_parameters()
        for i in range(niters):
            print('Iter %d/%d' % (i + 1, niters))
            self.update(fcmean=True)
            if update_parameters:
                self.update_parameters(fcmean=True)
            if print_parameters:
                self._print_parameters()

    def save(self, fname, path='.'):
        self._save(fname, path, 'fmap')


class BrainT1PVE(PVE):

    def __init__(self, img, mu=None, s2=1e-5, mask=None, 
                 alpha=None, beta=None, gamma=None, update_refmu=True,
                 ngb_size=6):

        self.tissues, p = matching_params['brainT1']
        self._init_data(img, mask)
        # guess initial tissue means and possibly noise variance by
        # moment matching
        if mu == None:
            mu, _s2 = moment_matching(self.data, *p)
            if s2 == None:
                s2 = _s2
        # guess adequate hyperparameters
        if ngb_size == 6:
            if alpha == None:
                alpha = np.array([0, 10.398556972836019, 29485.917556006651, 0, 7.030430723881528, 0])
            if beta == None:
                beta = 1.1851546491201328
            if gamma == None:
                gamma =  0.0048004191376676006
        self._finit(mu, s2, alpha, beta, gamma, update_refmu, ngb_size)
        


class MultichannelPVE(PVE):

    def __init__(self, imgs, mu, s2=1e-5, mask=None, 
                 alpha=0.0, beta=0.0, gamma=0.0, update_refmu=False,
                 ngb_size=6, tissues=None):
        """
        imgs: sequence of images
        """
        self.tissues = name_tissues(mu, tissues)
        self._init_data(imgs, mask)
        self._finit(mu, s2, alpha, beta, gamma, update_refmu, ngb_size)

    def _init_data(self, imgs, mask):
        # get image data
        img0 = imgs[0]
        data = img0.get_data()
        self.affine = img0.get_affine()
        self.shape = data.shape
        # masking -- by default, mask out zero intensities in the first image
        if mask == None:
            mask = binary_fill_holes(data > 0)
        X, Y, Z = np.where(mask)
        XYZ = np.zeros((X.shape[0], 3), dtype='intp')
        XYZ[:, 0], XYZ[:, 1], XYZ[:, 2] = X, Y, Z
        self.mask = mask
        self.XYZ = XYZ
        # data
        self.nchannels = len(imgs)
        self.data = np.zeros((X.shape[0], self.nchannels))
        for c in range(self.nchannels):
            self.data[:, c] = imgs[c].get_data()[mask]

    def set_gamma(self, gamma):
        self.gamma = np.array(gamma) * np.ones(self.nchannels)

    def _init_parameters(self, mu, s2):
        self.mu = np.asarray(mu, dtype=float)
        try:
            _s2 = float(s2)
            self.s2 = np.zeros(self.nchannels)
            self.s2.fill(_s2)
        except:
            self.s2 = np.asarray(s2)
        self.refmu = self.mu.copy()
        self._update_refmu()

    def _update_refmu(self):
        if self.update_refmu:
            self.refmu[:, :] = np.mean(self.mu, 0)

    def update_parameters(self, fcmean=False, freeze_mu=False, freeze_s2=False):
        self.mu = self.mu.copy()
        self.s2 = self.s2.copy()
        for k in range(self.nchannels):
            if fcmean:
                mu_k, s2_k = update_parameters_fcmean(self.data[:, k], self.masked_cmap())
            else:
                if self.update_refmu:
                    stab = self.gamma[k]
                    stab_s2 = True
                else:
                    stab = self.gamma[k] * self.s2[k] / self.data.shape[0]
                    stab_s2 = False
                mu_k, s2_k = update_parameters(self.data[:, k], self.masked_cmap(),
                                               stab, self.refmu[:, k], stab_s2)
            if not freeze_mu:
                self.mu[:, k] = mu_k
            if not freeze_s2:
                self.s2[k] = s2_k

        self._update_refmu()
        self._mu.append(self.mu)
        self._s2.append(self.s2)

    def _update_fcmean(self):
        npts = self.data.shape[0]
        w = self.s2[0] / self.s2
        d2 = 0
        for c in range(self.nchannels):
            tmp = np.maximum(np.abs(self.data[:, c].reshape((npts, 1)) - self.mu[:, c]), 1e-50) ** 2
            d2 += w[c] * tmp
        d2 = 1 / d2
        self.cmap[self.mask] = (d2.T / d2.sum(1)).T

    def simulate_data(self, k):
        data = np.zeros(self.shape)
        data[self.mask] = np.dot(self.masked_cmap(), self.mu[:, k])
        return data



class MultichannelFuzzyCMean(MultichannelPVE):

    def run(self, niters=1, update_parameters=True, print_parameters=False):
        if print_parameters:
            self._print_parameters()
        for i in range(niters):
            print('Iter %d/%d' % (i + 1, niters))
            self.update(fcmean=True)
            if update_parameters:
                self.update_parameters(fcmean=True)
            if print_parameters:
                self._print_parameters()

    def save(self, fname, path='.'):
        self._save(fname, path, 'fmap')


