# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np


# Define pre-set parameters for supported segmentation scenarios
matching_params = {
    'brainT1': (
        ('csf', 'gm', 'wm'), 
        (np.array([696.801288, 1637.6656123, 2198.68819106]),
         14250.595084960518,
         1642.2369045635603,
         254155.92296080582)),
    'brainT1_4k': (
        ('csf', 'gm', 'gm/wm', 'wm'), 
        (np.array([696.801288, 1637.6656123, 1918.17690168, 2198.68819106]),
         14250.595084960518,
         1642.2369045635603,
         254155.92296080582)),
    'brainT1_5k': (
        ('csf', 'csf/gm', 'gm', 'gm/wm', 'wm'), 
        (np.array([696.801288, 1167.23345015, 1637.6656123, 1918.17690168, 2198.68819106]),
         14250.595084960518,
         1642.2369045635603,
         254155.92296080582))
}


def moment_matching(dat, mu, s2, mean, var):
    """
    Moment matching strategy for parameter initialization to feed a
    segmentation algorithm.

    Parameters
    ----------
    data: array
      Image data.

    mu : array
      Template class-specific intensity means

    s2 : array
      Template class-specific intensity variances

    mean : float
      Template global intensity mean

    var : float
      Template global intensity variance

    Returns
    -------
    dat_mu: array
      Guess of class-specific intensity means

    dat_s2: array
      Guess of class-specific intensity variances
    """
    dat_mean = float(np.mean(dat))
    dat_var = float(np.var(dat))
    a = np.sqrt(dat_var / var)
    b = dat_mean - a * mean
    dat_mu = a * mu + b
    dat_s2 = (a ** 2) * s2
    return dat_mu, dat_s2


