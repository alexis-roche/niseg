#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Script example of partial volume estimation
"""
from argparse import ArgumentParser

import numpy as np
import nibabel as nb

from niseg import BrainT1PVE


# Parse command line
description = 'Estimate brain tissue concentrations of CSF, GM and WM from a skull \
stripped T1 image in CSF, GM and WM. If no mask image is provided, the mask is defined \
by thresholding the input image above zero (strictly).'

parser = ArgumentParser(description=description)
parser.add_argument('img', metavar='img', nargs='+', help='input image')
parser.add_argument('--mask', dest='mask', help='mask image')
parser.add_argument('--niters', dest='niters',
    help='number of iterations (default=%d)' % 25)
parser.add_argument('--beta', dest='beta',
    help='Spatial smoothness beta parameter (default=%f)' % 0.5)
parser.add_argument('--ngb_size', dest='ngb_size',
    help='Grid neighborhood system (default=%d)' % 6)
args = parser.parse_args()


def get_argument(dest, default):
    val = args.__getattribute__(dest)
    if val == None:
        return default
    else:
        return val

# Input image
img = nb.load(args.img[0])

# Input mask image
mask_img = get_argument('mask', None)
if mask_img == None:
    mask_img = img
else:
    mask_img = nb.load(mask_img)
mask = mask_img.get_data() > 0

# Other optional arguments
niters = get_argument('niters', 25)
beta = get_argument('beta', None)
ngb_size = get_argument('ngb_size', 6)

# Perform tissue classification
PV = BrainT1PVE(img, mask=mask, beta=beta, ngb_size=ngb_size)
PV.run(niters=niters, print_parameters=True)

# Save tissue concentration maps
PV.save('temp')
