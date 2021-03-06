# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os

def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('niseg', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_subpackage('testing')
    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_extension('_segmentation',
                         sources=['_segmentation.pyx', 
                                  'mrf.c',
                                  'pve.c'])

    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
