# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os

def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('segmentation', parent_package, top_path)
    config.add_subpackage('tests')


    lapack_info = get_info('lapack_opt', 0)
    if 'libraries' not in lapack_info:
        lapack_info = get_info('lapack', 0)
    # if Lapack not found on system        
    if not lapack_info:
        sources = [os.path.join(os.path.realpath('lib'),'lapack_lite','*.c')]
        library_dirs = []
        libraries = []
    # if Lapack found
    else:
        sources = []
        library_dirs = lapack_info['library_dirs']
        libraries = lapack_info['libraries']
        if 'include_dirs' in lapack_info:
            config.add_include_dirs(lapack_info['include_dirs'])

    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_extension('_segmentation',
                         sources=sources+['_segmentation.pyx', 
                                          'mrf.c', 'pve.c'],
                         library_dirs=library_dirs,
                         libraries=libraries,
                         extra_info=lapack_info)

    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
