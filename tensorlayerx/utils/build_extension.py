# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/5/22
import copy
import glob
import os
import os.path as osp
import re
import subprocess
import sys
from typing import Optional, List
import site

import setuptools
from pybind11.setup_helpers import Pybind11Extension
from setuptools.command.build_ext import build_ext

IS_WINDOWS = sys.platform == 'win32'


# _HERE = os.path.abspath(__file__)
# _TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
_TORCH_PATH = os.path.join(site.getsitepackages()[0], 'torch')
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')


SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()

COMMON_NVCC_FLAGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/EHsc']


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    return os.path.splitext(path)[1] in valid_ext


def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None

    return cuda_home


CUDA_HOME = _find_cuda_home()
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
# /usr/local/cuda-11.6
# C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA\\v10.2


# CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')


# >>> _join_cuda_home('bin', 'nvcc')
# >>> CUDA_HOME/bin/nvcc
def _join_cuda_home(*paths) -> str:
    r'''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def include_paths(cuda: bool = False, use_torch: bool = False) -> List[str]:
    '''
    Get the include paths required to build a C++ or CUDA extension.

    Parameters
    ----------
    cuda:
        If `True`, includes CUDA-specific include paths.

    Returns
    -------
    list[str]
        A list of include path strings.

    '''
    paths = []
    if cuda:
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    if use_torch:
        lib_include = os.path.join(_TORCH_PATH, 'include')
        paths += [
            lib_include,
            # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
            os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'),
            # Some internal (old) Torch headers don't properly prefix their includes,
            # so we need to pass -Itorch/lib/include/TH as well.
            os.path.join(lib_include, 'TH'),
            os.path.join(lib_include, 'THC')
        ]
    return paths


def library_paths(cuda: bool = False, use_torch: bool = False) -> List[str]:
    r'''
    Get the library paths required to build a C++ or CUDA extension.

    Parameters
    ----------
    cuda:
        If `True`, includes CUDA-specific library paths.

    Returns
    -------
    list[str]
        A list of library path strings.

    '''

    paths = []

    if cuda:
        if IS_WINDOWS:
            lib_dir = 'lib/x64'
        else:
            lib_dir = 'lib64'
            if (not os.path.exists(_join_cuda_home(lib_dir)) and
                    os.path.exists(_join_cuda_home('lib'))):
                # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
                # Note that it's also possible both don't exist (see
                # _find_cuda_home) - in that case we stay with 'lib64'.
                lib_dir = 'lib'

            paths.append(_join_cuda_home(lib_dir))
            if CUDNN_HOME is not None:
                paths.append(os.path.join(CUDNN_HOME, lib_dir))
    if use_torch:
        paths += [TORCH_LIB_PATH]
    return paths


def get_build_extension():
    try:
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension
    except ModuleNotFoundError:
        return build_ext
    

class PyBuildExtension(get_build_extension()):
    pass


def PyCUDAExtension(name, sources, use_torch: bool = False, *args, **kwargs):
    compile_extra_args = kwargs.get("compile_extra_args", [])
    if IS_WINDOWS:
        if not any(arg.startswith('/std:') for arg in compile_extra_args):
            compile_extra_args.append('/std:c++17')
    else:
        if not any(arg.startswith('-std:') for arg in compile_extra_args):
            compile_extra_args.append('-std=c++17')
    kwargs['extra_compile_args'] = compile_extra_args
    if use_torch:
        library_dirs = kwargs.get('library_dirs', [])
        library_dirs += library_paths(cuda=True, use_torch=use_torch)

        libraries = kwargs.get('libraries', [])
        libraries.append('c10')
        libraries.append('torch')
        libraries.append('torch_cpu')
        libraries.append('torch_python')
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')

        include_dirs = kwargs.get('include_dirs', [])
        include_dirs += include_paths(cuda=True, use_torch=use_torch)

        dlink_libraries = kwargs.get('dlink_libraries', [])
        dlink = kwargs.get('dlink', False) or dlink_libraries
        if dlink:
            extra_compile_args = kwargs.get('extra_compile_args', {})

            extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
            extra_compile_args_dlink += ['-dlink']
            extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
            extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

            # if (torch.version.cuda is not None) and TorchVersion(torch.version.cuda) >= '11.2':
            extra_compile_args_dlink += ['-dlto']   # Device Link Time Optimization started from cuda 11.2

            extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

            kwargs['extra_compile_args'] = extra_compile_args
    else:
        library_dirs = kwargs.get('library_dirs', [])
        library_dirs += library_paths(cuda=True)

        libraries = kwargs.get('libraries', [])
        libraries.append('cudart')

        include_dirs = kwargs.get('include_dirs', [])
        include_dirs += include_paths(cuda=True)

    include_pybind11 = kwargs.pop("include_pybind11", True)
    if include_pybind11:
        # If using setup_requires, this fails the first time - that's okay
        try:
            import pybind11
            pyinc = pybind11.get_include()
            if pyinc not in include_dirs:
                include_dirs.append(pyinc)
        except ModuleNotFoundError:
            pass

    kwargs['library_dirs'] = library_dirs
    kwargs['libraries'] = libraries
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    define_macros = kwargs.get("define_macros", [])
    define_macros.append(('WITH_CUDA', None))
    kwargs["define_macros"] = define_macros

    return setuptools.Extension(name, sources, *args, **kwargs)


def PyCppExtension(name, sources, use_torch: bool = False, *args, **kwargs):
    compile_extra_args = kwargs.get("compile_extra_args", [])
    if IS_WINDOWS:
        if not any(arg.startswith('/std:') for arg in compile_extra_args):
            compile_extra_args.append('/std:c++17')
    else:
        if not any(arg.startswith('-std:') for arg in compile_extra_args):
            compile_extra_args.append('-std=c++17')
    kwargs["compile_extra_args"] = compile_extra_args
    if use_torch:
        include_dirs = kwargs.get('include_dirs', [])
        include_dirs += include_paths(use_torch=use_torch)
        kwargs['include_dirs'] = include_dirs

        library_dirs = kwargs.get('library_dirs', [])
        library_dirs += library_paths(use_torch=use_torch)
        kwargs['library_dirs'] = library_dirs

        libraries = kwargs.get('libraries', [])
        libraries.append('c10')
        libraries.append('torch')
        libraries.append('torch_cpu')
        libraries.append('torch_python')
        kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])
    include_pybind11 = kwargs.pop("include_pybind11", True)
    if include_pybind11:
        # If using setup_requires, this fails the first time - that's okay
        try:
            import pybind11
            pyinc = pybind11.get_include()
            if pyinc not in include_dirs:
                include_dirs.append(pyinc)
        except ModuleNotFoundError:
            pass
    kwargs['include_dirs'] = include_dirs

    return setuptools.Extension(name, sources, *args, **kwargs)
