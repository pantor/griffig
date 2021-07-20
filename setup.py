from distutils.version import LooseVersion
import os
import re
import subprocess
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError as err:
            raise RuntimeError(
                'CMake must be installed to build the following extensions: ' +
                ', '.join(e.name for e in self.extensions)
            ) from err

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < LooseVersion('3.11.0'):
            raise RuntimeError('CMake >= 3.11.0 is required')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        build_type = os.environ.get('BUILD_TYPE', 'Release')
        build_args = [
            '--config', build_type,
            '--', '-j2',
        ]

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + extdir,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=' + extdir,
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE=' + extdir,
            f'-DPYTHON3_VERSION={sys.version_info.major}.{sys.version_info.minor}',
            f'-DPYBIND11_PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}',
            '-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE',
            '-DCMAKE_INSTALL_RPATH=$ORIGIN',
            '-DCMAKE_BUILD_TYPE=' + build_type,
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='griffig',
    version='0.0.4',
    description='Robotic Manipulation Learned from Imitation and Self-Supervision',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lars Berscheid',
    author_email='lars.berscheid@kit.edu',
    url='https://github.com/pantor/griffig',
    packages=find_packages(),
    license='LGPL',
    ext_modules=[CMakeExtension('_griffig'), CMakeExtension('pyaffx')],
    cmdclass=dict(build_ext=CMakeBuild),
    keywords=['robot', 'robotics', 'grasping', 'robot-learning'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: C++',
    ],
    setup_requires=[
        'setuptools>=18.0',
        'numpy',
        'some-pkg @ git+https://github.com/pybind/pybind.git@3.3.9#egg=some-pkg',
        'some-pkg @ git+https://gitlab.com/libeigen/eigen.git@v2.6.2#egg=some-pkg',
        'some-pkg @ git+https://github.com/opencv/opencv.git@4.5.2#egg=some-pkg',
    ],
    install_requires=[
        'loguru',
        'tensorflow>=2.4',
        'opencv-python',
        'numpy',
        'scipy>=1.5',
        'Pillow',
    ],
    python_requires='>=3.6',
)
