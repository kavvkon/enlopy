""" Setup script for enlopy"""

import codecs
from setuptools import setup, find_packages
import os

# import distutils.command.bdist_conda

import enlopy

HERE = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()

version = enlopy.__version__

requirements = ['numpy>=1.10',
                'scipy>=0.15',
                'matplotlib>=1.5',
                'pandas>=0.18']

setup(
    name="enlopy",
    author="Konstantinos Kavvadias",
    author_email="kavvkon (at) gmail (dot) com",
    url='https://github.com/kavvkon/enlopy',
    description='Python library with methods to generate, process, analyze, and plot energy related timeseries.',
    long_description=read('README.rst'),
    license="BSD-3-Clause",
    version=version,
    install_requires=requirements,
    keywords=['energy','timeseries','statistics','profile','demand'],
    packages=find_packages(),
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "Topic :: Utilities",
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'Natural Language :: English',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',

        #    distclass=distutils.command.bdist_conda.CondaDistribution,
        #    conda_buildnum=1,
        #    conda_features=['mkl'],

    ],

)
