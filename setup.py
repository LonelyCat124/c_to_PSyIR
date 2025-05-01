#!/usr/bin/env python

# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2017-2025, Science and Technology Facilities Council
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

"""Setup script. Used by easy_install and pip."""

import os
from setuptools import setup, find_packages


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_PATH, "src")
PACKAGES = find_packages(where=SRC_PATH)

NAME = 'c_to_PSyIR'
AUTHOR = ('LonelyCat124')
LICENSE = 'OSI Approved :: BSD 3-Clause License'

DESCRIPTION = ('An experimental project to investigate the ablity to convert'
               'a subset of C into PSyIR to enable extraction or other similar'
               'functionality.')

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
    'Programming Language :: Fortran',
    'Programming Language :: C',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
    'Topic :: Utilities',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS']

VERSION = "0.0.1"

if __name__ == '__main__':

    def get_files(directory, install_path, valid_suffixes):
        '''Utility routine that creates a list of 2-tuples, each consisting of
        the target installation directory and a list of files
        (specified relative to the project root directory).

        :param str directory: the directory containing the required files.
        :param str install_path: the location where the files will be placed.
        :param valid_suffixes: the suffixes of the required files.
        :type valid_suffixes: [str]

        :returns: a list of 2-tuples, each consisting of the target \
            installation directory and a list of files (specified relative \
            to the project root directory).
        :rtype: [(str, [str])]

        '''
        examples = []
        for dirpath, _, filenames in os.walk(directory):
            if ("__" not in dirpath) and filenames:
                rel_path = os.path.relpath(dirpath, directory)
                files = []
                for filename in filenames:
                    if any(filename.endswith(suffix) for suffix in
                           valid_suffixes):
                        files.append(
                            os.path.join(os.path.basename(install_path),
                                         rel_path, filename))
                if files:
                    examples.append((os.path.join(install_path, rel_path),
                                     files))
        return examples

    # We have all of the example, tutorial and wrapper libraries files
    # listed in MANIFEST.in but unless we specify them in the data_files
    # argument of setup() they don't seem to get installed.
    # Since the data_files argument doesn't accept wildcards we have to
    # explicitly list every file we want.
    # INSTALL_PATH controls where the files will be installed.
    # VALID_SUFFIXES controls the type of files to include.
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        license=LICENSE,
        classifiers=CLASSIFIERS,
        packages=PACKAGES,
        package_dir={"": "src"},
        install_requires=["PSyclone", "pycparser"],
        include_package_data=True,
        )
