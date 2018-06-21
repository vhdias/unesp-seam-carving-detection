#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as f:
    readme_file = f.read()

with open('LICENSE') as f:
    license_file = f.read()

with open('requirements.txt') as f:
    requirements_file = f.read()

setup(
    name='unesp-seam-carving-detection',
    version='0.1.0',
    description='',
    long_description=readme_file,
    author='',
    author_email='',
    url='https://github.com/wentel/unesp-seam-carving-detection',
    license=license_file,
    packages=find_packages(exclude=('tests', 'docs'))
)