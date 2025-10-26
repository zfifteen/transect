#!/usr/bin/env python3
"""
Setup script for TRANSEC - Time-Synchronized Encryption
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

setup(
    name='transec',
    version='0.1.0',
    description='Time-Synchronized Encryption for Zero-Handshake Messaging',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Z-Sandbox Project',
    author_email='your.email@example.com',
    url='https://github.com/zfifteen/transec',
    packages=find_packages(exclude=['tests', 'examples']),
    install_requires=[
        'cryptography>=42.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
        'optional': [
            'mpmath>=1.3.0',  # For prime optimization
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Security :: Cryptography',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='encryption cryptography zero-rtt transec comsec frequency-hopping',
    project_urls={
        'Documentation': 'https://github.com/zfifteen/transec/blob/main/README.md',
        'Source': 'https://github.com/zfifteen/transec',
        'Tracker': 'https://github.com/zfifteen/transec/issues',
    },
)
