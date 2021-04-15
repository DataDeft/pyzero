import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['pyzero']
from version import __version__

setup(
  name = 'pyzero',
  packages = find_packages(),
  entry_points={
    'console_scripts': [
      'pyzero = pyzero.cli:main',
    ],
  },
  version = __version__,
  license='MIT',
  description = 'MuZero in Python',
  author = 'DataDeft',
  author_email = 'istvan@datadeft.eu',
  url = 'https://github.com/DataDeft/pyzero',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'muzero'
  ],
  install_requires=[
    'numpy'
  ],
  classifiers=[
    'Development Status :: 0 - Alfa',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)
