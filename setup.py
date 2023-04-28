from pathlib import Path

from setuptools import setup, find_packages

# Load the version from file
__version__ = Path("VERSION").read_text().strip()

setup(
  name = 'open-musiclm',
  packages = find_packages(exclude=[]),
  version = __version__,
  license='MIT',
  description = 'Open MusicLM - Implementation of MusicLM, a text to music model published by Google Research, with a few modifications',
  author = 'Allen Zhang',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/zhvng/open-musiclm',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'audio generation',
    'musiclm',
  ],
  install_requires=[
    'torch',
    'torchvision',
    'torchaudio',
    'einops',
    'vector-quantize-pytorch',
    'librosa',
    'torchlibrosa',
    'ftfy',
    'tqdm',
    'transformers',
    'encodec',
    'gdown',
    'accelerate',
    'beartype',
    'joblib',
    'h5py',
    'sklearn',
    'wget',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)