from pathlib import Path

from setuptools import setup, find_packages

# Load the version from file
__version__ = Path("VERSION").read_text().strip()

setup(
  name = 'open-musiclm',
  packages = find_packages(exclude=[]),
  include_package_data=True,
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
    'einops>=0.6.1',
    'vector-quantize-pytorch>=1.2.2',
    'librosa',
    'torchlibrosa',
    'ftfy',
    'tqdm',
    'transformers',
    'encodec',
    'gdown',
    'accelerate>=0.17.0',
    'beartype',
    'joblib',
    'h5py',
    'scikit-learn',
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