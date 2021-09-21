from setuptools import setup

setup(
    name='ssl-for-slr',
    version='0.0.1-dev',
    packages=['sslforslr',
              'sslforslr.utils',
              'sslforslr.modules',
              'sslforslr.dataset',
              'sslforslr.models',
              'sslforslr.models.encoders',
              'sslforslr.models.cpc',
              'sslforslr.models.lim',
              'sslforslr.models.multitask',
              'sslforslr.models.wav2vec2',
              'sslforslr.models.vqwav2vec'],
)