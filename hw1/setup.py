# setup.py
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='rob831',
    version='0.1.0',
    packages=['rob831'],
    install_requires=required
)