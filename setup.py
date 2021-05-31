from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='slicer',
    version='1.0.0',
    packages=['slicer'],
    install_requires=requirements,
    url='',
    license='GPL',
    author='Miroslav Suruzhon',
    author_email='',
    description='Sequential LIgand Conformation ExploreR (SLICER) - A Sequential Monte Carlo Sampler for OpenMM'
)
