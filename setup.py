from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='openmmslicer',
    version='3.1.0',
    packages=find_packages(),
    install_requires=requirements,
    url='',
    license='GPL',
    author='Miroslav Suruzhon',
    author_email='',
    description='Sequential LIgand Conformation ExploreR (SLICER)---A Sequential Monte Carlo Sampler for OpenMM'
)
