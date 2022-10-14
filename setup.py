from setuptools import setup, find_packages

setup(
    name='pirl',
    version='0.0.0',
    url='https://github.com/anthony0727/physics-informed-metasurface', # TODO(jlab): migration
    author='JLAB',
    author_email='contact@jlab.com', # TODO(jlab)
    packages=find_packages(),
    install_requires=[
        'gym<0.26.0',
        'ray[rllib]',
        'matlabengine==9.12',
    ],
    python_requires='>=3.8'
)