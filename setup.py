from setuptools import setup, find_packages

setup(
    name='pirl',
    version='0.0.0',
    url='https://github.com/anthony0727/physics-informed-metasurface', # TODO(jlab): migration
    author='JLAB',
    author_email='contact@jlab.com', # TODO(jlab)
    packages=find_packages(),
    install_requires=[
        'gym',
        'ray[rllib]',
    ],
    python_requires='>=3.8'
)