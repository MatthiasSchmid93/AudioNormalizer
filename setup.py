from setuptools import setup, find_namespace_packages


setup(
    name='AudioNormalizer',
    version='1.3.1',
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        "setuptools",
        "mutagen==1.46.0",
        "numpy==1.21.5",
        "pydub==0.25.1",
        "PyQt5==5.12.3",
        "PySimpleGUI==4.60.5",
        "scipy==1.9.1",
        "jsonschema>=3.2.0",
        "matplotlib==3.5.2",
    ],
    package_data={'audio_normalizer': ['assets/*']},
    author='Matthias Christopher Schmid',
    author_email='schmid.mat93@gmail.com',
    description='A tool to normalize audio',
)