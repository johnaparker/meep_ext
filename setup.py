import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "meep_ext",
    version = "0.1.0",
    author = "John Parker",
    author_email = "japarker@uchicago.com",
    description = (""),
    license = "MIT",
    keywords = "",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=['meep_ext'],
    long_description=read('README.md'),
    install_requires=['numpy', 'scipy'],
    include_package_data = True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
    ],
)
