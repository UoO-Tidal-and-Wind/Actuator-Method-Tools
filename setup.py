"""
Setup script for amtools package.

This script uses setuptools to package and distribute the amtools Python package.
"""
from setuptools import setup, find_packages

setup(
    name="amtools",
    version="0.1.0",
    packages=find_packages(where="src"),  # Automatically find packages in src/
    package_dir={"": "src"},  # Directs setuptools to look for packages under the src/ directory
)
