from setuptools import setup, find_packages

setup(
    name="AMtools",
    version="0.1.0",
    packages=find_packages(where="src"),  # Automatically find packages in src/
    package_dir={"": "src"},  # Directs setuptools to look for packages under the src/ directory
)
