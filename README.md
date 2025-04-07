# Actuator-Post-Processing
Python package and scripts to be used with the actuator method output files.

| Current Status |
|:--------------:|
| ![Running](https://github.com/UoO-Tidal-and-Wind/Actuator-Method-Tools/actions/workflows/pylint.yml/badge.svg) |


## Installation
Source the bashrc and add the bin directory to the path using 
```bash
./install
```
Then install the Python package
```bash
pip install .
```

## Documentation
To make the documentation please run the following
```bash
pip install amtools[docs]
cd docs
sphinx-apidoc -f -o . ../src/amtools
make html
```
The documentation can be opened in the build directory.

## Contribution
Please follow the PEP8 formatting style (see [Contributing Guidelines](CONTRIBUTING.md)). Formatting checking with pylint is set up and should be followed unless necessary. 
