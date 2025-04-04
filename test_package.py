"""
Test script to check the successful import of the amtools package.

This script performs a simple test to ensure that the amtools package can
be imported correctly. It checks whether the package can be accessed without
any import errors and prints a confirmation message if the import test passes.

Usage:
    To run the test, simply execute this script. If the package imports
    correctly, the message "Package import test passed!" will be printed.
"""

import amtools  # This should match the name of your package (in src/amtools)

def test_package_import():
    """
    Tests whether the amtools package can be imported successfully.

    This function checks that the amtools package is not None, indicating
    that the package was imported correctly.

    Raises:
        AssertionError: If amtools is None or cannot be imported.
    """
    assert amtools is not None  # Simple test to check that the package can be imported

if __name__ == "__main__":
    test_package_import()
    print("Package import test passed!")
