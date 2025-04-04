# test_package.py

import AMtools  # This should match the name of your package (in src/AMtools)

def test_package_import():
    assert AMtools is not None  # Simple test to check that the package can be imported

if __name__ == "__main__":
    test_package_import()
    print("Package import test passed!")
