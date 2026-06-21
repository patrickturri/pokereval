import pokereval

def test_package_imports_and_has_version():
    assert isinstance(pokereval.__version__, str)
    assert pokereval.__version__
