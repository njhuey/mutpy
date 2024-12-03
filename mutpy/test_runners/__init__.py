from .unittest_runner import UnittestTestRunner
from importlib.util import find_spec


def pytest_installed():
    import importlib.util

    pytest_loader = importlib.util.find_spec("pytest")
    return pytest_loader is not None


class TestRunnerNotInstalledException(Exception):
    pass


def __pytest_not_installed(*args, **kwargs):
    raise TestRunnerNotInstalledException(
        'Pytest is not installed. Please run "pip install pytest" to resolve this issue.'
    )


if pytest_installed():
    from .pytest_runner import PytestTestRunner
else:
    PytestTestRunner = __pytest_not_installed
