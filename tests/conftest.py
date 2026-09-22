from __future__ import annotations

import os

_CONFIRMATION_TEST_ENV = "QLDPC_FNO_CONFIRMATION_TESTING"
_MISSING = object()
_prior_confirmation_testing: str | object = _MISSING


def pytest_configure(config) -> None:
    """Forbid production confirmation coordinates throughout test collection."""
    del config
    global _prior_confirmation_testing
    _prior_confirmation_testing = os.environ.get(_CONFIRMATION_TEST_ENV, _MISSING)
    os.environ[_CONFIRMATION_TEST_ENV] = "1"


def pytest_unconfigure(config) -> None:
    """Restore the caller's environment after the pytest session."""
    del config
    if _prior_confirmation_testing is _MISSING:
        os.environ.pop(_CONFIRMATION_TEST_ENV, None)
    else:
        os.environ[_CONFIRMATION_TEST_ENV] = str(_prior_confirmation_testing)
