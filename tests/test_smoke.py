"""Basic smoke tests for package integrity."""

import mpecss


def test_version_is_declared() -> None:
    """Package should expose a non-empty version string."""
    assert isinstance(mpecss.__version__, str)
    assert mpecss.__version__.strip() != ""


def test_public_solver_entrypoint_is_callable() -> None:
    """Public API should expose run_mpecss as a callable."""
    assert callable(mpecss.run_mpecss)
