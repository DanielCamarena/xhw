# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Full-suite acceptance requires development dependencies and forbids skipped checks."""

import importlib.util

import pytest


def pytest_sessionstart(session):
    required = ("dask", "h5netcdf", "h5py", "statsmodels", "pygam", "matplotlib",
                "plotly", "ipywidgets", "PIL")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise pytest.UsageError(f"Install xhw[dev] before running tests; missing: {missing}")


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None and reporter.stats.get("skipped"):
        session.exitstatus = 1
        reporter.write_sep("=", "Acceptance requires zero skipped checks")
