# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Verify that the installed starter imports and exposes its distribution version."""

from importlib.metadata import version

import xhw


def test_installed_package_version():
    assert xhw.__version__ == version("xhw")
