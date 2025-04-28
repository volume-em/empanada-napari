#!/usr/bin/env python
import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install

class CustomInstall(install):
    """After installing this package, immediately uninstall bad_dep."""
    def run(self):
        # 1) do the normal install
        super().run()

        # 2) try to uninstall bad_dep if it slipped in
        try:
            __import__("empanada")
        except ImportError:
            # it wasn’t installed—nothing to do
            return

        print("➤ uninstalling bad_dep…")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall",
                "empanada-dl", "-y"
            ])
        except subprocess.CalledProcessError:
            sys.stderr.write("⚠️  Warning: failed to uninstall bad_dep\n")

# This setup() will automatically load all metadata from setup.cfg
setup(
    cmdclass={
        'install': CustomInstall,
    },
)
