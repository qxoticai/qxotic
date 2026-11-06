"""python -m toknroll: the flags a C build needs, pointing into this install."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import toknroll


def flags(*args: str) -> list[str]:
    run = subprocess.run([sys.executable, "-m", "toknroll", *args], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return run.stdout.splitlines()


class MainTest(unittest.TestCase):
    def test_version_is_the_packages(self):
        self.assertEqual(flags("--version"), [toknroll.__version__])

    def test_cflags_name_the_packages_include_directory(self):
        (cflag,) = flags("--cflags")
        include = Path(cflag[2:])   # -I or /I
        self.assertEqual(include, Path(toknroll.__file__).resolve().parent / "include")
        if os.environ.get("TOKNROLL_INSTALLED"):   # an installed wheel carries the header; a checkout stages it with make python
            self.assertTrue((include / "toknroll.h").is_file(), include)

    def test_libs_name_the_library_directory(self):
        lines = flags("--libs")
        from toknroll._native import locate
        library = locate("toknroll")
        libdir = library.parent
        if sys.platform == "win32":
            self.assertEqual(lines, [f"/LIBPATH:{libdir}", "toknroll.lib"])
        else:
            self.assertEqual(lines, [f"{library} -Wl,-rpath,{libdir}"])
        (pkgconfig,) = flags("--pkgconfig")
        self.assertEqual(Path(pkgconfig), libdir / "pkgconfig")
        if os.environ.get("TOKNROLL_INSTALLED"):
            self.assertTrue((libdir / "pkgconfig" / "toknroll.pc").is_file())

    def test_nothing_asked_is_the_help(self):
        run = subprocess.run([sys.executable, "-m", "toknroll"], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 2)
        self.assertIn("--cflags", run.stdout)
