"""toknroll-config: the flags a C build needs, pointing into this install."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import toknroll
from toknroll._native import locate


def flags(*args: str) -> list[str]:
    code = f"from toknroll._config import main; import sys; sys.argv[1:] = {list(args)!r}; raise SystemExit(main())"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return run.stdout.splitlines()


class ConfigTest(unittest.TestCase):
    def test_version_is_the_packages(self):
        self.assertEqual(flags("--version"), [toknroll.__version__])

    def test_cflags_name_the_packages_include_directory(self):
        (cflag,) = flags("--cflags")
        include = Path(cflag[2:])   # -I or /I
        self.assertEqual(include, Path(toknroll.__file__).resolve().parent / "include")
        if os.environ.get("TOKNROLL_INSTALLED"):   # an installed wheel carries the header; a checkout stages it with make python
            self.assertTrue((include / "toknroll.h").is_file(), include)

    def test_libs_name_the_library(self):
        library = locate("toknroll")
        libdir = library.parent
        lines = flags("--libs")
        if sys.platform == "win32":
            self.assertEqual(lines, [f"/LIBPATH:{libdir}", "toknroll.lib"])
        else:
            self.assertEqual(lines, [f"{library} -Wl,-rpath,{libdir}"])
        (pkgconfigdir,) = flags("--pkgconfigdir")
        self.assertEqual(Path(pkgconfigdir), libdir / "pkgconfig")
        if os.environ.get("TOKNROLL_INSTALLED"):
            self.assertTrue((libdir / "pkgconfig" / "toknroll.pc").is_file())

    def test_nothing_asked_is_the_help(self):
        code = "from toknroll._config import main; raise SystemExit(main())"
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 2)
        self.assertIn("--cflags", run.stdout)

    def test_the_installed_script_is_on_the_path(self):
        if not os.environ.get("TOKNROLL_INSTALLED"):
            self.skipTest("the console script exists in an installed wheel")
        script = Path(sys.executable).parent / ("toknroll-config.exe" if sys.platform == "win32" else "toknroll-config")
        run = subprocess.run([str(script), "--version"], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.stdout.strip(), toknroll.__version__, run.stderr)
