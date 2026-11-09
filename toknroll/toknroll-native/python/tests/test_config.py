"""toknroll-config: the flags a C build needs, pointing into this install."""
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import toknroll
from toknroll._native import locate


def flags(*args: str, env=None) -> list[str]:
    code = f"from toknroll._config import main; import sys; sys.argv[1:] = {list(args)!r}; raise SystemExit(main())"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)
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
            self.assertEqual(lines, [str(library), f"-Wl,-rpath,{libdir}"])
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

    @unittest.skipUnless(os.environ.get("TOKNROLL_INSTALLED") and os.name != "nt", "installed POSIX wheel and C compiler")
    def test_c_consumer_under_a_path_with_spaces(self):
        from support import tiny_gguf

        example = Path(__file__).resolve().parents[2] / "examples/c/tokenize.c"
        with tempfile.TemporaryDirectory(prefix="toknroll consumer ") as directory:
            root = Path(directory)
            shutil.copytree(Path(toknroll.__file__).resolve().parent, root / "toknroll")
            env = dict(os.environ, PYTHONPATH=str(root))
            for name in ("TOKNROLL_LIBRARY", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
                env.pop(name, None)
            fixture = tiny_gguf(root / "tiny.gguf")
            executable = root / "tokenize"
            compiler = shlex.split(os.environ.get("CC", "cc"))
            for route in ("config", "pkg-config"):
                with self.subTest(route=route):
                    if route == "config":
                        arguments = flags("--cflags", "--libs", env=env)
                    else:
                        if not shutil.which("pkg-config"):
                            self.skipTest("pkg-config not installed")
                        (env["PKG_CONFIG_PATH"],) = flags("--pkgconfigdir", env=env)
                        arguments = shlex.split(subprocess.check_output(
                            ["pkg-config", "--cflags", "--libs", "toknroll"], env=env, text=True, timeout=60))
                    built = subprocess.run(compiler + ["-std=c99", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                                                      str(example), "-o", str(executable), *arguments],
                                           env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(built.returncode, 0, built.stderr)
                    for text in ("", "hello", "hello " * 400):
                        with self.subTest(text_length=len(text)):
                            run = subprocess.run([str(executable), fixture, text], env=env,
                                                 capture_output=True, text=True, timeout=60)
                            self.assertEqual(run.returncode, 0, run.stderr)
                            self.assertEqual(run.stderr, "")
                            lines = run.stdout.splitlines()
                            self.assertEqual(len(lines), 3, run.stdout)
                            self.assertEqual(lines[1], text)
                            if not text:
                                self.assertEqual(lines[0], "")
                                self.assertTrue(lines[2].startswith("0 tokens,"), lines[2])
                    with self.subTest(source="missing"):
                        run = subprocess.run([str(executable), str(root / "missing.gguf"), "hello"], env=env,
                                             capture_output=True, text=True, timeout=60)
                        self.assertEqual(run.returncode, 1, run.stderr)
                        self.assertEqual(run.stdout, "")
                        self.assertRegex(run.stderr, r"^load: .* \(-3\)\n$")
