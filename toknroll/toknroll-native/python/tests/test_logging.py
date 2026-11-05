"""TOKNROLL_LOG says how many lines, toknroll_log_set where: the package's callback sends them to
logging.getLogger("toknroll"), the bare library to stderr. Each case runs in a fresh interpreter, as the
library reads both once, at start. The warning every case can count on, offline: an option that replaces a
registered pre-tokenizer, logged as the first GGUF loads."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from support import HUB_JSON, fresh, needs_hub, tiny_gguf
REPLACE = "-Dtoknroll.gguf.pre.gpt2=alias:qwen2"
REPLACED = REPLACE + " replaces the registered pre-tokenizer 'gpt2'"

# the package, its records kept: [[level name, message], ...] on the last line of stdout
RECORDS = ("import json, logging\n"
           "lines = []\n"
           "class Keep(logging.Handler):\n"
           "    def emit(self, record): lines.append([record.levelname, record.getMessage()])\n"
           "logging.getLogger('toknroll').addHandler(Keep())\n"
           "logging.getLogger('toknroll').setLevel(logging.DEBUG)\n")
PRINT = "print(json.dumps(lines))\n"
# the library alone, bound but no callback set: its lines go to stderr
BARE = ("import ctypes\n"
        "from toknroll import _native as N\n"
        "lib = N.bind(ctypes.CDLL(str(N.locate('toknroll'))))\n"
        "def configure(*options):\n"
        "    assert lib.toknroll_configure((ctypes.c_char_p * len(options))(*(o.encode() for o in options)), len(options)) == 0\n")


def root(cache: str) -> str:
    return "-Dtoknroll.cache.root=" + cache


def records(run: subprocess.CompletedProcess) -> list:
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.gguf = tiny_gguf(Path(self.dir.name) / "tiny.gguf")

    def tearDown(self):
        self.dir.cleanup()


class PackageTest(Fixture):
    def warned(self, replace: bool = True) -> str:
        """the GGUF loaded, after the option that makes its loader warn"""
        return (f"toknroll.configure({REPLACE!r})\n" if replace else "") + f"toknroll.load({self.gguf!r})\n"

    def test_a_warning_reaches_logging_not_stderr(self):
        run = fresh(RECORDS + self.warned() + PRINT)
        self.assertEqual(records(run), [["WARNING", "com.qxotic.toknroll.gguf.GGUFTokenizerLoader: " + REPLACED]])
        self.assertEqual(run.stderr, "")

    def test_a_load_without_a_warning_logs_nothing(self):
        run = fresh(RECORDS + self.warned(replace=False) + PRINT)
        self.assertEqual(records(run), [])
        self.assertEqual(run.stderr, "")

    def test_levels(self):
        for log, want in (("off", 0), ("OFF", 0), ("error", 0), (" Error ", 0), ("warn", 1), ("debug", 1), ("", 1)):
            with self.subTest(log=log):
                self.assertEqual(len(records(fresh(RECORDS + self.warned() + PRINT, TOKNROLL_LOG=log))), want)

    def test_an_unknown_value_warns_once_and_means_warn(self):
        lines = records(fresh(RECORDS + self.warned() + PRINT, TOKNROLL_LOG="verbose"))
        self.assertEqual(lines, [["WARNING", "toknroll.native: TOKNROLL_LOG=verbose is not off, error, warn,"
                                             " info or debug: warn applies"],
                                 ["WARNING", "com.qxotic.toknroll.gguf.GGUFTokenizerLoader: " + REPLACED]])

    def test_a_reload_keeps_the_callback_the_library_calls(self):
        run = fresh(RECORDS + "import gc, importlib\n"
                    "from toknroll import _native as N\n"
                    "N.library()\n"                       # the callback set
                    "importlib.reload(N); gc.collect()\n"
                    + self.warned() + PRINT)             # and called, after the reload
        self.assertEqual(len(records(run)), 1, run.stderr)

    def test_log_set_is_once_before_any_other_call(self):
        run = fresh("from toknroll import _native as N\n"
                    "lib = N.library()\n"  # which set the package's callback
                    "again = lib.toknroll_log_set(N.log_callback('toknroll'), None), lib.toknroll_last_error(None)\n"
                    "lib.toknroll_tokenizer_load(b'/no/such/tokenizer.json')\n"
                    "late = lib.toknroll_log_set(N.log_callback('toknroll'), None), lib.toknroll_last_error(None)\n"
                    "print(again[0], again[1].decode()); print(late[0], late[1].decode())")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines(),
                         ["-2 toknroll_log_set must come once, before any other call into the library"] * 2)

    @needs_hub
    def test_a_download_logs_at_info_only(self):
        for log, downloads in ((None, False), ("off", False), ("info", True)):
            with self.subTest(log=log), tempfile.TemporaryDirectory() as cache:
                code = f"toknroll.configure({root(cache)!r})\ntoknroll.load({HUB_JSON!r})\n"
                run = fresh(RECORDS + code + PRINT, TOKNROLL_LOG=log, HF_HOME=os.path.join(cache, "hf"))  # the HF cache there too
                lines = records(run)
                self.assertEqual(any(level == "INFO" and "jinfer.hub: download" in m for level, m in lines), downloads, lines)
                self.assertEqual(run.stderr, "", "the library wrote to stderr")


class BareLibraryTest(Fixture):
    def bare(self, before: str, log: str | None = None) -> subprocess.CompletedProcess:
        run = fresh(BARE + before + f"configure({REPLACE!r})\nassert lib.toknroll_tokenizer_load({self.gguf.encode()!r})\n", TOKNROLL_LOG=log)
        self.assertEqual(run.returncode, 0, run.stderr)
        return run

    def test_without_a_callback_lines_go_to_stderr(self):
        self.assertEqual(self.bare("").stderr.splitlines(),
                         ["toknroll: warn: com.qxotic.toknroll.gguf.GGUFTokenizerLoader: " + REPLACED])

    def test_an_unknown_value_on_stderr(self):
        self.assertEqual(self.bare("", "bogus").stderr.splitlines()[0],
                         "toknroll: warn: toknroll.native: TOKNROLL_LOG=bogus is not off, error, warn, info or debug:"
                         " warn applies")

    def test_a_null_callback_silences_everything(self):
        self.assertEqual(self.bare("assert lib.toknroll_log_set(N.LogFn(), None) == 0\n", "debug").stderr, "")

    @needs_hub
    def test_info_writes_a_download_to_stderr(self):
        with tempfile.TemporaryDirectory() as cache:
            run = fresh(BARE + f"configure({root(cache)!r})\nassert lib.toknroll_tokenizer_load({HUB_JSON.encode()!r})",
                        TOKNROLL_LOG="info", HF_HOME=os.path.join(cache, "hf"))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("toknroll: info: jinfer.hub: download", run.stderr)


if __name__ == "__main__":
    unittest.main()
