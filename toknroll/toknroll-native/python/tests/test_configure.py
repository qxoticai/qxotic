"""toknroll.configure: runtime options before first use. Each case runs in a fresh interpreter."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

from support import GGUF, needs_gguf, toknroll

HERE = Path(__file__).resolve().parents[1]


def fresh(code: str, options: str | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "TOKNROLL_OPTIONS"}
    if options is not None:
        env["TOKNROLL_OPTIONS"] = options
    return subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(HERE)!r}); import toknroll\n{code}"],
                          capture_output=True, text=True, timeout=120, env=env)


@needs_gguf
class ConfigureTest(unittest.TestCase):
    def test_a_pre_tokenizer_alias_changes_the_split(self):
        text = "Numbers 1234567 and words"
        default = fresh(f"t = toknroll.load({GGUF!r}); print(list(t.encode({text!r})))")
        aliased = fresh(f"toknroll.configure('-Dtoknroll.gguf.pre.lfm2=alias:gpt2'); t = toknroll.load({GGUF!r}); print(list(t.encode({text!r})))")
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(aliased.returncode, 0, aliased.stderr)
        self.assertNotEqual(default.stdout, aliased.stdout, "aliasing LFM2's pre-tokenizer to GPT-2's must change the ids")

    def test_a_second_call_is_refused(self):
        out = fresh("toknroll.configure('-Dtoknroll.a=1', '-Dtoknroll.b=2')\n"
                    "try:\n    toknroll.configure('-Dtoknroll.c=3')\nexcept toknroll.InvalidArgument as e:\n    print('refused:', e)\n"
                    f"t = toknroll.load({GGUF!r}); print(t.vocabulary_size)")
        self.assertIn("refused: toknroll_configure must come once", out.stdout, out.stderr)
        self.assertIn("65536", out.stdout, out.stderr)

    def test_refused_after_first_use(self):
        out = fresh(f"t = toknroll.load({GGUF!r})\n"
                    "try:\n    toknroll.configure('-Dtoknroll.x=1')\nexcept toknroll.InvalidArgument as e:\n    print('refused:', e)")
        self.assertIn("refused: toknroll_configure must come once", out.stdout, out.stderr)

    def test_a_rejected_option_fails_configure_itself(self):
        out = fresh("try:\n    toknroll.configure('-XX:NoSuchOptionAtAll=1')\nexcept toknroll.ToknrollError as e:\n    print('failed:', e)")
        self.assertIn("failed: the runtime could not be started: an option from toknroll_configure was rejected", out.stdout, out.stdout + out.stderr)
        self.assertIn("NoSuchOptionAtAll", out.stderr, "the runtime names the option on stderr")

    def test_the_environment_has_the_last_word(self):
        text = "Numbers 1234567 and words"
        encode = f"t = toknroll.load({GGUF!r}); print(list(t.encode({text!r})))"
        gpt2 = fresh(f"toknroll.configure('-Dtoknroll.gguf.pre.lfm2=alias:gpt2'); {encode}")
        qwen2 = fresh(encode, options="-Dtoknroll.gguf.pre.lfm2=alias:qwen2")
        both = fresh(f"toknroll.configure('-Dtoknroll.gguf.pre.lfm2=alias:gpt2'); {encode}", options="-Dtoknroll.gguf.pre.lfm2=alias:qwen2")
        for run in (gpt2, qwen2, both):
            self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotEqual(gpt2.stdout, qwen2.stdout, "the two aliases must split differently for the test to mean anything")
        self.assertEqual(both.stdout, qwen2.stdout, "TOKNROLL_OPTIONS must override what configure set")

    def test_a_rejected_environment_option_names_the_variable(self):
        out = fresh(f"try:\n    toknroll.load({GGUF!r})\nexcept toknroll.ToknrollError as e:\n    print('failed:', e)", options="-XX:NoSuchOptionAtAll=1")
        self.assertIn("TOKNROLL_OPTIONS", out.stdout, out.stdout + out.stderr)
        self.assertIn("NoSuchOptionAtAll", out.stderr, "the runtime names the option on stderr")

    def test_bad_arguments(self):
        with self.assertRaises(TypeError):
            toknroll.configure(5)
        out = fresh("toknroll.configure()\nprint('empty ok')")
        self.assertIn("empty ok", out.stdout, out.stderr)


if __name__ == "__main__":
    unittest.main()
