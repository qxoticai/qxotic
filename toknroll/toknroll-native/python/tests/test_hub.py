"""Hub sources: tokenizer.json repositories and GGUF headers, metadata only. TOKNROLL_OFFLINE=1 skips."""
import tempfile
import unicodedata
import unittest

from support import EDGE_TEXTS, GGUF, HUB_GGUF, HUB_JSON, fresh, needs_hub, toknroll


@needs_hub
class HubTest(unittest.TestCase):
    def test_tokenizer_json_repository(self):
        with toknroll.load(HUB_JSON + "@c1899de289a04d12100db370d81485cdf75e47ca") as q:  # pinned: the ids below are this revision's
            self.assertEqual(list(q.encode("Hello, world!")), [9707, 11, 1879, 0])
            self.assertEqual(q.id("<|im_end|>"), 151645)
            self.assertEqual(q.vocabulary_size, 151669)
            for text in EDGE_TEXTS:  # Qwen3's tokenizer.json normalizes to NFC, as HF does: the round trip is NFC-exact
                with self.subTest(text=text[:30]):
                    self.assertEqual(q.decode(q.encode(text)), unicodedata.normalize("NFC", text))

    def test_a_tiktoken_repository_warmed_online_loads_offline(self):
        """A hub tiktoken checkout keeps its files in the loader's own cache: offline, the same root serves it."""
        with tempfile.TemporaryDirectory() as root:
            def run(source, *options):  # configure is once per process: each load in its own
                return fresh(f"toknroll.configure('-Dtoknroll.cache.root={root}', *{options!r})\n"
                             f"print(toknroll.load({source!r}).vocabulary_size)")
            warm = run("moonshotai/Kimi-K3")
            self.assertEqual(warm.returncode, 0, warm.stderr)
            cold = run("moonshotai/Kimi-K3", "-Dtoknroll.offline=true")
            self.assertEqual(cold.stdout, warm.stdout, cold.stderr)
            missing = run("moonshotai/Kimi-K2.6", "-Dtoknroll.offline=true")  # another repository: not served
            self.assertIn("is not cached, and offline mode never fetches", missing.stderr)

    def test_the_host_may_be_written(self):
        with toknroll.load(HUB_JSON) as a, toknroll.load("hf.co/" + HUB_JSON) as b, toknroll.load("huggingface.co/" + HUB_JSON) as c:
            self.assertEqual(a.encode("same"), b.encode("same"))
            self.assertEqual(a.encode("same"), c.encode("same"))

    def test_a_revision_pins_the_repository(self):
        with toknroll.load(HUB_JSON) as a, toknroll.load(HUB_JSON + "@main") as b:
            self.assertEqual(a.encode("same"), b.encode("same"))

    def test_a_gguf_named_by_its_path_is_the_quant_form(self):
        with toknroll.load(HUB_GGUF) as quant, toknroll.load("LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf") as path:
            self.assertEqual(quant.encode("Hello, world!"), path.encode("Hello, world!"))
            self.assertEqual(quant.vocabulary_size, path.vocabulary_size)

    def test_a_gguf_only_repository_root_is_not_found_and_names_its_quants(self):
        with self.assertRaises(toknroll.NotFound) as e:
            toknroll.load("LiquidAI/LFM2.5-350M-GGUF")
        self.assertIn("Q8_0", str(e.exception))

    def test_a_misspelled_host_is_invalid_not_an_owner(self):
        with self.assertRaises(toknroll.InvalidArgument) as e:
            toknroll.load("huggingface.com/" + HUB_JSON)
        self.assertIn("modelscope.cn", str(e.exception))

    def test_gguf_header_from_the_hub_matches_the_local_file(self):
        with toknroll.load(HUB_GGUF) as remote, toknroll.load(GGUF) as local:
            for text in EDGE_TEXTS[:20]:
                self.assertEqual(remote.encode(text), local.encode(text))
            self.assertEqual(remote.vocabulary_size, local.vocabulary_size)

    def test_a_missing_repository_is_an_error_with_the_reference_in_it(self):
        with self.assertRaises(toknroll.ToknrollError) as e:
            toknroll.load("LiquidAI/no-such-repository-xyz")
        self.assertIn("no-such-repository-xyz", str(e.exception))

    def test_a_malformed_reference(self):
        for ref in ("owner/", "/x/y:Q8", "owner/repo:"):
            with self.subTest(ref=ref), self.assertRaises(toknroll.ToknrollError):
                toknroll.load(ref)


if __name__ == "__main__":
    unittest.main()
