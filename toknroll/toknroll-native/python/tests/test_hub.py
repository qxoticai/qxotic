"""Hub sources: tokenizer.json repositories and GGUF headers, metadata only. TOKNROLL_OFFLINE=1 skips."""
import unicodedata
import unittest

from support import EDGE_TEXTS, GGUF, HUB_GGUF, HUB_JSON, needs_gguf, needs_hub, toknroll


@needs_hub
class HubTest(unittest.TestCase):
    def test_tokenizer_json_repository(self):
        with toknroll.load(HUB_JSON) as q:
            self.assertEqual(list(q.encode("Hello, world!")), [9707, 11, 1879, 0])
            self.assertEqual(q.id("<|im_end|>"), 151645)
            self.assertEqual(q.vocabulary_size, 151669)
            for text in EDGE_TEXTS:  # Qwen3's tokenizer.json normalizes to NFC, as HF does: the round trip is NFC-exact
                with self.subTest(text=text[:30]):
                    self.assertEqual(q.decode(q.encode(text)), unicodedata.normalize("NFC", text))

    def test_hf_prefix_is_the_default_hub(self):
        with toknroll.load(HUB_JSON) as a, toknroll.load("hf:" + HUB_JSON) as b:
            self.assertEqual(a.encode("same"), b.encode("same"))

    @needs_gguf
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
        for ref in ("a/b/c/d", "owner/", "/x/y:Q8", "owner/repo:"):
            with self.subTest(ref=ref), self.assertRaises(toknroll.ToknrollError):
                toknroll.load(ref)


if __name__ == "__main__":
    unittest.main()
