"""The tokenizer: loading, encode and decode, batches, the vocabulary, flags."""
import os
import pickle
import tempfile
import unittest
from array import array
from pathlib import Path

from support import EDGE_TEXTS, GGUF, needs_gguf, needs_numpy, numpy, toknroll

F = toknroll.TokenFlags


@needs_gguf
class LoadTest(unittest.TestCase):
    def test_load_from_path_str_and_pathlike(self):
        with toknroll.load(GGUF) as a, toknroll.load(Path(GGUF)) as b:
            self.assertEqual(a.encode("x"), b.encode("x"))
            self.assertEqual(a.source, b.source)

    def test_source_is_absolute_even_from_a_relative_path(self):
        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(GGUF))
            with toknroll.load(os.path.basename(GGUF)) as t:
                self.assertTrue(os.path.isabs(t.source), t.source)
                self.assertEqual(t.source, GGUF)
        finally:
            os.chdir(cwd)

    def test_repr_names_source_and_vocabulary(self):
        with toknroll.load(GGUF) as t:
            self.assertEqual(repr(t), f"Tokenizer({GGUF!r}, vocabulary_size={t.vocabulary_size})")
        self.assertEqual(repr(t), "Tokenizer(closed)")

    def test_many_loads_and_closes(self):
        for _ in range(50):
            toknroll.load(GGUF).close()

    def test_load_rejects_wrong_types(self):
        for bad in (None, 42, b"bytes", ["list"]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                toknroll.load(bad)

    @needs_gguf
    def test_a_gguf_loads_by_its_content_whatever_its_name(self):
        with tempfile.TemporaryDirectory() as d:   # an Ollama blob: sha256-<hex>, no extension
            blob = Path(d) / "sha256-3f1a"
            blob.write_bytes(Path(GGUF).read_bytes()[: 32 << 20])   # the header is all a tokenizer reads
            with toknroll.load(GGUF) as named, toknroll.load(str(blob)) as unnamed:
                self.assertEqual(named.encode("Hello, Tok'n'Roll!"), unnamed.encode("Hello, Tok'n'Roll!"))

    def test_a_file_of_no_known_kind_is_invalid(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("not a tokenizer")
        with self.assertRaises(toknroll.InvalidArgument) as e:
            toknroll.load(f.name)
        self.assertIn("GGUF", str(e.exception))

    def test_load_errors_are_typed(self):
        with self.assertRaises(toknroll.NotFound) as e:
            toknroll.load("/no/such/dir/tokenizer.json")
        self.assertIsInstance(e.exception, FileNotFoundError)
        self.assertEqual(e.exception.code, -3)
        self.assertIn(str(Path("/no/such/dir/tokenizer.json")), str(e.exception))   # as the platform spells it
        with self.assertRaises(toknroll.InvalidArgument) as e:
            toknroll.load("")
        self.assertIsInstance(e.exception, ValueError)
        with self.assertRaises(toknroll.InvalidArgument):
            toknroll.load("not-a-reference")
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(toknroll.ToknrollError):
                toknroll.load(empty)


@needs_gguf
class EncodeDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = toknroll.load(GGUF)

    @classmethod
    def tearDownClass(cls):
        cls.t.close()

    def test_ids_are_a_typed_array(self):
        ids = self.t.encode("Hello, world!")
        self.assertIsInstance(ids, array)
        self.assertEqual(ids.typecode, "i")
        self.assertEqual(ids.itemsize, 4)
        self.assertTrue(all(0 <= i < self.t.vocabulary_size for i in ids))

    def test_every_edge_text_round_trips(self):
        for text in EDGE_TEXTS:
            with self.subTest(text=text[:30]):
                ids = self.t.encode(text)
                self.assertEqual(self.t.decode(ids), text)
                self.assertEqual(self.t.decode_bytes(ids), text.encode())
                self.assertEqual(self.t.count_tokens(text), len(ids))

    def test_nul_bytes_are_not_a_string_terminator(self):
        self.assertEqual(self.t.decode(self.t.encode("a\x00b")), "a\x00b")
        self.assertGreater(self.t.count_tokens("a\x00b"), self.t.count_tokens("a"))
        self.assertIsNone(self.t.id("a\x00b"))

    def test_empty(self):
        self.assertEqual(len(self.t.encode("")), 0)
        self.assertEqual(self.t.count_tokens(""), 0)
        self.assertEqual(self.t.decode([]), "")
        self.assertEqual(self.t.decode_bytes(array("i")), b"")

    def test_long_text_past_the_first_pass_estimate(self):
        for text in ("x" * 100_000, "🎸" * 20_000, "word " * 200_000):
            with self.subTest(text=text[:8]):
                ids = self.t.encode(text)
                self.assertEqual(self.t.decode(ids), text)
                self.assertEqual(self.t.count_tokens(text), len(ids))

    def test_a_split_character_decodes_as_bytes_or_per_errors(self):
        emoji = self.t.encode("🎸")
        self.assertGreater(len(emoji), 1, "LFM2 splits this emoji into byte tokens")
        raw = self.t.decode_bytes(emoji[:1])
        self.assertTrue("🎸".encode().startswith(raw))
        self.assertEqual(self.t.decode(emoji[:1]), raw.decode("utf-8", "replace"))
        self.assertIn("�", self.t.decode(emoji[:1]))
        self.assertEqual(self.t.decode(emoji[:1], errors="ignore"), "")
        with self.assertRaises(UnicodeDecodeError):
            self.t.decode(emoji[:1], errors="strict")

    def test_decode_is_deterministic_and_prefix_consistent(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        ids = self.t.encode(text)
        for n in (0, 1, 7, len(ids) // 2, len(ids)):
            self.assertTrue(text.startswith(self.t.decode(ids[:n]).rstrip("�")) or n == 0)
        self.assertEqual(self.t.encode(text), ids)

    def test_specials_are_text_by_default_and_whole_on_request(self):
        plain, kept = self.t.encode("<|im_end|>"), self.t.encode("<|im_end|>", specials=True)
        self.assertGreater(len(plain), 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(self.t.id("<|im_end|>"), kept[0])
        self.assertEqual(self.t.decode(kept), "<|im_end|>")
        self.assertEqual(self.t.decode(plain), "<|im_end|>")
        self.assertEqual(self.t.encode("no specials here", specials=True), self.t.encode("no specials here"))
        mixed = self.t.encode("Hello<|im_end|>world<|im_start|>", specials=True)
        self.assertEqual(self.t.decode(mixed), "Hello<|im_end|>world<|im_start|>")
        self.assertEqual(sum(1 for i in mixed if self.t.token_flags(i) & (F.CONTROL | F.USER_DEFINED)), 2)
        self.assertEqual(self.t.count_tokens("<|im_end|>", specials=True), 1)

    def test_decode_without_specials_is_a_filter(self):
        ids = self.t.encode("Hello<|im_end|>", specials=True)
        self.assertEqual(self.t.decode(i for i in ids if self.t.token_flags(i) & F.NORMAL), "Hello")

    def test_encode_rejects_non_text(self):
        for bad in (None, 5, b"bytes", ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError) as e:
                    self.t.encode(bad)
                self.assertIn("encode", str(e.exception))
                with self.assertRaises(TypeError):
                    self.t.count_tokens(bad)


@needs_gguf
class BatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = toknroll.load(GGUF)

    @classmethod
    def tearDownClass(cls):
        cls.t.close()

    def test_batch_equals_one_by_one(self):
        texts = EDGE_TEXTS + [f"document {i} " * (i % 17) for i in range(300)]
        self.assertEqual(self.t.encode_batch(texts), [self.t.encode(x) for x in texts])
        self.assertEqual(self.t.encode_batch(texts, specials=True), [self.t.encode(x, specials=True) for x in texts])

    def test_batch_edges(self):
        self.assertEqual(self.t.encode_batch([]), [])
        self.assertEqual(self.t.encode_batch([""]), [array("i")])
        self.assertEqual(self.t.encode_batch(["", "", ""]), [array("i")] * 3)
        self.assertEqual(self.t.encode_batch(("tuple", "input")), [self.t.encode("tuple"), self.t.encode("input")])
        big = self.t.encode_batch(["x" * 100_000, "y", "", "🎸" * 1000])
        self.assertEqual(big, [self.t.encode("x" * 100_000), self.t.encode("y"), array("i"), self.t.encode("🎸" * 1000)])

    def test_batch_of_ten_thousand(self):
        texts = [f"text {i}: " + "words " * (i % 40) for i in range(10_000)]
        out = self.t.encode_batch(texts)
        self.assertEqual(len(out), 10_000)
        self.assertEqual(self.t.decode_batch(out), texts)
        self.assertEqual(sum(map(len, out)), sum(self.t.count_tokens(x) for x in texts))

    def test_batch_rejects_non_text(self):
        with self.assertRaises(TypeError) as e:
            self.t.encode_batch(["ok", 3])
        self.assertIn("encode_batch", str(e.exception))

    def test_decode_batch(self):
        batches = [self.t.encode(x) for x in EDGE_TEXTS]
        self.assertEqual(self.t.decode_batch(batches), EDGE_TEXTS)
        self.assertEqual(self.t.decode_batch([]), [])


@needs_gguf
class VocabularyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = toknroll.load(GGUF)

    @classmethod
    def tearDownClass(cls):
        cls.t.close()

    def test_size_and_round_trips(self):
        n = self.t.vocabulary_size
        self.assertEqual(n, 65536)
        for i in range(0, n, 509):
            spelling = self.t.token(i)
            self.assertIsInstance(spelling, str)
            self.assertEqual(self.t.id(spelling), i, f"id {i} spelled {spelling!r}")

    def test_absent(self):
        for i in (n := self.t.vocabulary_size, n + 1, 10**9, -1, -(10**9)):
            with self.subTest(id=i):
                self.assertIsNone(self.t.token(i))
                self.assertIsNone(self.t.token_flags(i))
        self.assertIsNone(self.t.id("surely not a token"))
        self.assertIsNone(self.t.id(""))

    def test_flags_are_one_kind_bit_each(self):
        kinds = F.NORMAL | F.UNKNOWN | F.CONTROL | F.USER_DEFINED | F.UNUSED | F.BYTE
        for i in range(0, self.t.vocabulary_size, 331):
            flags = self.t.token_flags(i)
            self.assertIsInstance(flags, F)
            self.assertEqual(bin(flags & kinds).count("1"), 1, f"id {i}: {flags!r}")
            self.assertEqual(flags & ~kinds, 0)
        self.assertEqual(self.t.token_flags(self.t.encode("Hello")[0]), F.NORMAL)
        self.assertTrue(self.t.token_flags(self.t.id("<|im_end|>")) & (F.CONTROL | F.USER_DEFINED))

    def test_flag_values_match_the_header(self):
        header = (Path(__file__).resolve().parents[2] / "include" / "toknroll.h").read_text()
        for flag in F:
            self.assertIn(f"#define TOKNROLL_TOKEN_{flag.name}", header)
            shift = int(header.split(f"TOKNROLL_TOKEN_{flag.name}")[1].split("<<")[1].split(")")[0])
            self.assertEqual(flag.value, 1 << shift, flag.name)

    def test_lookups_reject_wrong_types(self):
        for bad in ("x", 1.5, None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.t.token(bad)
                with self.assertRaises(TypeError):
                    self.t.token_flags(bad)
        with self.assertRaises(TypeError):
            self.t.id(5)

    def test_a_long_spelling_round_trips(self):
        longest = max((self.t.token(i) for i in range(0, self.t.vocabulary_size, 97)), key=len)
        self.assertEqual(self.t.id(longest), self.t.id(longest))
        self.assertGreater(len(longest.encode()), 0)


@needs_gguf
class PicklingTest(unittest.TestCase):
    def test_pickles_as_source(self):
        with toknroll.load(GGUF) as t:
            clone = pickle.loads(pickle.dumps(t))
            self.assertEqual(clone.source, t.source)
            self.assertEqual(clone.encode("hello"), t.encode("hello"))
            clone.close()

    def test_processes(self):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        # spawn, Python 3.14's default: a fork of a process with a live runtime inherits none of its threads
        with toknroll.load(GGUF) as t, ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
            self.assertEqual(list(pool.map(t.count_tokens, ["one", "two three", ""])), [1, 2, 0])


if __name__ == "__main__":
    unittest.main()


class BindTest(unittest.TestCase):
    def test_another_toknroll_is_refused_by_version(self):
        class Fn:
            restype = None

            def __call__(self):
                return b"9.9.9"

        class Lib:
            _name = "libother.so"
            toknroll_version = Fn()

        with self.assertRaisesRegex(OSError, f"{toknroll.__version__}.*9.9.9.*libother"):
            toknroll._native.bind(Lib())
