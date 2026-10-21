"""The edges of the binding: id buffers, errors, lifecycle, threads, the library itself."""
import gc
import os
import threading
import unittest
import weakref
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from support import EDGE_TEXTS, GGUF, needs_gguf, needs_numpy, numpy, toknroll
from toknroll import _native as N


@needs_gguf
class IdBufferTest(unittest.TestCase):
    """decode takes any int32 buffer in place, and any iterable of int by one copy."""

    @classmethod
    def setUpClass(cls):
        cls.t = toknroll.load(GGUF)
        cls.ids = cls.t.encode("Hello, Tok'n'Roll! 🎸")
        cls.text = "Hello, Tok'n'Roll! 🎸"

    @classmethod
    def tearDownClass(cls):
        cls.t.close()

    def test_every_accepted_shape(self):
        ids, text = self.ids, self.text
        shapes = {
            "array('i')": ids,
            "list": list(ids),
            "tuple": tuple(ids),
            "generator": (i for i in ids),
            "range": None,
            "memoryview": memoryview(ids),
            "array('l')": array("l", ids),
            "array('q')": array("q", ids),
        }
        for name, value in shapes.items():
            if value is None:
                continue
            with self.subTest(shape=name):
                self.assertEqual(self.t.decode(value), text)
        self.assertEqual(self.t.decode(range(0)), "")
        self.assertEqual(self.t.decode(iter([])), "")
        with self.assertRaises(TypeError) as e:  # bytes: ids, or bytes of ids? refused rather than guessed
            self.t.decode(ids.tobytes())
        self.assertIn("ambiguous", str(e.exception))
        self.assertEqual(self.t.decode(memoryview(ids.tobytes()).cast("i")), text)  # the explicit form works

    @needs_numpy
    def test_numpy_in_place_and_by_copy(self):
        ids, text = self.ids, self.text
        view = numpy.frombuffer(ids, dtype=numpy.int32)
        self.assertTrue(numpy.shares_memory(view, numpy.frombuffer(ids, dtype=numpy.int32)))
        self.assertEqual(self.t.decode(view), text)                           # int32: in place
        self.assertEqual(self.t.decode(view.astype(numpy.int64)), text)       # int64: copied
        self.assertEqual(self.t.decode(view.astype(numpy.uint32)), text)
        self.assertEqual(self.t.decode(view[::-1][::-1]), text)                # a non-contiguous view
        ro = view.copy(); ro.flags.writeable = False
        self.assertEqual(self.t.decode(ro), text)                             # read-only: copied
        self.assertEqual(self.t.decode(numpy.array([], dtype=numpy.int32)), "")

    def test_rejected_ids(self):
        for bad, kind in (([1.5], TypeError), (["a"], TypeError), ([2**40], TypeError), ([None], TypeError),
                          ("string", TypeError), (5, TypeError), ([-1], toknroll.InvalidArgument),
                          ([10**9], toknroll.InvalidArgument), ([self.t.vocabulary_size], toknroll.InvalidArgument)):
            with self.subTest(bad=bad), self.assertRaises(kind) as e:
                self.t.decode(bad)
            if kind is toknroll.InvalidArgument:
                self.assertIsInstance(e.exception, ValueError)
                self.assertIn("outside the vocabulary", str(e.exception))
                self.assertIn("index 0", str(e.exception))

    def test_a_million_ids(self):
        ids = self.t.encode("word " * 250_000)
        self.assertGreater(len(ids), 250_000)
        self.assertEqual(self.t.decode(ids), "word " * 250_000)
        self.assertEqual(len(self.t.decode_bytes(ids)), len("word " * 250_000))


@needs_gguf
class LifecycleTest(unittest.TestCase):
    def test_close_is_idempotent_and_final(self):
        t = toknroll.load(GGUF)
        t.close(); t.close()
        for call in (lambda: t.encode("x"), lambda: t.decode([1]), lambda: t.count_tokens("x"),
                     lambda: t.encode_batch(["x"]), lambda: t.token(1), lambda: t.id("x"),
                     lambda: t.token_flags(1), lambda: t.vocabulary_size):
            with self.assertRaises(ValueError) as e:
                call()
            self.assertIn("closed", str(e.exception))

    def test_with_block_closes(self):
        with toknroll.load(GGUF) as t:
            self.assertGreater(t.vocabulary_size, 0)
        self.assertEqual(repr(t), "Tokenizer(closed)")

    def test_garbage_collection_frees(self):
        t = toknroll.load(GGUF)
        ref = weakref.ref(t)
        del t
        gc.collect()
        self.assertIsNone(ref())

    def test_a_failed_load_leaves_nothing_half_built(self):
        with self.assertRaises(toknroll.NotFound):
            toknroll.load("/nope.gguf")
        gc.collect()  # a half-built object would raise in __del__ here


@needs_gguf
class ThreadTest(unittest.TestCase):
    def test_one_tokenizer_many_threads(self):
        with toknroll.load(GGUF) as t:
            expected = {text: t.encode(text) for text in EDGE_TEXTS}

            def work(k):
                out = {}
                for text in EDGE_TEXTS:
                    ids = t.encode(text)
                    out[text] = ids
                    assert t.decode(ids) == text
                    assert t.count_tokens(text) == len(ids)
                return out

            with ThreadPoolExecutor(32) as pool:
                for result in pool.map(work, range(64)):
                    self.assertEqual(result, expected)

    def test_errors_stay_on_their_thread(self):
        with toknroll.load(GGUF) as t:
            seen = []
            stop = threading.Event()

            def failing():
                while not stop.is_set():
                    try:
                        t.decode([10**9])
                    except toknroll.InvalidArgument:
                        seen.append(True)

            thread = threading.Thread(target=failing)
            thread.start()
            try:
                for _ in range(500):
                    self.assertEqual(t.decode(t.encode("fine")), "fine")
                    self.assertEqual(t.token(1), t.token(1))
            finally:
                stop.set(); thread.join()
            self.assertTrue(seen)

    def test_batch_and_singles_together(self):
        with toknroll.load(GGUF) as t:
            texts = EDGE_TEXTS * 20
            with ThreadPoolExecutor(8) as pool:
                futures = [pool.submit(t.encode_batch, texts) for _ in range(4)]
                futures += [pool.submit(lambda: [t.encode(x) for x in texts]) for _ in range(4)]
                results = [f.result() for f in futures]
            self.assertTrue(all(r == results[0] for r in results))


class ErrorTypesTest(unittest.TestCase):
    def test_hierarchy(self):
        self.assertTrue(issubclass(toknroll.InvalidArgument, toknroll.ToknrollError))
        self.assertTrue(issubclass(toknroll.InvalidArgument, ValueError))
        self.assertTrue(issubclass(toknroll.NotFound, toknroll.ToknrollError))
        self.assertTrue(issubclass(toknroll.NotFound, FileNotFoundError))
        self.assertEqual(toknroll.InvalidArgument("m", -2).code, -2)
        self.assertEqual(str(toknroll.NotFound("gone", -3)), "gone")

    def test_codes_match_the_header(self):
        header = (Path(__file__).resolve().parents[2] / "include" / "toknroll.h").read_text()
        for name, value in (("OK", N.OK), ("ERROR", N.ERROR), ("EINVAL", N.EINVAL), ("ENOTFOUND", N.ENOTFOUND)):
            line = next(l for l in header.splitlines() if l.startswith(f"#define TOKNROLL_{name} "))
            self.assertEqual(int(line.split()[2]), value, name)

    def test_versions_agree(self):
        root = Path(__file__).resolve().parents[2]
        header = (root / "include" / "toknroll.h").read_text()
        macros = {k: int(header.split(f"#define TOKNROLL_VERSION_{k} ")[1].split()[0]) for k in ("MAJOR", "MINOR", "PATCH")}
        c_version = f"{macros['MAJOR']}.{macros['MINOR']}.{macros['PATCH']}"
        pyproject = (root / "python" / "pyproject.toml").read_text()
        self.assertIn(f'version = "{c_version}"', pyproject)
        self.assertEqual(N.library().toknroll_version().decode(), c_version)

    def test_every_header_function_is_bound(self):
        header = (Path(__file__).resolve().parents[2] / "include" / "toknroll.h").read_text()
        declared = {l.split("(")[0].split()[-1].lstrip("*") for l in header.splitlines() if l.startswith("TOKNROLL_API")}
        lib = N.library()
        for name in declared:
            self.assertIsNotNone(getattr(lib, name).argtypes, f"{name} is declared but not typed in _native.bind")
        self.assertEqual(len(declared), 12)


class LibraryLocationTest(unittest.TestCase):
    def test_platform_names(self):
        folder, name = N._platform()
        self.assertRegex(folder, r"^(linux|darwin|windows)-(x86-64|aarch64|\w+)$")
        self.assertIn(name, ("libtoknroll.so", "libtoknroll.dylib", "toknroll.dll"))

    def test_env_override_points_at_a_missing_file(self):
        saved = os.environ.get("TOKNROLL_LIBRARY")
        os.environ["TOKNROLL_LIBRARY"] = "/no/such/libtoknroll.so"
        try:
            located = N._locate()  # the override is tried first, then the checkout and the package
            self.assertTrue(located.is_file())
        finally:
            if saved is None:
                del os.environ["TOKNROLL_LIBRARY"]
            else:
                os.environ["TOKNROLL_LIBRARY"] = saved


if __name__ == "__main__":
    unittest.main()
