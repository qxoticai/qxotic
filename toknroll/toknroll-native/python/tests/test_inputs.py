"""The edges of the binding: id buffers, errors, lifecycle, threads, the library itself."""
import ctypes
import ctypes.util
import gc
import multiprocessing
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
import unittest.mock
import weakref
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from support import EDGE_TEXTS, GGUF, needs_numpy, numpy, toknroll
from toknroll import _native as N


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
            "memoryview": memoryview(ids),
            "array('l')": array("l", ids),
            "array('q')": array("q", ids),
            "ctypes int32": (ctypes.c_int32 * len(ids))(*ids),  # format '<i': not native to tolist, iterated
            "ctypes int64": (ctypes.c_int64 * len(ids))(*ids),
        }
        for name, value in shapes.items():
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
        self.assertEqual(self.t.decode(view.astype(">i4")), text)             # big-endian: iterated
        self.assertEqual(self.t.decode(view.astype(">u4")), text)
        with self.assertRaises(TypeError):  # a 2-D array is a batch, not ids: refused as HF and tiktoken refuse it
            self.t.decode(numpy.stack([view, view]))
        with self.assertRaises(TypeError):
            self.t.decode(numpy.int32(ids[0]))                                # 0-D: not a sequence either

    def test_rejected_ids(self):
        for bad, kind in (([1.5], TypeError), (["a"], TypeError), ([2**40], OverflowError), ([None], TypeError),
                          ("string", TypeError), (5, TypeError), ([-1], toknroll.InvalidArgument),
                          ([10**9], toknroll.InvalidArgument), ([self.t.vocabulary_size], toknroll.InvalidArgument)):
            with self.subTest(bad=bad), self.assertRaises(kind) as e:
                self.t.decode(bad)
            if kind is toknroll.InvalidArgument:
                self.assertIsInstance(e.exception, ValueError)
                self.assertIn("no token with id", str(e.exception))   # a gap below the size is no token either
                self.assertIn("ids[0]", str(e.exception))

    def test_ids_read_in_place_cannot_move_during_the_call(self):
        """The call reads the array's memory with the GIL released: the owner it keeps is a buffer export, so a resize
        from another thread raises instead of moving the ids under the call."""
        ids = array("i", self.ids)
        owner, address, count = toknroll._ids(ids, "decode")
        self.assertEqual((address, count), (ids.buffer_info()[0], len(ids)))
        with self.assertRaises(BufferError):
            ids.append(1)
        del owner
        ids.append(1)  # released with the owner

    def test_a_million_ids(self):
        ids = self.t.encode("word " * 250_000)
        self.assertGreater(len(ids), 250_000)
        self.assertEqual(self.t.decode(ids), "word " * 250_000)
        self.assertEqual(len(self.t.decode_bytes(ids)), len("word " * 250_000))

    def test_a_result_outlives_the_next_decode(self):
        """Small decodes share their thread's buffer: what they return is a copy."""
        first, text = self.t.decode_bytes(self.ids), self.t.decode(self.ids)
        self.t.decode_bytes(self.t.encode("something else entirely"))
        self.assertEqual(first, self.text.encode())
        self.assertEqual(text, self.text)

    def test_long_tokens_overflow_the_shared_buffer(self):
        """Few ids of long tokens: past the thread's 64 KB buffer, the second pass gets its own."""
        long = next(i for i in range(self.t.vocabulary_size) if len(self.t.decode_bytes([i])) >= 16)
        one = self.t.decode_bytes([long])
        ids = array("i", [long]) * 8000  # a guess of 8 bytes an id stays under 64 KB; the bytes do not
        self.assertEqual(self.t.decode_bytes(ids), one * 8000)

    def test_decode_writes_whole_tokens_and_a_nul(self):
        """toknroll_decode follows snprintf: the size back, a NUL whenever byte_capacity is not 0, and
        over capacity the tokens that fit whole in byte_capacity - 1 bytes, not a byte-exact prefix."""
        lib, h, text = self.t._lib, self.t._handle, "Hello, Tok'n'Roll! Guitar"
        ids = self.t.encode(text)
        self.assertGreater(len(self.t.decode_bytes(ids[-1:])), 1)  # a byte short leaves a whole token out
        address, n = ids.buffer_info()[0], len(text)
        out = ctypes.create_string_buffer(n + 1)
        self.assertEqual(lib.toknroll_decode(h, address, len(ids), 0, out, n + 1, None), n)
        self.assertEqual(out.raw, text.encode() + b"\0")
        out, decoded_tokens = ctypes.create_string_buffer(n), ctypes.c_int32(-1)  # one short: the last left out
        self.assertEqual(lib.toknroll_decode(h, address, len(ids), 0, out, n, ctypes.byref(decoded_tokens)), n)
        self.assertEqual(out.value, self.t.decode_bytes(ids[:-1]))
        self.assertEqual(decoded_tokens.value, len(ids) - 1)
        out = ctypes.create_string_buffer(1)
        self.assertEqual(lib.toknroll_decode(h, address, len(ids), 0, out, 1, None), n)
        self.assertEqual(out.raw, b"\0")

    def test_byte_sizes_are_64_bit(self):
        """Every byte count is int64_t: a capacity or length of -2^40 is refused, where a binding cut
        to 32 bits would read it as 0. Over 4096 ids the size is counted in chunks first, and still
        exact."""
        lib, h, text = self.t._lib, self.t._handle, "word " * 5000
        ids = self.t.encode(text)
        self.assertGreater(len(ids), 4096)
        address = ids.buffer_info()[0]
        self.assertEqual(lib.toknroll_decode(h, address, len(ids), 0, None, 0, None), len(text.encode()))
        self.assertEqual(lib.toknroll_decode(h, address, len(ids), 0, None, -(1 << 40), None), N.EINVAL)
        self.assertEqual(lib.toknroll_token(h, ids[0], None, -(1 << 40)), N.EINVAL)
        self.assertEqual(lib.toknroll_encode(h, b"x", -(1 << 40), 0, None, 0), N.EINVAL)
        self.assertEqual(lib.toknroll_id(h, b"x", -(1 << 40)), N.EINVAL)


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


@unittest.skipUnless(hasattr(os, "fork"), "no fork on this platform")
class ForkTest(unittest.TestCase):
    """A child forked while other threads were inside the library inherited their locks and hung on its
    first call, until fork learned to wait for every call to leave. Each case runs in its own interpreter
    under a timeout, so a regression fails the case instead of hanging the suite."""

    def child_process(self, body):
        code = f"import os, sys, toknroll\nGGUF = {GGUF!r}\n" + textwrap.dedent(body)
        env = dict(os.environ, PYTHONPATH=str(Path(toknroll.__file__).resolve().parents[1]))
        try:
            run = subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            self.fail("the test process hung")
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_fork_while_threads_encode(self):
        self.child_process("""
            import signal, threading, time
            t = toknroll.load(GGUF)
            want, stop, done = list(t.encode("Hello, world!")), threading.Event(), []
            def busy():  # new words every time, so the runtime allocates and takes its locks
                i = 0
                while not stop.is_set():
                    t.encode_batch([f"busy {i} {i * 7919} text " * 50] * 4)
                    done.append(i)
                    i += 1
            for _ in range(4): threading.Thread(target=busy, daemon=True).start()
            while len(done) < 20: time.sleep(0.01)
            for _ in range(30):
                pid = os.fork()
                if pid == 0:
                    signal.alarm(10)  # a hung child dies rather than outlive the test
                    os._exit(0 if list(t.encode("Hello, world!")) == want and t.decode(want) == "Hello, world!" else 3)
                rc = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
                assert rc == 0, "the child hung on a lock its parent's thread held" if rc == -signal.SIGALRM else f"the child failed: {rc}"
                time.sleep(0.01)  # the threads run on between forks
            stop.set()
        """)

    def test_a_child_encodes_batches(self):
        """A batch runs on the runtime's pool, whose threads stay in the parent: a child forked while
        the parent's batches kept them busy must still finish its own."""
        self.child_process("""
            import signal, threading
            t = toknroll.load(GGUF)
            texts = [f"text {i} {i * 7919} " * 40 for i in range(2000)]  # big enough to split over every core
            want = [list(ids) for ids in t.encode_batch(texts)]
            stop = threading.Event()
            def busy():
                while not stop.is_set():
                    t.encode_batch(texts)
            for _ in range(2): threading.Thread(target=busy, daemon=True).start()
            for _ in range(20):
                pid = os.fork()
                if pid == 0:
                    signal.alarm(20)  # a hung child dies rather than outlive the test
                    os._exit(0 if [list(ids) for ids in t.encode_batch(texts)] == want else 3)
                rc = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
                assert rc == 0, "the child hung on a batch" if rc == -signal.SIGALRM else f"the child failed: {rc}"
            stop.set()
        """)

    def test_a_fork_pool_uses_the_tokenizer_made_before_it(self):
        self.child_process("""
            import multiprocessing
            t = toknroll.load(GGUF)
            want = list(t.encode("Hello, world!"))
            def work(text): return list(t.encode(text))
            with multiprocessing.get_context("fork").Pool(4) as pool:
                assert pool.map(work, ["Hello, world!"] * 64) == [want] * 64
        """)


class LengthTest(unittest.TestCase):
    """Text and spellings travel with their length in bytes: a slice of a buffer encodes as its own
    text, and a NUL is a character like any other."""

    def test_a_slice_encodes_as_its_own_text(self):
        lib = N.library()
        with toknroll.load(GGUF) as t:
            data = b"Hello, world! and the rest of the buffer"
            self.assertEqual(lib.toknroll_encode(t._handle, data, 13, 0, None, 0), len(t.encode("Hello, world!")))
            self.assertEqual(lib.toknroll_encode(t._handle, None, 0, 0, None, 0), 0)   # no text, no pointer

    def test_an_empty_batch_still_checks_its_flags(self):
        """Count 0 takes NULL arrays and returns 0, but an unknown flag is EINVAL there as everywhere."""
        lib = N.library()
        with toknroll.load(GGUF) as t:
            self.assertEqual(lib.toknroll_encode_batch(t._handle, None, None, 0, 0, None, 0, None), 0)
            self.assertEqual(lib.toknroll_encode_batch(t._handle, None, None, 0, 2, None, 0, None), N.EINVAL)
            code = ctypes.c_int32()
            self.assertIn(b"unknown flags", lib.toknroll_last_error(ctypes.byref(code)))
            self.assertEqual(code.value, N.EINVAL)

    def test_a_nul_is_a_character(self):
        with toknroll.load(GGUF) as t:
            self.assertEqual(t.decode(t.encode("a\0b")), "a\0b")
            self.assertEqual(t.count_tokens("a\0b"), len(t.encode("a\0b")))
            self.assertIsNone(t.id("<|im_start|>\0"))   # looked up whole, not cut at the NUL
            self.assertIsNotNone(t.id("<|im_start|>"))

    def test_a_nul_ends_no_c_string(self):
        """A source or an option is a C string: a NUL would load or configure the prefix silently,
        so it is refused as Python's `open` refuses it."""
        with self.assertRaisesRegex(ValueError, "embedded null byte"):
            toknroll.load(GGUF + "\0/junk")
        with self.assertRaisesRegex(ValueError, "embedded null byte"):
            toknroll.configure("-Dtoknroll.offline=true\0-Dtoknroll.cache.root=/x")

    @unittest.skipIf(os.name == "nt", "Windows strips a trailing space from a name, and a symlink needs a privilege")
    def test_a_path_is_taken_as_given(self):
        """No stripping: a name may end in a space, and a padded path is the missing file it names,
        as HF's from_file and Python's open take it."""
        with tempfile.TemporaryDirectory() as d:
            padded = os.path.join(d, "model ")
            os.symlink(GGUF, padded)
            with toknroll.load(padded) as t:
                self.assertEqual(t.source, padded)
        with self.assertRaises(toknroll.NotFound):
            toknroll.load(GGUF + " ")
        with self.assertRaises(toknroll.NotFound):
            toknroll.load(GGUF + "\n")


class UnloadTest(unittest.TestCase):
    def test_a_thread_exits_after_the_host_unloaded_the_library(self):
        """Its exit hook runs the library's code, as the runtime's own threads do: the library pins itself
        once started, so an unload succeeds and leaves it in place. Unpinned, the exit crashed."""
        code = textwrap.dedent(f"""
            import _ctypes, ctypes, sys, threading
            lib = ctypes.CDLL({str(N.locate("toknroll"))!r})
            lib.toknroll_tokenizer_load.restype = ctypes.c_void_p
            lib.toknroll_vocabulary_size.argtypes = [ctypes.c_void_p]
            called, unloaded = threading.Event(), threading.Event()
            def worker():
                assert lib.toknroll_vocabulary_size(lib.toknroll_tokenizer_load({GGUF!r}.encode())) > 0
                called.set()
                unloaded.wait()
            thread = threading.Thread(target=worker)
            thread.start()
            called.wait()
            (_ctypes.FreeLibrary if sys.platform == "win32" else _ctypes.dlclose)(lib._handle)
            unloaded.set()
            thread.join()
        """)
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)


class ErrorTypesTest(unittest.TestCase):
    def test_hierarchy(self):
        self.assertTrue(issubclass(toknroll.InvalidArgument, toknroll.ToknrollError))
        self.assertTrue(issubclass(toknroll.InvalidArgument, ValueError))
        self.assertTrue(issubclass(toknroll.NotFound, toknroll.ToknrollError))
        self.assertTrue(issubclass(toknroll.NotFound, FileNotFoundError))
        self.assertTrue(issubclass(toknroll.Unsupported, toknroll.ToknrollError))
        self.assertTrue(issubclass(toknroll.Unsupported, ValueError))
        self.assertEqual(toknroll.InvalidArgument("m", -2).code, -2)
        self.assertEqual(str(toknroll.NotFound("gone", -3)), "gone")

    def test_errors_pickle(self):
        """An error pickles with its code, so one raised in a worker reaches the parent, not a broken pool."""
        for error in (toknroll.ToknrollError("m", -1), toknroll.InvalidArgument("m", -2), toknroll.NotFound("m", -3),
                      toknroll.Unsupported("m", -4)):
            back = pickle.loads(pickle.dumps(error))
            self.assertEqual((type(back), str(back), back.code), (type(error), "m", error.code))
        with multiprocessing.get_context("spawn").Pool(1) as pool:  # spawn: the one start method every platform has
            with self.assertRaises(toknroll.NotFound):
                pool.apply(toknroll.load, ("/no/such/file",))

    def test_a_success_clears_the_last_error(self):
        lib, code = N.library(), ctypes.c_int32()
        with toknroll.load(GGUF) as t:
            self.assertIsNone(lib.toknroll_tokenizer_load(b"/no/such/file"))
            self.assertIn(b"such", lib.toknroll_last_error(ctypes.byref(code)))   # the path, spelled as the platform spells it
            self.assertEqual(code.value, N.ENOTFOUND)
            self.assertGreater(lib.toknroll_vocabulary_size(t._handle), 0)
            self.assertEqual(lib.toknroll_last_error(ctypes.byref(code)), b"")
            self.assertEqual(code.value, N.OK)

    def test_codes_match_the_header(self):
        header = (Path(__file__).resolve().parents[2] / "include" / "toknroll.h").read_text()
        for name, value in (("OK", N.OK), ("ERROR", N.ERROR), ("EINVAL", N.EINVAL), ("ENOTFOUND", N.ENOTFOUND)):
            line = next(l for l in header.splitlines() if l.startswith(f"#define TOKNROLL_{name} "))
            self.assertEqual(int(line.split()[2].strip("()")), value, name)

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
        self.assertEqual(len(declared), 13)


class ConfigurationTest(unittest.TestCase):
    def test_offline_set_to_anything_but_false_never_fetches(self):
        """A misspelled true (1, yes) must never mean online: the knob is offline unless it says false."""
        code = textwrap.dedent("""
            import toknroll
            try:
                toknroll.load("no-such-owner-qa/no-such-repo")
            except toknroll.NotFound as e:
                assert "never fetches" in str(e), e
            else:
                raise SystemExit("loaded")
            """)
        for value in ("true", "TRUE", "1", "yes", "bogus", ""):
            with self.subTest(value=value):
                env = dict(os.environ, PYTHONPATH=str(Path(toknroll.__file__).resolve().parents[1]),
                           TOKNROLL_OPTIONS=f"-Dtoknroll.offline={value}")
                run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stderr)


class LibraryLocationTest(unittest.TestCase):
    def test_platform_names(self):
        located = N.locate("toknroll")
        self.assertIn(located.name, ("libtoknroll.so", "libtoknroll.dylib", "toknroll.dll"))

    def test_env_override_points_at_a_missing_file(self):
        """An override is the one candidate: a typo in it is an error naming it, never another build in silence."""
        with unittest.mock.patch.dict(os.environ, {"TOKNROLL_LIBRARY": "/no/such/libtoknroll.so"}):
            with self.assertRaises(OSError) as e:
                N.locate("toknroll")
        self.assertIn(str(Path("/no/such/libtoknroll.so")), str(e.exception))  # as the platform spells it
        self.assertNotIn("target", str(e.exception))

    def test_another_shared_library_is_not_a_toknroll(self):
        libc = ctypes.util.find_library("c")
        if not libc:
            self.skipTest("no C library found by name")
        with self.assertRaises(OSError) as e:
            N.bind(ctypes.CDLL(libc))
        self.assertIn("not a toknroll library", str(e.exception))


if __name__ == "__main__":
    unittest.main()
