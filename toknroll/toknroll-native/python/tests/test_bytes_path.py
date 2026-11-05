"""Text that reaches the tokenizer as its own bytes: batches cut into runs, short buffers, ill-formed UTF-8."""
import ctypes as C
import random
import unittest
from array import array
from itertools import accumulate

from support import EDGE_TEXTS, GGUF, TIKTOKEN_JSON, toknroll

SOURCES = [s for s in (GGUF, TIKTOKEN_JSON) if s]

# Ill-formed bytes and the text the JDK decodes them to, as every text was decoded before the bytes path:
# an ill-formed sequence is one U+FFFD each, a surrogate written in UTF-8 one for all three bytes.
ILL_FORMED = [
    (b"\x80", "\ufffd"), (b"abc\xffdef", "abc\ufffddef"), (b"\xc0\xaf", "\ufffd\ufffd"), (b"\xc1\xbf", "\ufffd\ufffd"),
    (b"\xe0\x80\xaf", "\ufffd\ufffd\ufffd"), (b"\xed\xa0\x80", "\ufffd"), (b"\xf4\x90\x80\x80", "\ufffd" * 4),
    (b"\xf5\x80\x80\x80", "\ufffd" * 4), (b"caf\xc3", "caf\ufffd"), (b"\xe2\x82", "\ufffd"),
    (b"ok \xe2\x82\xac ok \xe2\x82", "ok \u20ac ok \ufffd"), (b"\xf0\x9f\x8e\xb8\xf0\x9f\x8e", "\U0001f3b8\ufffd"),
    (b"plain ascii, then \xfe\xff at the end", "plain ascii, then \ufffd\ufffd at the end"), (b"\xef\xbb\xbf\xff", "\ufeff\ufffd"),
]

def corpus(seed, count):
    """Texts of every size the runs meet, empty ones among them, in several scripts."""
    rng = random.Random(seed)
    pieces = ["the", "quick", "brown", "fox", "Ünïcödé", "日本語", "テキスト", "😀", "🎸🎹", "1234567", "3.14159",
              "don't", "WE'LL", " ", "  ", "\n", "\r\n", "\t", "<|im_end|>", "x" * 40, "--", "😀" * 9]
    texts = []
    for _ in range(count):
        size = rng.choice([0, 1, 5, 50, 500, 5_000, 50_000])
        words = []
        while sum(map(len, words)) < size:
            words.append(rng.choice(pieces))
        texts.append(" ".join(words) if rng.random() < 0.7 else "".join(words))
    return texts


def raw_encode(t, data, capacity, specials=False):
    out = array("i", bytes(4 * max(capacity, 1)))
    n = t._lib.toknroll_encode(t._handle, data, len(data), int(specials), out.buffer_info()[0], capacity)
    return n, out[: min(n, capacity)]


def raw_batch(t, texts, capacity, specials=False, before=b""):
    """toknroll_encode_batch over byte strings, as given, after `before` in their buffer:
    (total, written prefix, offsets)."""
    joined = before + b"".join(texts)
    text_offsets = array("q", [len(before), *(len(before) + n for n in accumulate(map(len, texts)))])
    offsets = array("i", bytes(4 * (len(texts) + 1)))
    out = array("i", bytes(4 * max(capacity, 1)))
    n = t._lib.toknroll_encode_batch(t._handle, joined, text_offsets.buffer_info()[0], len(texts), int(specials),
                                     out.buffer_info()[0], capacity, offsets.buffer_info()[0])
    return n, out[: min(n, capacity)], offsets


def split(ids, offsets):
    return [ids[s:e] for s, e in zip(offsets, offsets[1:])]


@unittest.skipIf(not SOURCES, "no local tokenizer: set TOKNROLL_TEST_GGUF or TOKNROLL_TEST_TIKTOKEN_JSON")
class BytesPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizers = [toknroll.load(s) for s in SOURCES]

    @classmethod
    def tearDownClass(cls):
        for t in cls.tokenizers:
            t.close()

    def test_a_batch_cut_into_runs_equals_one_by_one(self):
        texts = EDGE_TEXTS + corpus(1, 400) + ["", "", ""] + ["long " * 60_000]
        for t in self.tokenizers:
            self.assertEqual(t.encode_batch(texts), [t.encode(x) for x in texts], t.source)

    def test_fewer_texts_than_runs(self):
        for t in self.tokenizers:
            for texts in (["one"], ["", "two"], ["a", "b", "c"]):
                self.assertEqual(t.encode_batch(texts), [t.encode(x) for x in texts])

    def test_one_text_is_not_a_batch(self):
        """A str iterates as its characters, so a text given as a batch is refused; any other iterable of texts goes."""
        t = self.tokenizers[0]
        with self.assertRaises(TypeError):
            t.encode_batch("abc")
        self.assertEqual(t.encode_batch(iter(["ab", "c"])), [t.encode("ab"), t.encode("c")])

    def test_ids_decode_back_to_the_text(self):
        texts = corpus(2, 200)
        for t in self.tokenizers:
            self.assertEqual([t.decode(ids) for ids in t.encode_batch(texts)], texts)

    def test_a_short_buffer_gets_a_prefix_and_the_whole_count(self):
        texts = [x.encode() for x in corpus(3, 60)]
        for t in self.tokenizers:
            full, ids, offsets = raw_batch(t, texts, 10_000_000)
            self.assertEqual(len(ids), full)
            for capacity in (0, 1, full // 3, full - 1):
                n, prefix, short_offsets = raw_batch(t, texts, capacity)
                self.assertEqual(n, full)
                self.assertEqual(prefix, ids[:capacity])
                self.assertEqual(short_offsets, offsets)
            for x in texts[:5]:
                n, whole = raw_encode(t, x, 1_000_000)
                self.assertEqual(raw_encode(t, x, 0), (n, array("i")))
                self.assertEqual(raw_encode(t, x, n // 2), (n, whole[: n // 2]))

    def test_specials_split_the_text_and_the_rest_encodes_as_plain_text(self):
        special = "<|im_end|>"
        for t in self.tokenizers:
            if not t.token_flags(t.id(special)) & ~toknroll.TokenFlags.NORMAL:
                continue
            for x in corpus(5, 100) + [special, special * 3, "a" + special, special + "日本語😀"]:
                expected = array("i")
                for k, part in enumerate(x.split(special)):
                    expected += array("i", [t.id(special)] if k else []) + t.encode(part)
                self.assertEqual(t.encode(x, specials=True), expected, (t.source, x))

    def test_a_short_buffer_gets_a_prefix_and_the_whole_count_with_specials(self):
        texts = [x.encode() for x in corpus(6, 60)]
        for t in self.tokenizers:
            full, ids, offsets = raw_batch(t, texts, 10_000_000, True)
            for capacity in (0, 1, full // 3, full - 1):
                self.assertEqual(raw_batch(t, texts, capacity, True), (full, ids[:capacity], offsets))
            for x in texts[:5]:
                n, whole = raw_encode(t, x, 1_000_000, True)
                self.assertEqual(raw_encode(t, x, 0, True), (n, array("i")))
                self.assertEqual(raw_encode(t, x, n // 2, True), (n, whole[: n // 2]))

    def test_ill_formed_bytes_around_specials_encode_as_their_replacement(self):
        for t in self.tokenizers:
            for data, text in ILL_FORMED:
                data, text = data + b"<|im_end|>" + data, text + "<|im_end|>" + text
                expected = t.encode(text, specials=True)
                self.assertEqual(raw_encode(t, data, 1000, True), (len(expected), expected), (t.source, data))
            texts = [bad + b"<|im_end|>" for bad, _ in ILL_FORMED]
            expected = [t.encode(text + "<|im_end|>", specials=True) for _, text in ILL_FORMED]
            n, ids, offsets = raw_batch(t, texts, 10_000_000, True)
            self.assertEqual(split(ids, offsets), expected, t.source)

    def test_texts_may_start_anywhere_in_their_buffer(self):
        texts = [x.encode() for x in corpus(7, 40)]
        for t in self.tokenizers:
            expected = raw_batch(t, texts, 10_000_000)
            for before in (b"x", "日本".encode() * 33, b"<|im_end|>"):
                self.assertEqual(raw_batch(t, texts, 10_000_000, before=before), expected)
            n, ids, offsets = expected
            self.assertEqual(offsets[0], 0)
            self.assertEqual(offsets[-1], n)
            self.assertEqual(split(ids, offsets), [t.encode(x.decode()) for x in texts])
            self.assertEqual(raw_batch(t, [], 0), (0, array("i"), array("i", [0])))

    def test_an_unknown_flag_is_refused(self):
        text_offsets, offsets = array("q", [0, 1]), array("i", [0, 0])
        for t in self.tokenizers:
            for flags in (2, 3, 1 << 31):
                self.assertLess(t._lib.toknroll_encode(t._handle, b"x", 1, flags, None, 0), 0)
                self.assertIn("unknown flags", str(toknroll._error(t._lib)))
                n = t._lib.toknroll_encode_batch(t._handle, b"x", text_offsets.buffer_info()[0], 1, flags,
                                                 None, 0, offsets.buffer_info()[0])
                self.assertLess(n, 0)
                self.assertIn("unknown flags", str(toknroll._error(t._lib)))

    def test_text_offsets_past_two_to_the_31(self):
        # texts points 2^32 bytes before the buffer and the offsets start at 2^32: only 64-bit
        # offsets reach the bytes, and none outside them is read
        data, shift = "hello world, 日本".encode(), 1 << 32
        buffer = C.create_string_buffer(data, len(data))
        if C.addressof(buffer) < shift:
            self.skipTest("the buffer sits below 2^32")
        text_offsets = array("q", [shift, shift + 5, shift + len(data)])
        for t in self.tokenizers:
            offsets, out = array("i", bytes(4 * 3)), array("i", bytes(4 * 100))
            n = t._lib.toknroll_encode_batch(t._handle, C.c_char_p(C.addressof(buffer) - shift),
                                             text_offsets.buffer_info()[0], 2, 0, out.buffer_info()[0], 100,
                                             offsets.buffer_info()[0])
            self.assertEqual(split(out[:n], offsets), [t.encode("hello"), t.encode(" world, 日本")])

    def test_ill_formed_bytes_encode_as_their_replacement(self):
        for t in self.tokenizers:
            for data, text in ILL_FORMED:
                expected = t.encode(text)
                self.assertEqual(raw_encode(t, data, 1000), (len(expected), expected), (t.source, data))

    def test_ill_formed_texts_in_a_batch(self):
        texts, decoded = [], []
        for good, (bad, text) in zip(corpus(4, len(ILL_FORMED)), ILL_FORMED):
            texts += [good.encode(), bad]
            decoded += [good, text]
        for t in self.tokenizers:
            expected = [t.encode(x) for x in decoded]
            n, ids, offsets = raw_batch(t, texts, 10_000_000)
            got = split(ids, offsets)
            self.assertEqual(got, expected, t.source)


if __name__ == "__main__":
    unittest.main()
