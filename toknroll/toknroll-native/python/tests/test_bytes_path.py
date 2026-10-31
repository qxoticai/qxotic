"""Text that reaches the tokenizer as its own bytes: batches cut into runs, short buffers, ill-formed UTF-8."""
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
    n = t._lib.toknroll_encode(t._handle, data, len(data), specials, out.buffer_info()[0], capacity)
    return n, out[: min(n, capacity)]


def raw_batch(t, texts, capacity, specials=False):
    """toknroll_encode_batch over byte strings, as given: (total, written prefix, ends)."""
    joined = b"".join(texts)
    text_ends = array("i", accumulate(map(len, texts))) or array("i", [0])
    ends = array("i", bytes(4 * max(len(texts), 1)))
    out = array("i", bytes(4 * max(capacity, 1)))
    n = t._lib.toknroll_encode_batch(t._handle, joined, text_ends.buffer_info()[0], len(texts), specials,
                                     out.buffer_info()[0], capacity, ends.buffer_info()[0])
    return n, out[: min(n, capacity)], ends[: len(texts)]


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

    def test_ids_decode_back_to_the_text(self):
        texts = corpus(2, 200)
        for t in self.tokenizers:
            self.assertEqual(t.decode_batch(t.encode_batch(texts)), texts)

    def test_a_short_buffer_gets_a_prefix_and_the_whole_count(self):
        texts = [x.encode() for x in corpus(3, 60)]
        for t in self.tokenizers:
            full, ids, ends = raw_batch(t, texts, 10_000_000)
            self.assertEqual(len(ids), full)
            for capacity in (0, 1, full // 3, full - 1):
                n, prefix, short_ends = raw_batch(t, texts, capacity)
                self.assertEqual(n, full)
                self.assertEqual(prefix, ids[:capacity])
                self.assertEqual(short_ends, ends)
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
            full, ids, ends = raw_batch(t, texts, 10_000_000, True)
            for capacity in (0, 1, full // 3, full - 1):
                self.assertEqual(raw_batch(t, texts, capacity, True), (full, ids[:capacity], ends))
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
            n, ids, ends = raw_batch(t, texts, 10_000_000, True)
            self.assertEqual([ids[s:e] for s, e in zip([0, *ends[:-1]], ends)], expected, t.source)

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
            n, ids, ends = raw_batch(t, texts, 10_000_000)
            got = [ids[s:e] for s, e in zip([0, *ends[:-1]], ends)]
            self.assertEqual(got, expected, t.source)


if __name__ == "__main__":
    unittest.main()
