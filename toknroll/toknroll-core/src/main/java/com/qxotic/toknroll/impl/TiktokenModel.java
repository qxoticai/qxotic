package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.UnsupportedTokenizerException;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.spi.TiktokenEngine;
import com.qxotic.toknroll.spi.TiktokenTables;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * Byte-level BPE as tiktoken defines it, the reference encoder of the family.
 *
 * <p>Behind one of the scanner's regexes a {@link TiktokenEngine} encodes instead, when one is
 * found and agrees with this encoder on {@link TiktokenSplitter#sample}. Without one, a word is
 * merged here: its bytes' tokens, then the pair of lowest rank, the leftmost of equal ranks, until
 * no pair merges. A short word's ids are remembered across calls.
 */
final class TiktokenModel extends AbstractTokenizationModel {

    private static final int NO_TOKEN = -1;

    /** Words of up to this many bytes are remembered, up to {@link #MEMO_WORDS} of them. */
    private static final int MEMO_BYTES = 32;

    private static final int MEMO_WORDS = 1 << 16;

    private static final List<TiktokenEngine> ENGINES = engines();

    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    private final LongLongMap merges;
    private final int[] singleByteTokenId;
    private final byte[][] tokenBytesById;
    // a token of up to WIDE bytes as two words, its bytes zero-padded, for decode to store whole
    private static final int WIDE = 16;
    private final long[] spellingWords;
    // its length, or 0 for a longer token, which decode copies
    private final byte[] shortLengths;
    private final boolean ignoreMerges;
    private final Map<String, int[]> memo = new ConcurrentHashMap<>();

    /** An entry of {@link #encoders}: no engine encodes the rule set. */
    private static final Object NO_ENCODER = new Object();

    /** The engine's encoder per rule set, or {@link #NO_ENCODER}; null until first used. */
    private final AtomicReferenceArray<Object> encoders =
            new AtomicReferenceArray<>(TiktokenSplitter.RULE_SETS);

    private TiktokenModel(
            Vocabulary vocabulary,
            LongLongMap merges,
            int[] singleByteTokenId,
            byte[][] tokenBytesById,
            boolean ignoreMerges,
            float expectedTokensPerChar) {
        super(vocabulary, expectedTokensPerChar);
        this.merges = merges;
        this.singleByteTokenId = singleByteTokenId;
        this.tokenBytesById = tokenBytesById;
        this.spellingWords = new long[2 * tokenBytesById.length];
        this.shortLengths = new byte[tokenBytesById.length];
        for (int id = 0; id < tokenBytesById.length; id++) {
            byte[] spelling = tokenBytesById[id];
            if (spelling != null && spelling.length <= WIDE) {
                byte[] padded = Arrays.copyOf(spelling, WIDE);
                spellingWords[2 * id] = (long) LONGS.get(padded, 0);
                spellingWords[2 * id + 1] = (long) LONGS.get(padded, 8);
                shortLengths[id] = (byte) spelling.length;
            }
        }
        this.ignoreMerges = ignoreMerges;
    }

    static TiktokenModel fromVocabularyAndMerges(Vocabulary vocabulary, LongLongMap merges) {
        return fromVocabularyAndMerges(vocabulary, merges, false);
    }

    static TiktokenModel fromVocabularyAndMerges(
            Vocabulary vocabulary, LongLongMap merges, boolean ignoreMerges) {
        Objects.requireNonNull(vocabulary, "vocabulary");
        Objects.requireNonNull(merges, "merges");
        // Every token must be ByteLevel-encoded; loaders pre-encode special and control tokens.
        validateByteLevelVocabulary(vocabulary);
        return new TiktokenModel(
                vocabulary,
                merges,
                buildSingleByteTokenMap(vocabulary),
                buildTokenBytesById(vocabulary),
                ignoreMerges,
                estimateTokensPerChar(vocabulary.size()));
    }

    private static float estimateTokensPerChar(int vocabSize) {
        // Conservative heuristic for GPT-family corpora token density.
        if (vocabSize <= 60000) {
            return 0.34f;
        } else if (vocabSize <= 110000) {
            return 0.31f;
        } else if (vocabSize <= 150000) {
            return 0.30f;
        } else {
            return 0.28f;
        }
    }

    // ---- the model alone: the whole text is one word ----

    @Override
    protected IntSequence encodeImpl(CharSequence text) {
        IntSequence.Builder out =
                IntSequence.newBuilder(estimateInitialTokenCapacity(text.length()));
        encodeImplInto(text, out);
        return out.build();
    }

    @Override
    protected void encodeImplInto(CharSequence text, IntSequence.Builder out) {
        encodeInto(text, 0, text.length(), out);
    }

    // ---- behind a scanner ----

    /** Encodes {@code text[start, end)} as {@code splitter} cuts it. */
    void encode(
            CharSequence text,
            int start,
            int end,
            TiktokenSplitter splitter,
            IntSequence.Builder out) {
        TiktokenEngine.Encoder fast = encoder(splitter);
        if (fast != null && !splitter.coreCuts()) {
            fast.encode(text, start, end, out);
            return;
        }
        byte[] utf8 = TiktokenSplitter.utf8(text, start, end);
        if (fast != null) {
            encodeCut(fast, utf8, 0, utf8.length, splitter, out);
            return;
        }
        encodeWords(utf8, 0, utf8.length, splitter, out);
    }

    /** {@link #encode(CharSequence, int, int, TiktokenSplitter, IntSequence.Builder)} for UTF-8. */
    void encodeUtf8(
            byte[] utf8, int start, int end, TiktokenSplitter splitter, IntSequence.Builder out) {
        TiktokenEngine.Encoder fast = encoder(splitter);
        if (fast != null && !splitter.coreCuts()) {
            fast.encodeUtf8(utf8, start, end, out);
            return;
        }
        if (fast != null) {
            encodeCut(fast, utf8, start, end, splitter, out);
            return;
        }
        encodeWords(utf8, start, end, splitter, out);
    }

    /** The scanner's pieces, merged by an engine's word encoder when there is one. */
    private void encodeWords(
            byte[] text, int start, int end, TiktokenSplitter splitter, IntSequence.Builder out) {
        TiktokenEngine.WordEncoder words = words(splitter);
        if (words == null) {
            encode(text, start, end, splitter, out);
            return;
        }
        encodeWords(words, text, start, end, splitter, out);
    }

    private static void encodeWords(
            TiktokenEngine.WordEncoder words,
            byte[] text,
            int start,
            int end,
            TiktokenSplitter splitter,
            IntSequence.Builder out) {
        int[] ends = new int[Math.min(TiktokenSplitter.BATCH, Math.max(1, end - start))];
        for (int cursor = start; cursor < end; ) {
            int n = splitter.split(text, cursor, end, ends);
            words.encodeWords(text, cursor, ends, n, out);
            cursor = ends[n - 1];
        }
    }

    /** The engine on each piece the splitter's first stage cuts, as if each were the text. */
    private static void encodeCut(
            TiktokenEngine.Encoder fast,
            byte[] utf8,
            int start,
            int end,
            TiktokenSplitter splitter,
            IntSequence.Builder out) {
        for (int i = start; i < end; ) {
            int stop = splitter.cut(utf8, i, end);
            fast.encodeUtf8(utf8, i, stop, out);
            i = stop;
        }
    }

    /** The reference: the scanner's pieces, word by word. */
    private void encode(
            byte[] text, int start, int end, TiktokenSplitter splitter, IntSequence.Builder out) {
        int[] ends = new int[Math.min(TiktokenSplitter.BATCH, Math.max(1, end - start))];
        int cursor = start;
        while (cursor < end) {
            int n = splitter.split(text, cursor, end, ends);
            for (int k = 0; k < n; k++) {
                encodeWord(text, cursor, ends[k], out);
                cursor = ends[k];
            }
        }
    }

    private void encodeWord(byte[] text, int start, int end, IntSequence.Builder out) {
        int length = end - start;
        if (length <= 0) {
            return;
        }
        if (length == 1) {
            out.add(singleByteTokenId[text[start] & 0xFF]);
            return;
        }
        String key =
                length <= MEMO_BYTES
                        ? new String(text, start, length, StandardCharsets.ISO_8859_1)
                        : null;
        int[] ids = key != null ? memo.get(key) : null;
        if (ids == null) {
            ids = wordIds(text, start, end);
            if (key != null) {
                if (memo.size() >= MEMO_WORDS) {
                    memo.clear();
                }
                memo.put(key, ids);
            }
        }
        for (int id : ids) {
            out.add(id);
        }
    }

    private int[] wordIds(byte[] text, int start, int end) {
        if (ignoreMerges) {
            // a word that is a token is that token, as the vocabulary spells it
            String spelling = ByteLevel.encode(Arrays.copyOfRange(text, start, end));
            if (vocabulary().contains(spelling)) {
                return new int[] {vocabulary().id(spelling)};
            }
        }
        return merge(text, start, end);
    }

    /**
     * The word's tokens as a list linked through {@code next}, and every pair that merges as a
     * candidate in a min-heap: the rank in the high half, the left position in the low, so the
     * smallest is the lowest rank and the leftmost of equal ones. A merge leaves a candidate behind
     * that no longer holds, which is skipped when it comes up.
     */
    private int[] merge(byte[] text, int start, int end) {
        int n = end - start;
        int[] token = new int[n];
        int[] prev = new int[n];
        int[] next = new int[n];
        for (int i = 0; i < n; i++) {
            token[i] = singleByteTokenId[text[start + i] & 0xFF];
            prev[i] = i - 1;
            next[i] = i + 1;
        }
        next[n - 1] = -1;
        long[] heap = new long[3 * n]; // n - 1 pairs, then at most two per merge
        int size = 0;
        for (int i = 0; i + 1 < n; i++) {
            size = push(heap, size, candidate(token, i, i + 1));
        }
        int count = n;
        while (size > 0) {
            long top = heap[0];
            size = pop(heap, size);
            int i = (int) top;
            int j = token[i] < 0 ? -1 : next[i];
            if (j < 0) {
                continue;
            }
            long merged = merges.getPair(token[i], token[j]);
            if (merged == IntPair.NONE || IntPair.right(merged) != (int) (top >>> 32)) {
                continue;
            }
            token[i] = IntPair.left(merged);
            token[j] = -1;
            next[i] = next[j];
            if (next[i] >= 0) {
                prev[next[i]] = i;
            }
            count--;
            if (prev[i] >= 0) {
                size = push(heap, size, candidate(token, prev[i], i));
            }
            if (next[i] >= 0) {
                size = push(heap, size, candidate(token, i, next[i]));
            }
        }
        int[] ids = new int[count];
        for (int i = 0, k = 0; i >= 0; i = next[i]) {
            ids[k++] = token[i];
        }
        return ids;
    }

    /** The heap entry for the pair at {@code left}, or -1 when it does not merge. */
    private long candidate(int[] token, int left, int right) {
        long merged = merges.getPair(token[left], token[right]);
        return merged == IntPair.NONE ? -1 : ((long) IntPair.right(merged) << 32) | left;
    }

    private static int push(long[] heap, int size, long entry) {
        if (entry < 0) {
            return size;
        }
        int i = size;
        while (i > 0 && heap[(i - 1) >>> 1] > entry) {
            heap[i] = heap[(i - 1) >>> 1];
            i = (i - 1) >>> 1;
        }
        heap[i] = entry;
        return size + 1;
    }

    private static int pop(long[] heap, int size) {
        long last = heap[--size];
        int i = 0;
        while (2 * i + 1 < size) {
            int child = 2 * i + 1;
            if (child + 1 < size && heap[child + 1] < heap[child]) {
                child++;
            }
            if (heap[child] >= last) {
                break;
            }
            heap[i] = heap[child];
            i = child;
        }
        heap[i] = last;
        return size;
    }

    // ---- the engine ----

    private static List<TiktokenEngine> engines() {
        List<TiktokenEngine> engines = new ArrayList<>();
        try {
            for (TiktokenEngine engine : ServiceLoader.load(TiktokenEngine.class)) {
                engines.add(engine);
            }
        } catch (Throwable e) {
            // a provider that cannot even be loaded is no engine
        }
        return engines;
    }

    private TiktokenEngine.Encoder encoder(TiktokenSplitter splitter) {
        int rules = splitter.rules();
        if (rules < 0) {
            return null;
        }
        Object encoder = encoders.get(rules);
        if (encoder == null) {
            // checked against the scanner the engine's rules stand for, without the cuts core makes
            encoder = findEncoder(rules, splitter.coreCuts() ? splitter.base() : splitter);
            Object first = encoders.compareAndExchange(rules, null, encoder);
            encoder = first != null ? first : encoder;
        }
        return encoder != NO_ENCODER ? (TiktokenEngine.Encoder) encoder : null;
    }

    /** The engine's word encoder, or {@link #NO_ENCODER}; null until first used. */
    private final AtomicReference<Object> words = new AtomicReference<>();

    private TiktokenEngine.WordEncoder words(TiktokenSplitter splitter) {
        Object words = this.words.get();
        if (words == null) {
            words = findWords(splitter);
            Object first = this.words.compareAndExchange(null, words);
            words = first != null ? first : words;
        }
        return words != NO_ENCODER ? (TiktokenEngine.WordEncoder) words : null;
    }

    /** The first engine's word encoder that merges the sample's words as this model does. */
    private Object findWords(TiktokenSplitter splitter) {
        if ("none".equals(System.getProperty("toknroll.engine"))) {
            return NO_ENCODER;
        }
        String sample = TiktokenSplitter.sample();
        byte[] utf8 = TiktokenSplitter.utf8(sample, 0, sample.length());
        IntSequenceBuilder expected = new IntSequenceBuilder();
        encode(utf8, 0, utf8.length, splitter, expected);
        int[] want = expected.build().toArray();
        for (TiktokenEngine engine : ENGINES) {
            try {
                TiktokenEngine.WordEncoder candidate = engine.words(new Tables());
                if (candidate == null) {
                    continue;
                }
                IntSequence.Builder got = IntSequence.newBuilder(want.length);
                encodeWords(candidate, utf8, 0, utf8.length, splitter, got);
                if (Arrays.equals(want, got.build().toArray())) {
                    return candidate;
                }
            } catch (Throwable e) {
                // passed over like an encoder that disagrees
            }
        }
        return NO_ENCODER;
    }

    /** The first engine's encoder that agrees with this model, or {@link #NO_ENCODER}. */
    private Object findEncoder(int rules, TiktokenSplitter splitter) {
        if ("none".equals(System.getProperty("toknroll.engine"))) {
            return NO_ENCODER;
        }
        for (TiktokenEngine engine : ENGINES) {
            try {
                TiktokenEngine.Encoder candidate = engine.encoder(new Tables(), rules);
                if (candidate != null && agrees(candidate, splitter)) {
                    return candidate;
                }
            } catch (Throwable e) {
                // passed over like one that disagrees: say, the Vector API module is not enabled
            }
        }
        return NO_ENCODER;
    }

    /** Whether {@code candidate} encodes the sample as this model does, from chars and bytes. */
    private boolean agrees(TiktokenEngine.Encoder candidate, TiktokenSplitter splitter) {
        String sample = TiktokenSplitter.sample();
        byte[] utf8 = TiktokenSplitter.utf8(sample, 0, sample.length());
        IntSequenceBuilder expected = new IntSequenceBuilder();
        encode(utf8, 0, utf8.length, splitter, expected);
        IntSequence.Builder fromChars = IntSequence.newBuilder(expected.size());
        candidate.encode(sample, 0, sample.length(), fromChars);
        IntSequence.Builder fromBytes = IntSequence.newBuilder(expected.size());
        candidate.encodeUtf8(utf8, 0, utf8.length, fromBytes);
        int[] want = expected.build().toArray();
        return Arrays.equals(want, fromChars.build().toArray())
                && Arrays.equals(want, fromBytes.build().toArray());
    }

    /** The model as an engine sees it. */
    private final class Tables implements TiktokenTables {
        private final int[] left;
        private final int[] right;
        private final int[] result;
        private final int[] rank;

        Tables() {
            List<long[]> entries = new ArrayList<>();
            merges.forEach((key, value) -> entries.add(new long[] {key, value}));
            int count = entries.size();
            left = new int[count];
            right = new int[count];
            result = new int[count];
            rank = new int[count];
            for (int i = 0; i < count; i++) {
                long[] entry = entries.get(i);
                left[i] = IntPair.left(entry[0]);
                right[i] = IntPair.right(entry[0]);
                result[i] = IntPair.left(entry[1]);
                rank[i] = IntPair.right(entry[1]);
            }
        }

        @Override
        public int tokenCount() {
            return tokenBytesById.length;
        }

        @Override
        public byte[] tokenBytes(int id) {
            return tokenBytesById[id];
        }

        @Override
        public int mergeCount() {
            return left.length;
        }

        @Override
        public int mergeLeft(int merge) {
            return left[merge];
        }

        @Override
        public int mergeRight(int merge) {
            return right[merge];
        }

        @Override
        public int mergeResult(int merge) {
            return result[merge];
        }

        @Override
        public int mergeRank(int merge) {
            return rank[merge];
        }

        @Override
        public boolean ignoreMerges() {
            return ignoreMerges;
        }
    }

    // ---- counting and decoding ----

    @Override
    public void encodeInto(
            CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw invalidRange(startInclusive, endExclusive, text.length());
        }
        byte[] utf8 = TiktokenSplitter.utf8(text, startInclusive, endExclusive);
        encodeWord(utf8, 0, utf8.length, out);
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        int total = countBytes(tokens); // which refuses an unknown id
        byte[] out = new byte[total];
        decodeKnownInto(tokens, 0, tokens.length(), out, 0, total);
        return out;
    }

    @Override
    public int countBytes(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        int total = 0;
        for (int i = 0, n = tokens.length(); i < n; i++) {
            total += spellingLength(tokens.intAt(i));
        }
        return total;
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(out, "out");
        int length = tokens.length();
        if (tokenStartIndex < 0 || tokenStartIndex > length) {
            throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
        }
        if (out.hasArray()) {
            return decodeBytesInto(tokens, tokenStartIndex, out, out.array());
        }
        int consumed = 0;
        for (int i = tokenStartIndex; i < length; i++) {
            int tokenId = tokens.intAt(i);
            byte[] tokenBytes =
                    (tokenId >= 0 && tokenId < tokenBytesById.length)
                            ? tokenBytesById[tokenId]
                            : null;
            if (tokenBytes == null) {
                throw unknownToken(tokenId);
            }
            if (tokenBytes.length > out.remaining()) {
                if (consumed == 0) {
                    throw new IllegalArgumentException(
                            "Not enough output space for token at index "
                                    + i
                                    + ": need "
                                    + tokenBytes.length
                                    + ", remaining "
                                    + out.remaining());
                }
                break;
            }
            out.put(tokenBytes);
            consumed++;
        }
        return consumed;
    }

    /**
     * Over the buffer's array: the tokens that fit are counted first, which tells where the output
     * ends, then written by {@link #decodeKnownInto}. Nothing past the output is written.
     */
    private int decodeBytesInto(IntSequence tokens, int from, ByteBuffer out, byte[] array) {
        int at = out.arrayOffset() + out.position();
        int end = out.arrayOffset() + out.limit();
        int last = from, stop = at; // the tokens that fit, and where their bytes end
        for (int n = tokens.length(); last < n; last++) {
            int length = spellingLength(tokens.intAt(last));
            if (stop + length > end) {
                break;
            }
            stop += length;
        }
        if (last == from && from < tokens.length()) {
            throw new IllegalArgumentException(
                    "Not enough output space for token at index "
                            + from
                            + ": need "
                            + spellingLength(tokens.intAt(from))
                            + ", remaining "
                            + (end - at));
        }
        decodeKnownInto(tokens, from, last, array, at, stop);
        out.position(stop - out.arrayOffset());
        return last - from;
    }

    /**
     * Writes the known ids {@code tokens[from, last)}, which spell exactly {@code [at, stop)} of
     * {@code array}: a short token in two 8-byte stores while its WIDE-byte window stays within the
     * output, the bytes past the token the next tokens' own; the last tokens of the output, and
     * long ones, copied exactly.
     */
    private void decodeKnownInto(
            IntSequence tokens, int from, int last, byte[] array, int at, int stop) {
        for (int i = from; i < last; i++) {
            int id = tokens.intAt(i);
            int length = shortLengths[id];
            if (length != 0 && at + WIDE <= stop) {
                LONGS.set(array, at, spellingWords[2 * id]);
                LONGS.set(array, at + 8, spellingWords[2 * id + 1]);
            } else {
                byte[] spelling = tokenBytesById[id];
                length = spelling.length;
                System.arraycopy(spelling, 0, array, at, length);
            }
            at += length;
        }
    }

    /** How many bytes an id spells; an unknown id is refused. */
    private int spellingLength(int id) {
        if (id >= 0 && id < tokenBytesById.length) {
            int length = shortLengths[id];
            if (length != 0) {
                return length;
            }
            byte[] spelling = tokenBytesById[id];
            if (spelling != null) {
                return spelling.length;
            }
        }
        throw unknownToken(id);
    }

    /** Names the engine encoder once one encodes for this model. */
    @Override
    public String toString() {
        for (int rules = 0; rules < encoders.length(); rules++) {
            Object encoder = encoders.get(rules);
            if (encoder != null && encoder != NO_ENCODER) {
                return "Tiktoken BPE, encoded by " + encoder.getClass().getName();
            }
        }
        Object words = this.words.get();
        if (words != null && words != NO_ENCODER) {
            return "Tiktoken BPE, words encoded by " + words.getClass().getName();
        }
        return "Tiktoken BPE";
    }

    // ---- tables ----

    private static int[] buildSingleByteTokenMap(Vocabulary vocabulary) {
        int[] map = new int[256];
        Arrays.fill(map, NO_TOKEN);
        for (Map.Entry<String, Integer> entry : vocabulary) {
            byte[] bytes = ByteLevel.decode(entry.getKey());
            if (bytes.length != 1) {
                continue;
            }
            int value = bytes[0] & 0xFF;
            int tokenId = entry.getValue();
            int current = map[value];
            if (current == NO_TOKEN || tokenId < current) {
                map[value] = tokenId;
            }
        }
        for (int i = 0; i < 256; i++) {
            if (map[i] == NO_TOKEN) {
                throw missingSingletonByte(i);
            }
        }
        return map;
    }

    private static IndexOutOfBoundsException invalidRange(
            int startInclusive, int endExclusive, int length) {
        return new IndexOutOfBoundsException(
                "Invalid range ["
                        + startInclusive
                        + ", "
                        + endExclusive
                        + ") for text length "
                        + length);
    }

    private static NoSuchElementException unknownToken(int tokenId) {
        return new NoSuchElementException(String.valueOf(tokenId));
    }

    private static UnsupportedTokenizerException missingSingletonByte(int value) {
        // a valid vocabulary, but byte-level BPE as toknroll runs it needs all 256
        return new UnsupportedTokenizerException("Missing singleton byte token for value " + value);
    }

    /**
     * Every id's bytes, by {@link VocabularyImpl#ids}: two ids can share a spelling (a special
     * token whose bytes are also an ordinary token, as Laguna's {@code |◊|}).
     */
    private static byte[][] buildTokenBytesById(Vocabulary vocabulary) {
        BitSet ids = VocabularyImpl.ids(vocabulary);
        byte[][] table = new byte[Math.max(vocabulary.size(), ids.length())][];
        for (int id = ids.nextSetBit(0); id >= 0; id = ids.nextSetBit(id + 1)) {
            table[id] = ByteLevel.decode(vocabulary.token(id));
        }
        return table;
    }

    private static void validateByteLevelVocabulary(Vocabulary vocabulary) {
        for (Map.Entry<String, Integer> e : vocabulary) {
            String token = e.getKey();
            int id = e.getValue();
            if (ByteLevel.isValidEncoding(token)) {
                continue;
            }
            throw new IllegalArgumentException(
                    "Token at id "
                            + id
                            + " is not valid ByteLevel encoding: "
                            + escapeToken(token)
                            + ". All vocabulary tokens must be byte-level encoded.");
        }
    }

    private static String escapeToken(String token) {
        StringBuilder out = new StringBuilder(token.length() * 2 + 2);
        out.append('"');
        for (int i = 0; i < token.length(); i++) {
            char ch = token.charAt(i);
            if (ch >= 0x20 && ch <= 0x7E && ch != '\\' && ch != '"') {
                out.append(ch);
                continue;
            }
            switch (ch) {
                case '\\':
                    out.append("\\\\");
                    break;
                case '"':
                    out.append("\\\"");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '\r':
                    out.append("\\r");
                    break;
                case '\t':
                    out.append("\\t");
                    break;
                default:
                    out.append("\\u");
                    String hex = Integer.toHexString(ch).toUpperCase();
                    for (int p = hex.length(); p < 4; p++) {
                        out.append('0');
                    }
                    out.append(hex);
            }
        }
        out.append('"');
        return out.toString();
    }
}
