package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.spi.IdBuffer;
import com.qxotic.toknroll.spi.TiktokenEngine;
import com.qxotic.toknroll.spi.TiktokenTables;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

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

    private final LongLongMap merges;
    private final int[] singleByteTokenId;
    private final byte[][] tokenBytesById;
    private final boolean ignoreMerges;
    private final Map<String, Integer>
            wholeTokens; // with ignoreMerges: a word's bytes -> its token
    private final Map<String, int[]> memo = new ConcurrentHashMap<>();

    // The engine's encoder per rule set, resolved on first use. A benign race: every resolution
    // has the same outcome, and one that sees the flag before the encoder encodes here meanwhile.
    private final TiktokenEngine.Encoder[] encoders = new TiktokenEngine.Encoder[7];
    private final boolean[] resolved = new boolean[7];

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
        this.ignoreMerges = ignoreMerges;
        this.wholeTokens = ignoreMerges ? wholeTokens(tokenBytesById) : null;
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

    private static Map<String, Integer> wholeTokens(byte[][] tokenBytesById) {
        Map<String, Integer> tokens = new HashMap<>();
        for (int id = 0; id < tokenBytesById.length; id++) {
            byte[] bytes = tokenBytesById[id];
            if (bytes != null && bytes.length > 0) {
                tokens.putIfAbsent(new String(bytes, StandardCharsets.ISO_8859_1), id);
            }
        }
        return tokens;
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
        byte[] utf8 = TiktokenSplitter.utf8(text, 0, text.length());
        encodeWord(utf8, 0, utf8.length - TiktokenSplitter.PADDING, out);
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
        if (fast != null) {
            IdBuffer ids = borrow(out);
            fast.encode(text, start, end, ids);
            giveBack(ids, out);
            return;
        }
        byte[] utf8 = TiktokenSplitter.utf8(text, start, end);
        encode(utf8, 0, utf8.length - TiktokenSplitter.PADDING, splitter, out);
    }

    /** {@link #encode(CharSequence, int, int, TiktokenSplitter, IntSequence.Builder)} for UTF-8. */
    void encodeUtf8(
            byte[] utf8, int start, int end, TiktokenSplitter splitter, IntSequence.Builder out) {
        TiktokenEngine.Encoder fast = encoder(splitter);
        if (fast != null) {
            IdBuffer ids = borrow(out);
            fast.encodeUtf8(utf8, start, end, ids);
            giveBack(ids, out);
            return;
        }
        // the scanner reads past a piece, so it gets a padded copy
        byte[] padded = Arrays.copyOfRange(utf8, start, end + TiktokenSplitter.PADDING);
        encode(padded, 0, end - start, splitter, out);
    }

    /** The reference: the scanner's pieces, word by word. */
    private void encode(
            byte[] text, int start, int end, TiktokenSplitter splitter, IntSequence.Builder out) {
        int[] ends = new int[TiktokenSplitter.BATCH];
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
            Integer whole =
                    wholeTokens.get(
                            new String(text, start, end - start, StandardCharsets.ISO_8859_1));
            if (whole != null) {
                return new int[] {whole};
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
        if (!resolved[rules]) {
            encoders[rules] = findEncoder(rules, splitter);
            resolved[rules] = true;
        }
        return encoders[rules];
    }

    private TiktokenEngine.Encoder findEncoder(int rules, TiktokenSplitter splitter) {
        if ("none".equals(System.getProperty("toknroll.engine"))) {
            return null;
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
        return null;
    }

    /** Whether {@code candidate} encodes the sample as this model does, from chars and bytes. */
    private boolean agrees(TiktokenEngine.Encoder candidate, TiktokenSplitter splitter) {
        String sample = TiktokenSplitter.sample();
        byte[] utf8 = TiktokenSplitter.utf8(sample, 0, sample.length());
        int length = utf8.length - TiktokenSplitter.PADDING;
        IntSequenceBuilder expected = new IntSequenceBuilder();
        encode(utf8, 0, length, splitter, expected);
        IdBuffer fromChars = new IdBuffer(expected.size());
        candidate.encode(sample, 0, sample.length(), fromChars);
        IdBuffer fromBytes = new IdBuffer(expected.size());
        candidate.encodeUtf8(Arrays.copyOf(utf8, length), 0, length, fromBytes);
        int[] want = Arrays.copyOf(expected.data(), expected.size());
        return Arrays.equals(want, Arrays.copyOf(fromChars.array(), fromChars.size()))
                && Arrays.equals(want, Arrays.copyOf(fromBytes.array(), fromBytes.size()));
    }

    /** {@code out}'s own array when core made it, so the engine writes the ids there. */
    private static IdBuffer borrow(IntSequence.Builder out) {
        if (out instanceof IntSequenceBuilder) {
            IntSequenceBuilder builder = (IntSequenceBuilder) out;
            return new IdBuffer(builder.data(), builder.size());
        }
        return new IdBuffer(64);
    }

    private static void giveBack(IdBuffer ids, IntSequence.Builder out) {
        if (out instanceof IntSequenceBuilder) {
            ((IntSequenceBuilder) out).adopt(ids.array(), ids.size());
            return;
        }
        int[] array = ids.array();
        out.ensureCapacity(out.size() + ids.size());
        for (int i = 0; i < ids.size(); i++) {
            out.add(array[i]);
        }
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
        encodeWord(utf8, 0, utf8.length - TiktokenSplitter.PADDING, out);
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        IntSequence.Builder out =
                IntSequence.newBuilder(estimateInitialTokenCapacity(endExclusive - startInclusive));
        encodeInto(text, startInclusive, endExclusive, out);
        return out.size();
    }

    @Override
    public int countBytes(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        int length = tokens.length();
        int totalBytes = 0;
        for (int i = 0; i < length; i++) {
            int tokenId = tokens.intAt(i);
            byte[] tokenBytes =
                    (tokenId >= 0 && tokenId < tokenBytesById.length)
                            ? tokenBytesById[tokenId]
                            : null;
            if (tokenBytes == null) {
                throw unknownToken(tokenId);
            }
            totalBytes += tokenBytes.length;
        }
        return totalBytes;
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(out, "out");
        int length = tokens.length();
        if (tokenStartIndex < 0 || tokenStartIndex > length) {
            throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
        }
        if (tokenStartIndex == length) {
            return 0;
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

    /** Names the engine encoder once one encodes for this model. */
    @Override
    public String toString() {
        for (TiktokenEngine.Encoder encoder : encoders) {
            if (encoder != null) {
                return "Tiktoken BPE, encoded by " + encoder.getClass().getName();
            }
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

    private static IllegalArgumentException missingSingletonByte(int value) {
        return new IllegalArgumentException("Missing singleton byte token for value " + value);
    }

    private static byte[][] buildTokenBytesById(Vocabulary vocabulary) {
        int maxId = -1;
        for (Map.Entry<String, Integer> e : vocabulary) {
            if (e.getValue() > maxId) {
                maxId = e.getValue();
            }
        }
        byte[][] table = new byte[maxId + 1][];
        for (Map.Entry<String, Integer> e : vocabulary) {
            table[e.getValue()] = ByteLevel.decode(e.getKey());
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
