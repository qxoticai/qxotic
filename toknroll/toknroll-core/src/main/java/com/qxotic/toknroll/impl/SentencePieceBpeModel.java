package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Vocabulary;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * SentencePiece-flavored BPE model.
 *
 * <p>This model is intentionally strict: vocabulary symbols and merge rules are consumed exactly as
 * provided. Any external normalization (for example metaspace conversions such as {@code '▁' <-> '
 * '}) must be applied by callers before creating the model.
 *
 * <p>Two merge strategies:
 *
 * <ul>
 *   <li>Short sequences (≤128 tokens): simple O(n²) scan-and-shift.
 *   <li>Long sequences: O(n log n) with an intrusive linked list, edge-cached merge candidates, and
 *       a primitive binary min-heap.
 * </ul>
 */
final class SentencePieceBpeModel extends AbstractTokenizationModel {

    private static final String BYTE_00_TOKEN = "<0x00>";
    private static final int BMP_SIZE = 1 << 16;
    private static final int BYTE_FALLBACK_SIZE = 256;
    private static final String MERGE_STRATEGY_PROPERTY = "toknroll.spbpe.merge.strategy";
    private static final String FAST_THRESHOLD_PROPERTY = "toknroll.spbpe.fast.threshold";
    private static final MergeStrategy MERGE_STRATEGY =
            parseMergeStrategy(System.getProperty(MERGE_STRATEGY_PROPERTY));
    private static final int FAST_MERGE_THRESHOLD = 128;
    private static final int NO_INDEX = -1;

    private final int[] byteTokenIds;
    private final String[] tokensById;
    private final Map<String, Integer> tokenToId;
    private final LongLongMap packedMerges;
    private final int[] bmpSingletonIds;
    private final boolean[] isByteToken;

    private enum MergeStrategy {
        AUTO,
        SIMPLE,
        FAST
    }

    private SentencePieceBpeModel(
            Vocabulary vocabulary,
            int[] byteTokenIds,
            String[] tokensById,
            Map<String, Integer> tokenToId,
            LongLongMap packedMerges,
            int[] bmpSingletonIds,
            boolean[] isByteToken,
            float expectedTokensPerChar) {
        super(vocabulary, expectedTokensPerChar);
        this.byteTokenIds = byteTokenIds;
        this.tokensById = tokensById;
        this.tokenToId = tokenToId;
        this.packedMerges = packedMerges;
        this.bmpSingletonIds = bmpSingletonIds;
        this.isByteToken = isByteToken;
    }

    static SentencePieceBpeModel fromVocabularyAndMerges(
            Vocabulary vocabulary, LongLongMap directMerges) {
        return create(vocabulary, Objects.requireNonNull(directMerges, "directMerges"));
    }

    /**
     * Compatibility factory used by tests that define SentencePiece merges via token scores.
     *
     * <p>When explicit merges are unavailable, this reconstructs a ranked merge table from
     * vocabulary + scores and then applies the normal greedy merge engine. This preserves
     * score-priority behavior while avoiding per-step dynamic pair rescans.
     */
    public static SentencePieceBpeModel fromVocabulary(Vocabulary vocabulary, float[] scores) {
        Objects.requireNonNull(vocabulary, "vocabulary");
        Objects.requireNonNull(scores, "scores");

        int maxId = -1;
        for (Map.Entry<String, Integer> e : vocabulary) {
            if (e.getValue() != null && e.getValue() > maxId) {
                maxId = e.getValue();
            }
        }
        if (maxId < 0) {
            return create(vocabulary, new LongLongMap(new long[0], new long[0]));
        }
        if (scores.length <= maxId) {
            throw new IllegalArgumentException(
                    "scores length "
                            + scores.length
                            + " must be greater than max token id "
                            + maxId);
        }

        return create(vocabulary, new LongLongMap(new long[0], new long[0]), scores);
    }

    private static SentencePieceBpeModel create(Vocabulary vocabulary, LongLongMap directMerges) {
        return create(vocabulary, directMerges, null);
    }

    private static SentencePieceBpeModel create(
            Vocabulary vocabulary, LongLongMap directMerges, float[] tokenScores) {
        Objects.requireNonNull(vocabulary, "vocabulary");

        int size = vocabulary.size();
        String[] tokensById = new String[size];
        Map<String, Integer> rawTokenToId = new HashMap<>(Math.max(16, size * 2));
        for (Map.Entry<String, Integer> entry : vocabulary) {
            String token = entry.getKey();
            Integer id = entry.getValue();
            if (token != null && id != null) {
                rawTokenToId.put(token, id);
                if (id >= 0 && id < size) {
                    tokensById[id] = token;
                }
            }
        }

        Map<String, Integer> tokenToId = new HashMap<>(rawTokenToId);

        int[] bmpSingletonIds = new int[BMP_SIZE];
        Arrays.fill(bmpSingletonIds, -1);
        for (Map.Entry<String, Integer> entry : rawTokenToId.entrySet()) {
            String token = entry.getKey();
            if (token.codePointCount(0, token.length()) == 1) {
                int cp = token.codePointAt(0);
                if (cp >= 0 && cp < bmpSingletonIds.length) {
                    bmpSingletonIds[cp] = entry.getValue();
                }
            }
        }

        LongLongMap packedMerges =
                tokenScores == null
                        ? directMerges
                        : buildPackedMergesFromScores(tokenToId, tokenScores);

        boolean[] isByteToken = new boolean[size];
        int[] byteTokenIds = new int[BYTE_FALLBACK_SIZE];
        Arrays.fill(byteTokenIds, -1);
        for (int i = 0; i < size; i++) {
            String token = tokensById[i];
            isByteToken[i] = isByteToken(vocabulary, i, token);
            if (isByteToken[i]) {
                int byteValue = tryParseByteTokenValue(token);
                if (byteValue >= 0) {
                    byteTokenIds[byteValue] = i;
                }
            }
        }

        if (rawTokenToId.containsKey(BYTE_00_TOKEN) && byteTokenIds[0] < 0) {
            throw new IllegalArgumentException("Token <0x00> exists but is not marked as BYTE");
        }
        validateByteFallbackTable(byteTokenIds);

        return new SentencePieceBpeModel(
                vocabulary,
                byteTokenIds,
                tokensById,
                tokenToId,
                packedMerges,
                bmpSingletonIds,
                isByteToken,
                estimateTokensPerChar(vocabulary.size()));
    }

    private static LongLongMap buildPackedMergesFromScores(
            Map<String, Integer> tokenToId, float[] tokenScores) {
        Objects.requireNonNull(tokenToId, "tokenToId");
        Objects.requireNonNull(tokenScores, "tokenScores");

        Map<Long, Integer> pairToMergedId = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            String merged = entry.getKey();
            Integer mergedIdObj = entry.getValue();
            if (merged == null || mergedIdObj == null || merged.length() < 2) {
                continue;
            }
            int mergedId = mergedIdObj;
            float mergedScore = tokenScores[mergedId];
            if (!Float.isFinite(mergedScore)) {
                continue;
            }

            int split = Character.charCount(merged.codePointAt(0));
            while (split < merged.length()) {
                Integer leftId = tokenToId.get(merged.substring(0, split));
                Integer rightId = tokenToId.get(merged.substring(split));
                if (leftId != null && rightId != null) {
                    long pair = IntPair.of(leftId, rightId);
                    pairToMergedId.putIfAbsent(pair, mergedId);
                }
                split += Character.charCount(merged.codePointAt(split));
            }
        }

        if (pairToMergedId.isEmpty()) {
            return new LongLongMap(new long[0], new long[0]);
        }

        long[] keys = new long[pairToMergedId.size()];
        int[] mergedIds = new int[pairToMergedId.size()];
        int write = 0;
        for (Map.Entry<Long, Integer> entry : pairToMergedId.entrySet()) {
            keys[write] = entry.getKey();
            mergedIds[write] = entry.getValue();
            write++;
        }

        Integer[] order = new Integer[mergedIds.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(
                order,
                (a, b) -> {
                    float as = tokenScores[mergedIds[a]];
                    float bs = tokenScores[mergedIds[b]];
                    int cmp = Float.compare(bs, as);
                    if (cmp != 0) {
                        return cmp;
                    }
                    return Integer.compare(mergedIds[a], mergedIds[b]);
                });

        int[] ranks = new int[mergedIds.length];
        int rank = -1;
        int prevBits = 0;
        boolean first = true;
        for (int idx : order) {
            int bits = Float.floatToRawIntBits(tokenScores[mergedIds[idx]]);
            if (first || bits != prevBits) {
                rank++;
                prevBits = bits;
                first = false;
            }
            ranks[idx] = rank;
        }

        long[] values = new long[mergedIds.length];
        for (int i = 0; i < mergedIds.length; i++) {
            values[i] = packMerge(ranks[i], mergedIds[i]);
        }

        return new LongLongMap(keys, values);
    }

    private static float estimateTokensPerChar(int vocabSize) {
        // Heuristic based on observed SPBPE corpora token density.
        // Values are conservative because capacity estimation includes
        // an additional safety factor.
        if (vocabSize <= 50000) {
            return 0.31f;
        } else if (vocabSize <= 150000) {
            return 0.29f;
        } else {
            return 0.28f;
        }
    }

    /** Packs a merge rank and merged token ID into a single long. */
    static long packMerge(int rank, int mergedId) {
        return ((long) rank << 32) | (mergedId & 0xFFFFFFFFL);
    }

    /** Extracts the merge rank from a packed merge value. */
    private static int unpackRank(long packed) {
        return (int) (packed >>> 32);
    }

    /** Extracts the merged token ID from a packed merge value. */
    private static int unpackMergedId(long packed) {
        return (int) packed;
    }

    @Override
    protected IntSequence encodeImpl(CharSequence text) {
        IntSequence.Builder out =
                IntSequence.newBuilder(estimateInitialTokenCapacity(text.length()));
        encodeImplInto(text, out);
        return out.build();
    }

    private int estimateInitialTokenCapacity(int charCount) {
        float ratio = Math.max(1.0e-6f, expectedTokensPerChar());
        int predicted = (int) Math.ceil(charCount * ratio * 1.15f) + 8;
        return Math.max(8, predicted);
    }

    @Override
    protected void encodeImplInto(CharSequence text, IntSequence.Builder out) {
        int textLen = text.length();
        int[] ids = new int[Math.max(16, textLen)];
        int size = 0;

        for (int pos = 0; pos < textLen; ) {
            int cp = Character.codePointAt(text, pos);
            int charCount = Character.charCount(cp);
            Integer id = resolveCodePointTokenId(text, pos, cp, charCount);

            if (id != null) {
                ids = ensureCapacity(ids, size + 1);
                ids[size++] = id;
            } else {
                byte[] utf8 =
                        text.subSequence(pos, pos + charCount)
                                .toString()
                                .getBytes(StandardCharsets.UTF_8);
                ids = ensureCapacity(ids, size + utf8.length);
                for (byte b : utf8) {
                    ids[size++] = requireByteFallbackTokenId(Byte.toUnsignedInt(b));
                }
            }
            pos += charCount;
        }

        int[] merged = mergeGreedy(ids, size);
        out.addAll(IntSequence.wrap(merged));
    }

    private int[] mergeGreedy(int[] ids, int length) {
        if (length < 2) {
            return Arrays.copyOf(ids, length);
        }
        switch (MERGE_STRATEGY) {
            case SIMPLE:
                return mergeGreedySimple(ids, length);
            case FAST:
                return mergeGreedyFast(ids, length);
            default:
                int threshold = readFastMergeThreshold();
                return length <= threshold
                        ? mergeGreedySimple(ids, length)
                        : mergeGreedyFast(ids, length);
        }
    }

    private static MergeStrategy parseMergeStrategy(String raw) {
        if (raw == null || raw.isBlank()) {
            return MergeStrategy.AUTO;
        }
        String value = raw.trim().toLowerCase(Locale.ROOT);
        switch (value) {
            case "auto":
                return MergeStrategy.AUTO;
            case "simple":
                return MergeStrategy.SIMPLE;
            case "fast":
                return MergeStrategy.FAST;
            default:
                throw new IllegalArgumentException(
                        "Invalid "
                                + MERGE_STRATEGY_PROPERTY
                                + "='"
                                + raw
                                + "' (expected: auto|simple|fast)");
        }
    }

    private static int readFastMergeThreshold() {
        String raw = System.getProperty(FAST_THRESHOLD_PROPERTY);
        if (raw == null || raw.isBlank()) {
            return FAST_MERGE_THRESHOLD;
        }
        try {
            return Math.max(0, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid " + FAST_THRESHOLD_PROPERTY + "='" + raw + "' (expected integer >= 0)",
                    e);
        }
    }

    /** Simple path: scan all adjacent pairs, pick the best (lowest rank), shift array. */
    private int[] mergeGreedySimple(int[] ids, int length) {
        int[] work = ids;
        int len = length;

        while (true) {
            int bestPos = -1;
            int bestMerged = -1;
            int bestRank = Integer.MAX_VALUE;

            for (int i = 0; i + 1 < len; i++) {
                long packed = packedMerges.getPair(work[i], work[i + 1]);
                if (packed == IntPair.NONE) {
                    continue;
                }
                int rank = unpackRank(packed);
                if (rank < bestRank || (rank == bestRank && i < bestPos)) {
                    bestRank = rank;
                    bestMerged = unpackMergedId(packed);
                    bestPos = i;
                }
            }

            if (bestPos < 0) {
                break;
            }

            work[bestPos] = bestMerged;
            if (bestPos + 1 < len - 1) {
                System.arraycopy(work, bestPos + 2, work, bestPos + 1, len - bestPos - 2);
            }
            len--;
        }

        return Arrays.copyOf(work, len);
    }

    // ---- Fast merge path: edge-cached linked list + min-heap ----

    /**
     * Fast path: intrusive doubly-linked list with per-node edge cache and a primitive min-heap.
     */
    private int[] mergeGreedyFast(int[] ids, int length) {
        Scratch s = new Scratch();
        return mergeGreedyFastImpl(ids, length, s);
    }

    private int[] mergeGreedyFastImpl(int[] ids, int length, Scratch s) {
        s.ensureNodes(length);

        int[] tokenIds = s.tokenIds;
        int[] next = s.next;
        int[] prev = s.prev;
        int[] edgeStamp = s.edgeStamp;
        int[] edgeRank = s.edgeRank;
        int[] edgeMergedId = s.edgeMergedId;

        // Initialize linked list and tokens.
        System.arraycopy(ids, 0, tokenIds, 0, length);
        for (int i = 0; i < length; i++) {
            prev[i] = i - 1;
            next[i] = i + 1;
            edgeStamp[i] = 0;
        }
        next[length - 1] = NO_INDEX;

        // Build initial edge cache and heap.
        s.heapSize = 0;
        for (int left = 0; left + 1 < length; left++) {
            refreshEdge(left, tokenIds, next, edgeStamp, edgeRank, edgeMergedId, s);
        }

        int tokenCount = length;
        while (tokenCount > 1 && popMinCandidate(edgeStamp, edgeMergedId, s)) {
            int left = s.popLeft;
            int mergedId = s.popMergedId;

            int right = next[left];
            if (right == NO_INDEX) {
                continue;
            }
            int leftPrev = prev[left];
            int rightNext = next[right];

            // Apply merge: left absorbs right.
            tokenIds[left] = mergedId;
            next[left] = rightNext;
            if (rightNext != NO_INDEX) {
                prev[rightNext] = left;
            }

            // Detach right node.
            prev[right] = NO_INDEX;
            next[right] = NO_INDEX;
            edgeStamp[right] = 0;
            tokenCount--;

            // Refresh affected edges.
            if (leftPrev != NO_INDEX) {
                refreshEdge(leftPrev, tokenIds, next, edgeStamp, edgeRank, edgeMergedId, s);
            }
            refreshEdge(left, tokenIds, next, edgeStamp, edgeRank, edgeMergedId, s);
        }

        // Compact: walk linked list to build result.
        int[] result = new int[tokenCount];
        int outPos = 0;
        for (int idx = 0; idx != NO_INDEX; idx = next[idx]) {
            result[outPos++] = tokenIds[idx];
        }
        return result;
    }

    /**
     * Recomputes the edge cache for the pair (left, next[left]) and pushes to the heap if valid.
     */
    private void refreshEdge(
            int left,
            int[] tokenIds,
            int[] next,
            int[] edgeStamp,
            int[] edgeRank,
            int[] edgeMergedId,
            Scratch s) {
        int right = next[left];
        if (right == NO_INDEX) {
            edgeStamp[left] = 0;
            return;
        }
        long packed = packedMerges.getPair(tokenIds[left], tokenIds[right]);
        if (packed == IntPair.NONE) {
            edgeStamp[left] = 0;
            return;
        }

        int rank = unpackRank(packed);
        int merged = unpackMergedId(packed);

        // Skip heap push if edge is unchanged (same rank and merged id).
        if (edgeStamp[left] != 0 && edgeRank[left] == rank && edgeMergedId[left] == merged) {
            return;
        }

        int stamp = s.nextStamp(edgeStamp);
        edgeStamp[left] = stamp;
        edgeRank[left] = rank;
        edgeMergedId[left] = merged;
        heapPush(rank, left, stamp, s);
    }

    /** Pops the minimum-rank candidate from the heap, skipping stale entries. */
    private static boolean popMinCandidate(int[] edgeStamp, int[] edgeMergedId, Scratch s) {
        while (s.heapSize > 0) {
            int rank = s.heapRank[0];
            long node = s.heapNode[0];
            int left = (int) (node >>> 32);
            int stamp = (int) node;

            int last = --s.heapSize;
            if (last > 0) {
                s.heapRank[0] = s.heapRank[last];
                s.heapNode[0] = s.heapNode[last];
                heapifyDown(s, 0);
            }

            if (edgeStamp[left] != stamp) {
                continue; // stale
            }

            s.popLeft = left;
            s.popMergedId = edgeMergedId[left];
            return true;
        }
        return false;
    }

    /** Pushes a merge candidate onto the min-heap. */
    private static void heapPush(int rank, int left, int stamp, Scratch s) {
        if (s.heapSize == s.heapRank.length) {
            s.ensureHeapSize(s.heapSize + 1);
        }
        int idx = s.heapSize++;
        long node = (((long) left) << 32) | (stamp & 0xFFFFFFFFL);

        // Sift up using copy-down.
        while (idx > 0) {
            int parent = (idx - 1) >>> 1;
            if (!heapLess(rank, node, s.heapRank[parent], s.heapNode[parent])) {
                break;
            }
            s.heapRank[idx] = s.heapRank[parent];
            s.heapNode[idx] = s.heapNode[parent];
            idx = parent;
        }

        s.heapRank[idx] = rank;
        s.heapNode[idx] = node;
    }

    /** Restores the min-heap invariant by sinking the element at {@code idx}. Copy-down style. */
    private static void heapifyDown(Scratch s, int idx) {
        int rank = s.heapRank[idx];
        long node = s.heapNode[idx];

        int half = s.heapSize >>> 1;
        while (idx < half) {
            int child = (idx << 1) + 1;
            int rightChild = child + 1;

            if (rightChild < s.heapSize
                    && heapLess(
                            s.heapRank[rightChild],
                            s.heapNode[rightChild],
                            s.heapRank[child],
                            s.heapNode[child])) {
                child = rightChild;
            }
            if (!heapLess(s.heapRank[child], s.heapNode[child], rank, node)) {
                break;
            }

            s.heapRank[idx] = s.heapRank[child];
            s.heapNode[idx] = s.heapNode[child];
            idx = child;
        }

        s.heapRank[idx] = rank;
        s.heapNode[idx] = node;
    }

    /** Returns true if candidate A has higher priority (lower rank, then lower left index). */
    private static boolean heapLess(int rankA, long nodeA, int rankB, long nodeB) {
        if (rankA != rankB) {
            return rankA < rankB;
        }
        int leftA = (int) (nodeA >>> 32);
        int leftB = (int) (nodeB >>> 32);
        return leftA < leftB;
    }

    // ---- Scratch buffer pool ----

    private static final class Scratch {
        int[] tokenIds = new int[0];
        int[] prev = new int[0];
        int[] next = new int[0];
        int[] edgeStamp = new int[0];
        int[] edgeRank = new int[0];
        int[] edgeMergedId = new int[0];
        int stampCounter = 1;

        int[] heapRank = new int[0];
        long[] heapNode = new long[0];
        int heapSize;

        int popLeft;
        int popMergedId;

        void ensureNodes(int needed) {
            if (tokenIds.length < needed) {
                int cap = grow(tokenIds.length, needed);
                tokenIds = new int[cap];
                prev = new int[cap];
                next = new int[cap];
                edgeStamp = new int[cap];
                edgeRank = new int[cap];
                edgeMergedId = new int[cap];
            }
            ensureHeapSize(Math.max(8, needed * 4));
        }

        int nextStamp(int[] stampArray) {
            int s = ++stampCounter;
            if (s == 0) {
                Arrays.fill(stampArray, 0);
                stampCounter = 1;
                return 1;
            }
            return s;
        }

        void ensureHeapSize(int needed) {
            if (heapRank.length < needed) {
                int cap = grow(heapRank.length, needed);
                heapRank = Arrays.copyOf(heapRank, cap);
                heapNode = Arrays.copyOf(heapNode, cap);
            }
        }

        private static int grow(int current, int needed) {
            int c = Math.max(8, current);
            while (c < needed) {
                c = c + (c >>> 1);
            }
            return c;
        }
    }

    // ---- Decode methods (unchanged) ----

    @Override
    public String decode(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        StringBuilder sb = new StringBuilder();
        ByteArrayOutputStream byteRun = new ByteArrayOutputStream();

        for (int i = 0; i < tokens.length(); i++) {
            int tokenId = tokens.intAt(i);
            String token = tokenText(tokenId);
            if (isByteTokenId(tokenId)) {
                byteRun.write(parseByteToken(token));
            } else {
                if (byteRun.size() > 0) {
                    sb.append(byteRun.toString(StandardCharsets.UTF_8));
                    byteRun.reset();
                }
                sb.append(token);
            }
        }

        if (byteRun.size() > 0) {
            sb.append(byteRun.toString(StandardCharsets.UTF_8));
        }
        return sb.toString();
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        return decode(tokens).getBytes(StandardCharsets.UTF_8);
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

        int index = tokenStartIndex;
        boolean wroteAny = false;

        while (index < length) {
            int tokenId = tokens.intAt(index);
            String token = tokenText(tokenId);

            byte[] piece;
            int consumed;

            if (isByteTokenId(tokenId)) {
                int runEnd = index + 1;
                while (runEnd < length) {
                    int nextId = tokens.intAt(runEnd);
                    if (!isByteTokenId(nextId)) {
                        break;
                    }
                    runEnd++;
                }

                byte[] raw = new byte[runEnd - index];
                for (int i = index; i < runEnd; i++) {
                    raw[i - index] = parseByteToken(tokenText(tokens.intAt(i)));
                }
                piece = new String(raw, StandardCharsets.UTF_8).getBytes(StandardCharsets.UTF_8);
                consumed = runEnd - index;
            } else {
                piece = token.getBytes(StandardCharsets.UTF_8);
                consumed = 1;
            }

            if (piece.length > out.remaining()) {
                if (!wroteAny) {
                    throw new IllegalArgumentException("Not enough output space");
                }
                break;
            }

            out.put(piece);
            wroteAny = true;
            index += consumed;
        }

        return index - tokenStartIndex;
    }

    @Override
    public int countBytes(IntSequence tokens) {
        return decodeBytes(tokens).length;
    }

    // ---- Utilities ----

    private static int[] ensureCapacity(int[] array, int needed) {
        if (needed <= array.length) {
            return array;
        }
        int next = array.length;
        while (next < needed) {
            next <<= 1;
        }
        return Arrays.copyOf(array, next);
    }

    private String tokenText(int id) {
        String token = tokenTextOrNull(id);
        if (token == null) {
            throw new IllegalArgumentException("Unknown token id: " + id);
        }
        return token;
    }

    private String tokenTextOrNull(int id) {
        return (id >= 0 && id < tokensById.length) ? tokensById[id] : null;
    }

    private Integer resolveCodePointTokenId(
            CharSequence text, int pos, int codePoint, int charCount) {
        if (codePoint < bmpSingletonIds.length) {
            int sid = bmpSingletonIds[codePoint];
            return sid >= 0 ? sid : null;
        }
        return tokenToId.get(text.subSequence(pos, pos + charCount).toString());
    }

    private boolean isByteTokenId(int tokenId) {
        return tokenId >= 0 && tokenId < isByteToken.length && isByteToken[tokenId];
    }

    private int requireByteFallbackTokenId(int unsignedByte) {
        int byteTokenId = byteTokenIds[unsignedByte];
        if (byteTokenId < 0) {
            throw new IllegalArgumentException(
                    String.format("Missing byte fallback token <0x%02X>", unsignedByte));
        }
        return byteTokenId;
    }

    private static boolean isByteToken(Vocabulary vocabulary, int tokenId, String token) {
        return token != null
                && token.length() == 6
                && token.startsWith("<0x")
                && token.endsWith(">")
                && vocabulary.isTokenOfType(tokenId, StandardTokenType.BYTE);
    }

    private static void validateByteFallbackTable(int[] byteTokenIds) {
        int byteTokenCount = 0;
        boolean hasNonZeroByteToken = false;
        for (int i = 0; i < byteTokenIds.length; i++) {
            if (byteTokenIds[i] >= 0) {
                byteTokenCount++;
                if (i != 0) {
                    hasNonZeroByteToken = true;
                }
            }
        }
        if (hasNonZeroByteToken || byteTokenCount > 1) {
            for (int i = 0; i < byteTokenIds.length; i++) {
                if (byteTokenIds[i] < 0) {
                    throw new IllegalArgumentException(
                            String.format("Byte fallback table is partial, missing <0x%02X>", i));
                }
            }
        }
    }

    private static byte parseByteToken(String token) {
        return (byte) Integer.parseInt(token.substring(3, 5), 16);
    }

    private static int tryParseByteTokenValue(String token) {
        if (token == null
                || token.length() != 6
                || !token.startsWith("<0x")
                || !token.endsWith(">")) {
            return -1;
        }
        try {
            return Integer.parseInt(token.substring(3, 5), 16);
        } catch (NumberFormatException e) {
            return -1;
        }
    }
}
