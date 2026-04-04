package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/** Generic, reusable BPE tokenizer implementation. */
public final class GenericBpeTokenizer extends AbstractTokenizer {

    public static final String LARGE_CHUNK_THRESHOLD_PROPERTY =
            "qxotic.tokenizer.generic.largeChunkThreshold";
    public static final String TINY_CHUNK_THRESHOLD_PROPERTY =
            "qxotic.tokenizer.generic.tinyChunkThreshold";

    private static final int DEFAULT_LARGE_CHUNK_THRESHOLD = 96;
    private static final int DEFAULT_TINY_CHUNK_THRESHOLD = 3;
    private static final int NO_INDEX = -1;

    private final BpeMergeTable mergeTable;
    private final BpeSymbolEncoder symbolEncoder;
    private final byte[][] decodedTokenBytesCache;
    private final int tinyChunkThreshold;
    private final int largeChunkThreshold;
    private final AtomicReference<Scratch> scratchSlot = new AtomicReference<>();

    public GenericBpeTokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            BpeMergeTable mergeTable,
            BpeSymbolEncoder symbolEncoder) {
        this(
                vocabulary,
                normalizer,
                splitter,
                mergeTable,
                symbolEncoder,
                Integer.getInteger(TINY_CHUNK_THRESHOLD_PROPERTY, DEFAULT_TINY_CHUNK_THRESHOLD),
                Integer.getInteger(LARGE_CHUNK_THRESHOLD_PROPERTY, DEFAULT_LARGE_CHUNK_THRESHOLD));
    }

    public GenericBpeTokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            BpeMergeTable mergeTable,
            BpeSymbolEncoder symbolEncoder,
            int tinyChunkThreshold,
            int largeChunkThreshold) {
        super(vocabulary, normalizer, splitter);
        this.mergeTable = Objects.requireNonNull(mergeTable, "mergeTable");
        this.symbolEncoder = Objects.requireNonNull(symbolEncoder, "symbolEncoder");
        this.decodedTokenBytesCache = new byte[vocabulary.size()][];
        this.tinyChunkThreshold = Math.max(1, Math.min(3, tinyChunkThreshold));
        this.largeChunkThreshold = Math.max(8, largeChunkThreshold);
    }

    @Override
    protected IntSequence encodeImpl(CharSequence text) {
        IntSequence.Builder out = IntSequence.newBuilder(Math.max(8, text.length()));
        encodeImplInto(text, out);
        return out.build();
    }

    @Override
    protected void encodeImplInto(CharSequence text, IntSequence.Builder out) {
        Scratch s = acquireScratch();
        try {
            int size = encodeInitialTokens(text, 0, text.length(), s);
            mergeChunk(s.encodedTokens, size, out, true, s);
        } finally {
            releaseScratch(s);
        }
    }

    @Override
    public void encodeInto(
            CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");
        requireValidRange(text, startInclusive, endExclusive);

        CharSequence normalized =
                startInclusive == 0 && endExclusive == text.length()
                        ? normalizer.apply(text)
                        : normalizer.apply(text.subSequence(startInclusive, endExclusive));

        Scratch s = acquireScratch();
        try {
            out.ensureCapacity(out.size() + Math.max(8, normalized.length()));
            splitter.splitAll(
                    normalized,
                    0,
                    normalized.length(),
                    (source, chunkStart, chunkEnd) -> {
                        int size = encodeInitialTokens(source, chunkStart, chunkEnd, s);
                        mergeChunk(s.encodedTokens, size, out, true, s);
                    });
        } finally {
            releaseScratch(s);
        }
    }

    @Override
    public int countTokens(CharSequence text) {
        Objects.requireNonNull(text, "text");
        CharSequence normalized = normalizer.apply(text);
        Scratch s = acquireScratch();
        try {
            s.countAccumulator = 0;
            splitter.splitAll(
                    normalized,
                    0,
                    normalized.length(),
                    (source, startInclusive, endExclusive) -> {
                        int size = encodeInitialTokens(source, startInclusive, endExclusive, s);
                        s.countAccumulator += mergeChunk(s.encodedTokens, size, null, false, s);
                    });
            return s.countAccumulator;
        } finally {
            releaseScratch(s);
        }
    }

    private int encodeInitialTokens(
            CharSequence source, int startInclusive, int endExclusive, Scratch s) {
        int charLen = endExclusive - startInclusive;
        if (charLen <= 0) {
            s.encodedTokens = Scratch.EMPTY_INT;
            return 0;
        }
        int needed = symbolEncoder.maxEncodedLength(charLen);
        s.ensureEncodedTokens(needed);
        s.encodedTokens = s.encodedTokenScratch;
        return symbolEncoder.encodeChunkToTokenIdsInto(
                source, startInclusive, endExclusive, vocabulary(), s.encodedTokens);
    }

    private Scratch acquireScratch() {
        Scratch s = scratchSlot.getAndSet(null);
        return s != null ? s : new Scratch();
    }

    private void releaseScratch(Scratch s) {
        scratchSlot.compareAndSet(null, s);
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        int totalBytes = decodedByteLength(tokens);
        int length = tokens.length();

        byte[] out = new byte[totalBytes];
        ByteBuffer buffer = ByteBuffer.wrap(out);
        int tokenIndex = 0;
        while (tokenIndex < length) {
            int consumed = decodeBytesInto(tokens, tokenIndex, buffer);
            if (consumed <= 0) {
                throw new IllegalStateException(
                        "decodeBytesInto made no progress at token " + tokenIndex);
            }
            tokenIndex += consumed;
        }
        return out;
    }

    @Override
    public int countBytes(IntSequence tokens) {
        return decodedByteLength(tokens);
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
            byte[] chunk = tokenBytes(tokens.intAt(i));
            if (chunk.length > out.remaining()) {
                if (consumed == 0) {
                    throw insufficientSpace(i, chunk.length, out.remaining());
                }
                break;
            }
            out.put(chunk);
            consumed++;
        }
        return consumed;
    }

    @Override
    public String toString() {
        return "Generic BPE Tokenizer";
    }

    private int decodedByteLength(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        int length = tokens.length();
        int totalBytes = 0;
        for (int i = 0; i < length; i++) {
            totalBytes += tokenBytes(tokens.intAt(i)).length;
        }
        return totalBytes;
    }

    private byte[] tokenBytes(int tokenId) {
        if (tokenId < 0
                || tokenId >= decodedTokenBytesCache.length
                || !vocabulary().contains(tokenId)) {
            throw new NoSuchElementException(String.valueOf(tokenId));
        }
        byte[] cached = decodedTokenBytesCache[tokenId];
        if (cached == null) {
            cached = symbolEncoder.decodeTokenBytes(tokenId, vocabulary());
            decodedTokenBytesCache[tokenId] = cached;
        }
        return cached;
    }

    private static IllegalArgumentException insufficientSpace(
            int tokenIndex, int needed, int remaining) {
        return new IllegalArgumentException(
                "Not enough output space for token at index "
                        + tokenIndex
                        + ": need "
                        + needed
                        + ", remaining "
                        + remaining);
    }

    private int mergeChunk(
            int[] inputTokens, int size, IntSequence.Builder out, boolean keepOutput, Scratch s) {
        if (size == 0) {
            return 0;
        }
        if (size <= tinyChunkThreshold) {
            return keepOutput
                    ? mergeTiny(inputTokens, size, out, true)
                    : mergeTiny(inputTokens, size, null, false);
        }
        if (size < largeChunkThreshold) {
            return keepOutput
                    ? encodeSmall(inputTokens, size, out, s)
                    : countSmall(inputTokens, size, s);
        }
        return keepOutput
                ? encodeLarge(inputTokens, size, out, s)
                : countLarge(inputTokens, size, s);
    }

    private static void requireValidRange(CharSequence text, int startInclusive, int endExclusive) {
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for text length "
                            + text.length());
        }
    }

    private int mergeTiny(int[] tokens, int size, IntSequence.Builder out, boolean keepOutput) {
        if (size == 1) {
            if (keepOutput) {
                out.add(tokens[0]);
            }
            return 1;
        }
        if (size == 3) {
            int t0 = tokens[0];
            int t1 = tokens[1];
            int t2 = tokens[2];
            long m01 = mergeTable.mergeInfo(t0, t1);
            long m12 = mergeTable.mergeInfo(t1, t2);
            int r01 = (m01 == IntPair.NONE) ? -1 : IntPair.right(m01);
            int r12 = (m12 == IntPair.NONE) ? -1 : IntPair.right(m12);

            if (r01 < 0 && r12 < 0) {
                if (keepOutput) {
                    out.add(t0);
                    out.add(t1);
                    out.add(t2);
                }
                return 3;
            }
            if (r12 < 0 || (r01 >= 0 && r01 <= r12)) {
                int merged01 = IntPair.left(m01);
                long m012 = mergeTable.mergeInfo(merged01, t2);
                if (m012 != IntPair.NONE) {
                    if (keepOutput) {
                        out.add(IntPair.left(m012));
                    }
                    return 1;
                }
                if (keepOutput) {
                    out.add(merged01);
                    out.add(t2);
                }
                return 2;
            }

            int merged12 = IntPair.left(m12);
            long m012 = mergeTable.mergeInfo(t0, merged12);
            if (m012 != IntPair.NONE) {
                if (keepOutput) {
                    out.add(IntPair.left(m012));
                }
                return 1;
            }
            if (keepOutput) {
                out.add(t0);
                out.add(merged12);
            }
            return 2;
        }

        long m = mergeTable.mergeInfo(tokens[0], tokens[1]);
        if (m != IntPair.NONE) {
            if (keepOutput) {
                out.add(IntPair.left(m));
            }
            return 1;
        }
        if (keepOutput) {
            out.add(tokens[0]);
            out.add(tokens[1]);
        }
        return 2;
    }

    private int encodeSmall(int[] inputTokens, int size, IntSequence.Builder out, Scratch s) {
        return mergeSmall(inputTokens, size, out, true, s);
    }

    private int countSmall(int[] inputTokens, int size, Scratch s) {
        return mergeSmall(inputTokens, size, null, false, s);
    }

    private int mergeSmall(
            int[] inputTokens, int size, IntSequence.Builder out, boolean keepOutput, Scratch s) {
        s.ensureTokens(size);
        s.ensurePairRanks(size - 1);
        int[] tokens = s.tokens;
        int[] pairRanks = s.pairRanks;
        System.arraycopy(inputTokens, 0, tokens, 0, size);

        for (int i = 0; i + 1 < size; i++) {
            long info = mergeTable.mergeInfo(tokens[i], tokens[i + 1]);
            pairRanks[i] = (info == IntPair.NONE) ? -1 : IntPair.right(info);
        }

        while (size >= 2) {
            int bestRank = Integer.MAX_VALUE;
            int bestPos = NO_INDEX;
            int pairCount = size - 1;

            for (int i = 0; i < pairCount; i++) {
                int rank = pairRanks[i];
                if (rank >= 0 && rank < bestRank) {
                    bestRank = rank;
                    bestPos = i;
                }
            }
            if (bestPos == NO_INDEX) {
                break;
            }

            long merged = mergeTable.mergeInfo(tokens[bestPos], tokens[bestPos + 1]);
            if (merged == IntPair.NONE) {
                pairRanks[bestPos] = -1;
                continue;
            }
            tokens[bestPos] = IntPair.left(merged);

            int tail = size - bestPos - 2;
            if (tail > 0) {
                System.arraycopy(tokens, bestPos + 2, tokens, bestPos + 1, tail);
                System.arraycopy(pairRanks, bestPos + 1, pairRanks, bestPos, tail);
            }
            size--;

            int newPairCount = size - 1;
            if (bestPos > 0) {
                long leftInfo = mergeTable.mergeInfo(tokens[bestPos - 1], tokens[bestPos]);
                pairRanks[bestPos - 1] = (leftInfo == IntPair.NONE) ? -1 : IntPair.right(leftInfo);
            }
            if (bestPos < newPairCount) {
                long rightInfo = mergeTable.mergeInfo(tokens[bestPos], tokens[bestPos + 1]);
                pairRanks[bestPos] = (rightInfo == IntPair.NONE) ? -1 : IntPair.right(rightInfo);
            }
        }

        if (keepOutput) {
            out.ensureCapacity(out.size() + size);
            for (int i = 0; i < size; i++) {
                out.add(tokens[i]);
            }
        }
        return size;
    }

    private int encodeLarge(int[] inputTokens, int size, IntSequence.Builder out, Scratch s) {
        return mergeLarge(inputTokens, size, out, true, s);
    }

    private int countLarge(int[] inputTokens, int size, Scratch s) {
        return mergeLarge(inputTokens, size, null, false, s);
    }

    private int mergeLarge(
            int[] inputTokens, int size, IntSequence.Builder out, boolean keepOutput, Scratch s) {
        s.ensureNodes(size);
        int[] token = s.nodeToken;
        int[] prev = s.prev;
        int[] next = s.next;
        int[] edgeStamp = s.edgeStamp;
        int[] edgeRight = s.edgeRight;
        int[] edgeRank = s.edgeRank;
        int[] edgeMerged = s.edgeMerged;

        System.arraycopy(inputTokens, 0, token, 0, size);
        for (int i = 0; i < size; i++) {
            prev[i] = i - 1;
            next[i] = (i + 1 < size) ? i + 1 : NO_INDEX;
            edgeStamp[i] = 0;
        }

        s.heapSize = 0;
        for (int left = 0; left + 1 < size; left++) {
            refreshEdge(left, token, next, edgeStamp, edgeRight, edgeRank, edgeMerged, s);
        }

        int tokenCount = size;
        while (tokenCount > 1 && popMinCandidate(edgeStamp, edgeRank, s)) {
            int left = s.popLeft;
            int mergedToken = s.popToken;
            int right = next[left];
            if (right == NO_INDEX) {
                continue;
            }
            int leftPrev = prev[left];
            int rightNext = next[right];

            token[left] = mergedToken;
            next[left] = rightNext;
            if (rightNext != NO_INDEX) {
                prev[rightNext] = left;
            }
            tokenCount--;

            if (leftPrev != NO_INDEX) {
                refreshEdge(leftPrev, token, next, edgeStamp, edgeRight, edgeRank, edgeMerged, s);
            }
            refreshEdge(left, token, next, edgeStamp, edgeRight, edgeRank, edgeMerged, s);
        }

        if (keepOutput) {
            out.ensureCapacity(out.size() + tokenCount);
            for (int i = 0; i != NO_INDEX; i = next[i]) {
                out.add(token[i]);
            }
        }
        return tokenCount;
    }

    private void refreshEdge(
            int left,
            int[] token,
            int[] next,
            int[] edgeStamp,
            int[] edgeRight,
            int[] edgeRank,
            int[] edgeMerged,
            Scratch s) {
        int right = next[left];
        if (right == NO_INDEX) {
            edgeStamp[left] = 0;
            return;
        }
        long info = mergeTable.mergeInfo(token[left], token[right]);
        if (info == IntPair.NONE) {
            edgeStamp[left] = 0;
            return;
        }
        int rank = IntPair.right(info);
        int mergedToken = IntPair.left(info);

        if (edgeStamp[left] != 0 && edgeRight[left] == right && edgeRank[left] == rank) {
            return;
        }

        int stamp = s.nextStamp(edgeStamp);
        edgeStamp[left] = stamp;
        edgeRight[left] = right;
        edgeRank[left] = rank;
        edgeMerged[left] = mergedToken;
        heapPush(rank, left, stamp, s);
    }

    private boolean popMinCandidate(int[] edgeStamp, int[] edgeRank, Scratch s) {
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

            if (edgeStamp[left] != stamp || edgeRank[left] != rank) {
                continue;
            }

            s.popLeft = left;
            s.popToken = s.edgeMerged[left];
            return true;
        }
        return false;
    }

    private void heapPush(int rank, int left, int stamp, Scratch s) {
        if (s.heapSize == s.heapRank.length) {
            s.ensureHeap(Math.max(8, s.heapSize + 1));
        }
        int idx = s.heapSize++;
        long node = (((long) left) << 32) | (stamp & 0xFFFFFFFFL);

        while (idx > 0) {
            int parent = (idx - 1) >>> 1;
            if (s.heapRank[parent] <= rank) {
                break;
            }
            s.heapRank[idx] = s.heapRank[parent];
            s.heapNode[idx] = s.heapNode[parent];
            idx = parent;
        }
        s.heapRank[idx] = rank;
        s.heapNode[idx] = node;
    }

    private void heapifyDown(Scratch s, int idx) {
        int rank = s.heapRank[idx];
        long node = s.heapNode[idx];
        int half = s.heapSize >>> 1;
        while (idx < half) {
            int child = (idx << 1) + 1;
            int right = child + 1;
            if (right < s.heapSize && s.heapRank[right] < s.heapRank[child]) {
                child = right;
            }
            if (s.heapRank[child] >= rank) {
                break;
            }
            s.heapRank[idx] = s.heapRank[child];
            s.heapNode[idx] = s.heapNode[child];
            idx = child;
        }
        s.heapRank[idx] = rank;
        s.heapNode[idx] = node;
    }

    private static final class Scratch {
        static final int[] EMPTY_INT = new int[0];

        int[] encodedTokenScratch = EMPTY_INT;
        int[] encodedTokens = EMPTY_INT;

        int[] tokens = EMPTY_INT;
        int[] pairRanks = EMPTY_INT;

        int[] nodeToken = EMPTY_INT;
        int[] prev = EMPTY_INT;
        int[] next = EMPTY_INT;
        int[] edgeStamp = EMPTY_INT;
        int[] edgeRight = EMPTY_INT;
        int[] edgeRank = EMPTY_INT;
        int[] edgeMerged = EMPTY_INT;

        int[] heapRank = EMPTY_INT;
        long[] heapNode = new long[0];
        int heapSize;
        int stampCounter = 1;

        int popLeft;
        int popToken;
        int countAccumulator;

        void ensureEncodedTokens(int needed) {
            if (encodedTokenScratch.length < needed) {
                encodedTokenScratch = new int[grow(encodedTokenScratch.length, needed)];
            }
        }

        void ensureTokens(int needed) {
            if (tokens.length < needed) {
                tokens = new int[grow(tokens.length, needed)];
            }
        }

        void ensurePairRanks(int needed) {
            if (pairRanks.length < needed) {
                pairRanks = new int[grow(pairRanks.length, needed)];
            }
        }

        void ensureNodes(int needed) {
            if (nodeToken.length < needed) {
                int cap = grow(nodeToken.length, needed);
                nodeToken = new int[cap];
                prev = new int[cap];
                next = new int[cap];
                edgeStamp = new int[cap];
                edgeRight = new int[cap];
                edgeRank = new int[cap];
                edgeMerged = new int[cap];
            }
            ensureHeap(Math.max(8, needed * 4));
        }

        int nextStamp(int[] stamps) {
            int s = ++stampCounter;
            if (s == 0) {
                Arrays.fill(stamps, 0);
                stampCounter = 1;
                return 1;
            }
            return s;
        }

        void ensureHeap(int needed) {
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
}
