package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Fast, flat-array TikToken-compatible BPE tokenizer.
 *
 * <p>Implements a dual-path merge strategy:
 *
 * <ul>
 *   <li>Small chunks: in-place scan/merge (low constant overhead)
 *   <li>Large chunks: min-heap + intrusive list over primitive arrays
 * </ul>
 */
public final class FastTikToken extends AbstractTokenizationModel {

    public static final String LARGE_CHUNK_THRESHOLD_PROPERTY = "toknroll.fast.largeChunkThreshold";
    public static final String TINY_CHUNK_THRESHOLD_PROPERTY = "toknroll.fast.tinyChunkThreshold";
    public static final String EXACT_LOOKUP_ENABLED_PROPERTY = "toknroll.fast.exactLookupEnabled";

    private static final byte REPLACEMENT_B0 = (byte) 0xEF;
    private static final byte REPLACEMENT_B1 = (byte) 0xBF;
    private static final byte REPLACEMENT_B2 = (byte) 0xBD;
    private static final int DEFAULT_LARGE_CHUNK_THRESHOLD = 96;
    private static final int DEFAULT_TINY_CHUNK_THRESHOLD = 3;
    private static final int NO_TOKEN = -1;
    private static final int NO_INDEX = -1;

    private final LongLongMap merges;
    private final int[] singleByteTokenId;
    private final byte[][] tokenBytesById;
    private final ExactTokenLookup exactTokenLookup;
    private final int tinyChunkThreshold;
    private final int largeChunkThreshold;
    private final ConcurrentLinkedQueue<Scratch> scratchPool = new ConcurrentLinkedQueue<>();

    public FastTikToken(
            Vocabulary vocabulary,
            LongLongMap merges,
            int[] singleByteTokenId,
            byte[][] tokenBytesById,
            ExactTokenLookup exactTokenLookup,
            int tinyChunkThreshold,
            int largeChunkThreshold) {
        super(vocabulary);
        this.merges = Objects.requireNonNull(merges, "merges");
        this.singleByteTokenId = Objects.requireNonNull(singleByteTokenId, "singleByteTokenId");
        if (singleByteTokenId.length != 256) {
            throw new IllegalArgumentException("singleByteTokenId length must be 256");
        }
        this.tokenBytesById = Objects.requireNonNull(tokenBytesById, "tokenBytesById");
        this.exactTokenLookup = Objects.requireNonNull(exactTokenLookup, "exactTokenLookup");
        this.tinyChunkThreshold = Math.max(1, Math.min(3, tinyChunkThreshold));
        this.largeChunkThreshold = Math.max(8, largeChunkThreshold);
    }

    public FastTikToken(
            Vocabulary vocabulary,
            LongLongMap merges,
            int[] singleByteTokenId,
            byte[][] tokenBytesById,
            ExactTokenLookup exactTokenLookup) {
        this(
                vocabulary,
                merges,
                singleByteTokenId,
                tokenBytesById,
                exactTokenLookup,
                DEFAULT_TINY_CHUNK_THRESHOLD,
                DEFAULT_LARGE_CHUNK_THRESHOLD);
    }

    public static FastTikToken fromTiktoken(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        Objects.requireNonNull(mergeableRanks, "mergeableRanks");
        Objects.requireNonNull(specialTokens, "specialTokens");

        BuildArtifacts artifacts = buildArtifacts(mergeableRanks);
        LongLongMap merges = artifacts.merges;
        int[] singleByteTokenId = buildSingleByteTokenMap(mergeableRanks);
        Vocabulary vocabulary =
                VocabularyWithSpecials.create(new VocabularyImpl(mergeableRanks), specialTokens);
        byte[][] tokenBytesById = buildTokenBytesById(vocabulary, SymbolCodec.BYTE_LEVEL);
        boolean exactLookupEnabled =
                Boolean.parseBoolean(System.getProperty(EXACT_LOOKUP_ENABLED_PROPERTY, "true"));
        ExactTokenLookup exactTokenLookup =
                exactLookupEnabled
                        ? ExactTokenLookup.fromMergeableRanks(mergeableRanks)
                        : ExactTokenLookup.EMPTY;
        int tinyThreshold =
                Integer.getInteger(TINY_CHUNK_THRESHOLD_PROPERTY, DEFAULT_TINY_CHUNK_THRESHOLD);
        int threshold =
                Integer.getInteger(LARGE_CHUNK_THRESHOLD_PROPERTY, DEFAULT_LARGE_CHUNK_THRESHOLD);

        return new FastTikToken(
                vocabulary,
                merges,
                singleByteTokenId,
                tokenBytesById,
                exactTokenLookup,
                tinyThreshold,
                threshold);
    }

    @Override
    public float expectedTokensPerChar() {
        return 0.6f;
    }

    @Override
    protected IntSequence encodeImpl(CharSequence text) {
        IntSequence.Builder out = IntSequence.newBuilder(Math.max(8, text.length()));
        Scratch s = acquireScratch();
        try {
            encodeChunkRange(text, 0, text.length(), out, true, s);
        } finally {
            releaseScratch(s);
        }
        return out.build();
    }

    @Override
    protected void encodeImplInto(CharSequence text, IntSequence.Builder out) {
        Scratch s = acquireScratch();
        try {
            encodeChunkRange(text, 0, text.length(), out, true, s);
        } finally {
            releaseScratch(s);
        }
    }

    @Override
    public void encodeInto(
            CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw invalidRange(startInclusive, endExclusive, text.length());
        }
        Scratch s = acquireScratch();
        try {
            out.ensureCapacity(out.size() + Math.max(8, endExclusive - startInclusive));
            encodeChunkRange(text, startInclusive, endExclusive, out, true, s);
        } finally {
            releaseScratch(s);
        }
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        Objects.requireNonNull(text, "text");
        Scratch s = acquireScratch();
        try {
            return countChunkRange(text, startInclusive, endExclusive, s);
        } finally {
            releaseScratch(s);
        }
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
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

        byte[] out = new byte[totalBytes];
        int offset = 0;
        for (int i = 0; i < length; i++) {
            int tokenId = tokens.intAt(i);
            byte[] tokenBytes = tokenBytesById[tokenId];
            System.arraycopy(tokenBytes, 0, out, offset, tokenBytes.length);
            offset += tokenBytes.length;
        }
        return out;
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

    @Override
    public String toString() {
        return "Fast TikToken BPE";
    }

    private int encodeChunkRange(
            CharSequence source,
            int startInclusive,
            int endExclusive,
            IntSequence.Builder out,
            boolean keepOutput,
            Scratch s) {
        if (startInclusive >= endExclusive) {
            return 0;
        }
        int byteLength = utf8EncodeRange(source, startInclusive, endExclusive, s);
        if (byteLength == 0) {
            return 0;
        }
        if (byteLength > tinyChunkThreshold) {
            int exactTokenId = exactTokenLookup.find(s.utf8Bytes, byteLength);
            if (exactTokenId >= 0) {
                if (keepOutput) {
                    out.add(exactTokenId);
                }
                return 1;
            }
        }
        if (byteLength <= tinyChunkThreshold) {
            return keepOutput
                    ? encodeTiny(s.utf8Bytes, byteLength, out)
                    : countTiny(s.utf8Bytes, byteLength);
        }
        if (byteLength < largeChunkThreshold) {
            return keepOutput
                    ? encodeSmall(s.utf8Bytes, byteLength, out, s)
                    : countSmall(s.utf8Bytes, byteLength, s);
        }
        return keepOutput
                ? encodeLarge(s.utf8Bytes, byteLength, out, s)
                : countLarge(s.utf8Bytes, byteLength, s);
    }

    private int countChunkRange(
            CharSequence source, int startInclusive, int endExclusive, Scratch s) {
        return encodeChunkRange(source, startInclusive, endExclusive, null, false, s);
    }

    private int encodeTiny(byte[] bytes, int byteLength, IntSequence.Builder out) {
        if (byteLength == 1) {
            int t0 = singleByteTokenId[bytes[0] & 0xFF];
            out.add(t0);
            return 1;
        }
        if (byteLength == 3) {
            int t0 = singleByteTokenId[bytes[0] & 0xFF];
            int t1 = singleByteTokenId[bytes[1] & 0xFF];
            int t2 = singleByteTokenId[bytes[2] & 0xFF];
            int merge01 = (int) merges.getPair(t0, t1);
            int merge12 = (int) merges.getPair(t1, t2);

            if (merge01 < 0 && merge12 < 0) {
                out.add(t0);
                out.add(t1);
                out.add(t2);
                return 3;
            }

            if (merge12 < 0 || (merge01 >= 0 && merge01 <= merge12)) {
                int merge012 = (int) merges.getPair(merge01, t2);
                if (merge012 >= 0) {
                    out.add(merge012);
                    return 1;
                }
                out.add(merge01);
                out.add(t2);
                return 2;
            }

            int merge012 = (int) merges.getPair(t0, merge12);
            if (merge012 >= 0) {
                out.add(merge012);
                return 1;
            }
            out.add(t0);
            out.add(merge12);
            return 2;
        }
        int t0 = singleByteTokenId[bytes[0] & 0xFF];
        int t1 = singleByteTokenId[bytes[1] & 0xFF];
        int mergedRank = (int) merges.getPair(t0, t1);
        if (mergedRank >= 0) {
            out.add(mergedRank);
            return 1;
        }
        out.add(t0);
        out.add(t1);
        return 2;
    }

    private int countTiny(byte[] bytes, int byteLength) {
        if (byteLength == 1) {
            return 1;
        }
        if (byteLength == 3) {
            int t0 = singleByteTokenId[bytes[0] & 0xFF];
            int t1 = singleByteTokenId[bytes[1] & 0xFF];
            int t2 = singleByteTokenId[bytes[2] & 0xFF];
            int merge01 = (int) merges.getPair(t0, t1);
            int merge12 = (int) merges.getPair(t1, t2);

            if (merge01 < 0 && merge12 < 0) {
                return 3;
            }
            if (merge12 < 0 || (merge01 >= 0 && merge01 <= merge12)) {
                return merges.getPair(merge01, t2) >= 0 ? 1 : 2;
            }
            return merges.getPair(t0, merge12) >= 0 ? 1 : 2;
        }
        int t0 = singleByteTokenId[bytes[0] & 0xFF];
        int t1 = singleByteTokenId[bytes[1] & 0xFF];
        return merges.getPair(t0, t1) >= 0 ? 1 : 2;
    }

    /**
     * Benchmark helper: runs only BPE merge path on raw bytes (no normalization/splitting/UTF-8
     * transcoding).
     */
    public int encodeBytesForBenchmark(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        Scratch s = acquireScratch();
        try {
            int n = bytes.length;
            if (n < largeChunkThreshold) {
                return countSmall(bytes, n, s);
            }
            return countLarge(bytes, n, s);
        } finally {
            releaseScratch(s);
        }
    }

    /** Benchmark helper forcing small merge path. */
    public int encodeBytesSmallForBenchmark(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        Scratch s = acquireScratch();
        try {
            return countSmall(bytes, bytes.length, s);
        } finally {
            releaseScratch(s);
        }
    }

    /** Benchmark helper forcing large merge path. */
    public int encodeBytesLargeForBenchmark(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        Scratch s = acquireScratch();
        try {
            return countLarge(bytes, bytes.length, s);
        } finally {
            releaseScratch(s);
        }
    }

    private Scratch acquireScratch() {
        Scratch s = scratchPool.poll();
        return (s != null) ? s : new Scratch();
    }

    private void releaseScratch(Scratch s) {
        if (s != null) {
            scratchPool.offer(s);
        }
    }

    private int encodeSmall(byte[] bytes, int byteLength, IntSequence.Builder out, Scratch s) {
        return mergeSmall(bytes, byteLength, out, true, s);
    }

    private int countSmall(byte[] bytes, int byteLength, Scratch s) {
        return mergeSmall(bytes, byteLength, null, false, s);
    }

    private int mergeSmall(
            byte[] bytes, int byteLength, IntSequence.Builder out, boolean keepOutput, Scratch s) {
        if (byteLength == 0) {
            return 0;
        }
        s.ensureTokens(byteLength);
        s.ensurePairRanks(byteLength - 1);
        int[] tokens = s.tokens;
        int[] pairRanks = s.pairRanks;
        int size = byteLength;

        int i = 0;
        int unrollLimit = size & ~3;
        for (; i < unrollLimit; i += 4) {
            tokens[i + 3] = singleByteTokenId[bytes[i + 3] & 0xFF];
            tokens[i + 2] = singleByteTokenId[bytes[i + 2] & 0xFF];
            tokens[i + 1] = singleByteTokenId[bytes[i + 1] & 0xFF];
            tokens[i] = singleByteTokenId[bytes[i] & 0xFF];
        }
        for (; i < size; i++) {
            tokens[i] = singleByteTokenId[bytes[i] & 0xFF];
        }

        for (i = 0; i + 1 < size; i++) {
            pairRanks[i] = (int) merges.getPair(tokens[i], tokens[i + 1]);
        }

        while (size >= 2) {
            int bestRank = Integer.MAX_VALUE;
            int bestPos = NO_INDEX;
            int pairCount = size - 1;

            int scan = 0;
            int scanLimit = pairCount & ~3;
            for (; scan < scanLimit; scan += 4) {
                int mergedRank = pairRanks[scan];
                if (mergedRank >= 0 && mergedRank < bestRank) {
                    bestRank = mergedRank;
                    bestPos = scan;
                }
                mergedRank = pairRanks[scan + 1];
                if (mergedRank >= 0 && mergedRank < bestRank) {
                    bestRank = mergedRank;
                    bestPos = scan + 1;
                }
                mergedRank = pairRanks[scan + 2];
                if (mergedRank >= 0 && mergedRank < bestRank) {
                    bestRank = mergedRank;
                    bestPos = scan + 2;
                }
                mergedRank = pairRanks[scan + 3];
                if (mergedRank >= 0 && mergedRank < bestRank) {
                    bestRank = mergedRank;
                    bestPos = scan + 3;
                }
            }
            for (; scan < pairCount; scan++) {
                int mergedRank = pairRanks[scan];
                if (mergedRank < 0) {
                    continue;
                }
                if (mergedRank < bestRank) {
                    bestRank = mergedRank;
                    bestPos = scan;
                }
            }

            if (bestPos == NO_INDEX) {
                break;
            }

            tokens[bestPos] = bestRank;
            int tail = size - bestPos - 2;
            if (tail > 0) {
                System.arraycopy(tokens, bestPos + 2, tokens, bestPos + 1, tail);
                System.arraycopy(pairRanks, bestPos + 1, pairRanks, bestPos, tail);
            }

            size--;
            int newPairCount = size - 1;

            if (bestPos > 0) {
                pairRanks[bestPos - 1] = (int) merges.getPair(tokens[bestPos - 1], tokens[bestPos]);
            }
            if (bestPos < newPairCount) {
                pairRanks[bestPos] = (int) merges.getPair(tokens[bestPos], tokens[bestPos + 1]);
            }
        }

        if (keepOutput) {
            out.ensureCapacity(out.size() + size);
            for (int outIdx = 0; outIdx < size; outIdx++) {
                out.add(tokens[outIdx]);
            }
        }
        return size;
    }

    private int encodeLarge(byte[] bytes, int byteLength, IntSequence.Builder out, Scratch s) {
        return mergeLarge(bytes, byteLength, out, true, s);
    }

    private int countLarge(byte[] bytes, int byteLength, Scratch s) {
        return mergeLarge(bytes, byteLength, null, false, s);
    }

    private int mergeLarge(
            byte[] bytes, int byteLength, IntSequence.Builder out, boolean keepOutput, Scratch s) {
        int n = byteLength;
        s.ensureNodes(n);

        int[] token = s.token;
        int[] prev = s.prev;
        int[] next = s.next;
        int[] edgeStamp = s.edgeStamp;
        int[] edgeRight = s.edgeRight;
        int[] edgeRank = s.edgeRank;

        initLargeNodes(bytes, n, token, prev, next, edgeStamp);

        s.heapSize = 0;
        for (int left = 0; left + 1 < n; left++) {
            refreshEdge(left, token, next, edgeStamp, edgeRight, edgeRank, s);
        }

        int tokenCount = n;
        while (tokenCount > 1 && popMinCandidate(edgeStamp, s)) {
            int left = s.popLeft;
            int mergeId = s.popRank;

            int right = next[left];
            if (right == NO_INDEX) {
                continue;
            }
            int leftPrev = prev[left];
            int rightNext = next[right];

            token[left] = mergeId;
            next[left] = rightNext;
            if (rightNext != NO_INDEX) {
                prev[rightNext] = left;
            }

            // Fully detach the removed right node so stale heap candidates that still reference
            // it are invalidated and cannot merge an unreachable node.
            prev[right] = NO_INDEX;
            next[right] = NO_INDEX;
            edgeStamp[right] = 0;

            tokenCount--;

            if (leftPrev != NO_INDEX) {
                refreshEdge(leftPrev, token, next, edgeStamp, edgeRight, edgeRank, s);
            }
            refreshEdge(left, token, next, edgeStamp, edgeRight, edgeRank, s);
        }

        if (keepOutput) {
            out.ensureCapacity(out.size() + tokenCount);
            for (int idx = 0; idx != NO_INDEX; idx = next[idx]) {
                out.add(token[idx]);
            }
        }

        return tokenCount;
    }

    private void initLargeNodes(
            byte[] bytes, int n, int[] token, int[] prev, int[] next, int[] edgeStamp) {
        int i = 0;
        int unrollLimit = n & ~3;
        for (; i < unrollLimit; i += 4) {
            token[i] = singleByteTokenId[bytes[i] & 0xFF];
            prev[i] = i - 1;
            next[i] = i + 1;
            edgeStamp[i] = 0;

            token[i + 1] = singleByteTokenId[bytes[i + 1] & 0xFF];
            prev[i + 1] = i;
            next[i + 1] = i + 2;
            edgeStamp[i + 1] = 0;

            token[i + 2] = singleByteTokenId[bytes[i + 2] & 0xFF];
            prev[i + 2] = i + 1;
            next[i + 2] = i + 3;
            edgeStamp[i + 2] = 0;

            token[i + 3] = singleByteTokenId[bytes[i + 3] & 0xFF];
            prev[i + 3] = i + 2;
            next[i + 3] = i + 4;
            edgeStamp[i + 3] = 0;
        }
        for (; i < n; i++) {
            token[i] = singleByteTokenId[bytes[i] & 0xFF];
            prev[i] = i - 1;
            next[i] = (i + 1 < n) ? i + 1 : NO_INDEX;
            edgeStamp[i] = 0;
        }
        if (n > 0) {
            next[n - 1] = NO_INDEX;
        }
    }

    private void refreshEdge(
            int left,
            int[] token,
            int[] next,
            int[] edgeStamp,
            int[] edgeRight,
            int[] edgeRank,
            Scratch s) {
        int right = next[left];
        if (right == NO_INDEX) {
            edgeStamp[left] = 0;
            return;
        }
        int mergedRank = (int) merges.getPair(token[left], token[right]);
        if (mergedRank < 0) {
            edgeStamp[left] = 0;
            return;
        }

        if (edgeStamp[left] != 0 && edgeRight[left] == right && edgeRank[left] == mergedRank) {
            return;
        }

        int stamp = s.nextStamp(edgeStamp);
        edgeStamp[left] = stamp;
        edgeRight[left] = right;
        edgeRank[left] = mergedRank;
        heapPush(mergedRank, left, stamp, s);
    }

    private boolean popMinCandidate(int[] edgeStamp, Scratch s) {
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
                continue;
            }

            s.popLeft = left;
            s.popRank = rank;
            return true;
        }
        return false;
    }

    private void heapPush(int rank, int left, int stamp, Scratch s) {
        if (s.heapSize == s.heapRank.length) {
            s.ensureHeapSize(s.heapSize + 1);
        }
        int idx = s.heapSize++;
        long node = (((long) left) << 32) | (stamp & 0xFFFFFFFFL);

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

    private void heapifyDown(Scratch s, int idx) {
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

    private static boolean heapLess(int rankA, long nodeA, int rankB, long nodeB) {
        if (rankA != rankB) {
            return rankA < rankB;
        }
        int leftA = (int) (nodeA >>> 32);
        int leftB = (int) (nodeB >>> 32);
        return leftA < leftB;
    }

    private static int utf8EncodeRange(
            CharSequence source, int startInclusive, int endExclusive, Scratch scratch) {
        int maxLen = (endExclusive - startInclusive) << 2;
        scratch.ensureUtf8(maxLen);
        byte[] dst = scratch.utf8Bytes;

        int dp = 0;
        for (int i = startInclusive; i < endExclusive; i++) {
            char c = source.charAt(i);
            if (c < 0x80) {
                dst[dp++] = (byte) c;
                continue;
            }
            if (c < 0x800) {
                dst[dp++] = (byte) (0xC0 | (c >>> 6));
                dst[dp++] = (byte) (0x80 | (c & 0x3F));
                continue;
            }
            if (Character.isHighSurrogate(c)) {
                if (i + 1 < endExclusive) {
                    char low = source.charAt(i + 1);
                    if (Character.isLowSurrogate(low)) {
                        int cp = Character.toCodePoint(c, low);
                        dst[dp++] = (byte) (0xF0 | (cp >>> 18));
                        dst[dp++] = (byte) (0x80 | ((cp >>> 12) & 0x3F));
                        dst[dp++] = (byte) (0x80 | ((cp >>> 6) & 0x3F));
                        dst[dp++] = (byte) (0x80 | (cp & 0x3F));
                        i++;
                        continue;
                    }
                }
                dst[dp++] = REPLACEMENT_B0;
                dst[dp++] = REPLACEMENT_B1;
                dst[dp++] = REPLACEMENT_B2;
                continue;
            }
            if (Character.isLowSurrogate(c)) {
                dst[dp++] = REPLACEMENT_B0;
                dst[dp++] = REPLACEMENT_B1;
                dst[dp++] = REPLACEMENT_B2;
                continue;
            }
            dst[dp++] = (byte) (0xE0 | (c >>> 12));
            dst[dp++] = (byte) (0x80 | ((c >>> 6) & 0x3F));
            dst[dp++] = (byte) (0x80 | (c & 0x3F));
        }
        return dp;
    }

    private static BuildArtifacts buildArtifacts(Map<String, Integer> mergeableRanks) {
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(mergeableRanks.entrySet());
        entries.sort(Comparator.comparingInt(Map.Entry::getValue));

        int maxId = -1;
        for (Map.Entry<String, Integer> e : entries) {
            if (e.getValue() > maxId) {
                maxId = e.getValue();
            }
        }
        long[] keys = new long[Math.max(16, entries.size())];
        long[] values = new long[keys.length];
        int size = 0;

        for (Map.Entry<String, Integer> entry : entries) {
            String token = entry.getKey();
            int rank = entry.getValue();
            if (token.length() <= 1) {
                continue;
            }
            List<String> parts = bpeSplit(mergeableRanks, token, rank);
            if (parts.size() != 2) {
                continue;
            }
            Integer left = mergeableRanks.get(parts.get(0));
            Integer right = mergeableRanks.get(parts.get(1));
            if (left == null || right == null) {
                continue;
            }

            if (size == keys.length) {
                int newCap = keys.length + (keys.length >>> 1);
                keys = Arrays.copyOf(keys, newCap);
                values = Arrays.copyOf(values, newCap);
            }

            keys[size] = IntPair.of(left, right);
            values[size] = rank;
            size++;
        }

        return new BuildArtifacts(
                new LongLongMap(Arrays.copyOf(keys, size), Arrays.copyOf(values, size)));
    }

    private static final class BuildArtifacts {
        final LongLongMap merges;

        BuildArtifacts(LongLongMap merges) {
            this.merges = merges;
        }
    }

    private static List<String> bpeSplit(
            Map<String, Integer> mergeableRanks, String token, Integer maxRank) {
        List<String> parts = new ArrayList<>(token.length());
        for (int i = 0; i < token.length(); i++) {
            parts.add(String.valueOf(token.charAt(i)));
        }

        while (true) {
            Integer bestPos = null;
            Integer bestRank = null;
            String bestMerged = null;
            for (int i = 0; i + 1 < parts.size(); i++) {
                String merged = parts.get(i) + parts.get(i + 1);
                Integer rank = mergeableRanks.get(merged);
                if (rank != null && (bestRank == null || rank < bestRank)) {
                    bestPos = i;
                    bestRank = rank;
                    bestMerged = merged;
                }
            }

            if (bestRank == null || (maxRank != null && bestRank >= maxRank)) {
                break;
            }

            List<String> newParts = new ArrayList<>(parts.size() - 1);
            newParts.addAll(parts.subList(0, bestPos));
            newParts.add(bestMerged);
            newParts.addAll(parts.subList(bestPos + 2, parts.size()));
            parts = newParts;
        }

        return parts;
    }

    private static int[] buildSingleByteTokenMap(Map<String, Integer> mergeableRanks) {
        int[] map = new int[256];
        Arrays.fill(map, NO_TOKEN);
        for (Map.Entry<String, Integer> entry : mergeableRanks.entrySet()) {
            byte[] bytes = SymbolCodec.BYTE_LEVEL.decodeSymbols(entry.getKey());
            if (bytes.length == 1) {
                map[bytes[0] & 0xFF] = entry.getValue();
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

    private static byte[][] buildTokenBytesById(Vocabulary vocabulary, SymbolCodec symbolCodec) {
        int maxId = -1;
        for (Map.Entry<String, Integer> e : vocabulary) {
            if (e.getValue() > maxId) {
                maxId = e.getValue();
            }
        }
        byte[][] table = new byte[Math.max(0, maxId + 1)][];
        for (Map.Entry<String, Integer> e : vocabulary) {
            table[e.getValue()] = symbolCodec.decodeSymbols(e.getKey());
        }
        return table;
    }

    private static final class Scratch {
        byte[] utf8Bytes = new byte[0];
        int[] tokens = new int[0];
        int[] pairRanks = new int[0];
        int[] token = new int[0];
        int[] prev = new int[0];
        int[] next = new int[0];
        int[] edgeStamp = new int[0];
        int[] edgeRight = new int[0];
        int[] edgeRank = new int[0];
        int stampCounter = 1;

        int[] heapRank = new int[0];
        long[] heapNode = new long[0];
        int heapSize;

        int popLeft;
        int popRank;
        int countAccumulator;

        void ensureUtf8(int needed) {
            if (utf8Bytes.length < needed) {
                utf8Bytes = new byte[grow(utf8Bytes.length, needed)];
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
            if (token.length < needed) {
                int cap = grow(token.length, needed);
                token = new int[cap];
                prev = new int[cap];
                next = new int[cap];
                edgeStamp = new int[cap];
                edgeRight = new int[cap];
                edgeRank = new int[cap];
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

    private static final class ExactTokenLookup {
        private static final ExactTokenLookup EMPTY =
                new ExactTokenLookup(new byte[0][], new int[0], new int[0], 0, 0);

        private final byte[][] keys;
        private final int[] ids;
        private final int[] hashes;
        private final int mask;
        private final int maxTokenBytes;

        private ExactTokenLookup(
                byte[][] keys, int[] ids, int[] hashes, int mask, int maxTokenBytes) {
            this.keys = keys;
            this.ids = ids;
            this.hashes = hashes;
            this.mask = mask;
            this.maxTokenBytes = maxTokenBytes;
        }

        static ExactTokenLookup fromMergeableRanks(Map<String, Integer> mergeableRanks) {
            if (mergeableRanks.isEmpty()) {
                return EMPTY;
            }

            int capacity = 1;
            int needed = Math.max(16, mergeableRanks.size() * 2);
            while (capacity < needed) {
                capacity <<= 1;
            }

            byte[][] keys = new byte[capacity][];
            int[] ids = new int[capacity];
            int[] hashes = new int[capacity];
            Arrays.fill(ids, NO_TOKEN);
            int mask = capacity - 1;
            int maxTokenBytes = 0;

            for (Map.Entry<String, Integer> entry : mergeableRanks.entrySet()) {
                byte[] tokenBytes = SymbolCodec.BYTE_LEVEL.decodeSymbols(entry.getKey());
                if (tokenBytes.length == 0) {
                    continue;
                }
                if (tokenBytes.length > maxTokenBytes) {
                    maxTokenBytes = tokenBytes.length;
                }

                int hash = hashBytes(tokenBytes, tokenBytes.length);
                int slot = hash & mask;
                while (ids[slot] != NO_TOKEN) {
                    byte[] existing = keys[slot];
                    if (hashes[slot] == hash
                            && bytesEqual(existing, tokenBytes, tokenBytes.length)) {
                        break;
                    }
                    slot = (slot + 1) & mask;
                }
                keys[slot] = tokenBytes;
                ids[slot] = entry.getValue();
                hashes[slot] = hash;
            }

            return new ExactTokenLookup(keys, ids, hashes, mask, maxTokenBytes);
        }

        int find(byte[] bytes, int length) {
            if (length <= 0 || length > maxTokenBytes || ids.length == 0) {
                return NO_TOKEN;
            }
            int hash = hashBytes(bytes, length);
            int slot = hash & mask;
            while (ids[slot] != NO_TOKEN) {
                if (hashes[slot] == hash && bytesEqual(keys[slot], bytes, length)) {
                    return ids[slot];
                }
                slot = (slot + 1) & mask;
            }
            return NO_TOKEN;
        }

        private static int hashBytes(byte[] bytes, int length) {
            // Hot path favors short ASCII-ish chunks; specialize tiny lengths to reduce loop
            // overhead.
            switch (length) {
                case 1:
                    return 31 + bytes[0];
                case 2:
                    return ((31 + bytes[0]) * 31) + bytes[1];
                case 3:
                    return (((31 + bytes[0]) * 31) + bytes[1]) * 31 + bytes[2];
                case 4:
                    return ((((31 + bytes[0]) * 31) + bytes[1]) * 31 + bytes[2]) * 31 + bytes[3];
                default:
                    int h = 1;
                    for (int i = 0; i < length; i++) {
                        h = 31 * h + bytes[i];
                    }
                    return h;
            }
        }

        private static boolean bytesEqual(byte[] a, byte[] b, int length) {
            if (a == null || a.length != length) {
                return false;
            }
            for (int i = 0; i < length; i++) {
                if (a[i] != b[i]) {
                    return false;
                }
            }
            return true;
        }
    }
}
