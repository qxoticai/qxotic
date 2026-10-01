package com.qxotic.toknroll.vector;

import com.qxotic.toknroll.spi.IdBuffer;
import com.qxotic.toknroll.spi.TiktokenEngine;
import com.qxotic.toknroll.spi.TiktokenTables;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * A tiktoken model behind one regex, encoded fast: the text as UTF-8, cut by {@link BitSplit} a
 * batch of pieces at a time, each piece from the thread's {@link WordCache} or merged.
 *
 * <p>A cache hit and a single byte write their ids straight into the output array, in a loop of
 * their own ({@link WordCache#run}); only a miss leaves it. A miss is a token as it is when the
 * merge loop would give it back whole ({@link ExactLookup}), else merged: by hand up to three
 * bytes, by a scan over flat arrays up to {@link #LARGE} bytes, by a heap beyond.
 */
final class FastEncoder implements TiktokenEngine.Encoder {

    /** Readable bytes past a word, so it can be loaded eight at a time. */
    static final int PADDING = 16;

    /** Piece ends per batch. */
    static final int BATCH = 1024;

    /** Words of this many bytes and more merge with the heap. */
    private static final int LARGE = 96;

    /** A thread keeps its scratch unless a giant word grew it past this many elements. */
    private static final int MAX_RETAINED = 128 * 1024;

    private static final int NO_TOKEN = -1;
    private static final int NO_INDEX = -1;

    private final BitSplit split;
    private final PairMap merges;
    private final int[] singleByte;
    private final ExactLookup exact;
    private final ThreadLocal<Scratch> scratch = ThreadLocal.withInitial(() -> new Scratch(true));

    FastEncoder(TiktokenTables model, BitSplit split) {
        this.split = split;
        this.merges = new PairMap(model);
        this.singleByte = singleByteTokens(model);
        byte[][] tokenBytes = new byte[model.tokenCount()][];
        for (int id = 0; id < tokenBytes.length; id++) {
            tokenBytes[id] = model.tokenBytes(id);
        }
        // Last, since proving which tokens fold runs the merge loop.
        this.exact =
                new ExactLookup(tokenBytes, model.ignoreMerges() ? null : provenFolds(tokenBytes));
    }

    /** The token of each byte value: the lowest id of the one-byte tokens. */
    private static int[] singleByteTokens(TiktokenTables model) {
        int[] map = new int[256];
        Arrays.fill(map, NO_TOKEN);
        for (int id = model.tokenCount() - 1; id >= 0; id--) {
            byte[] bytes = model.tokenBytes(id);
            if (bytes != null && bytes.length == 1) {
                map[bytes[0] & 0xFF] = id;
            }
        }
        for (int b = 0; b < 256; b++) {
            if (map[b] == NO_TOKEN) {
                throw new IllegalArgumentException("no token for the byte " + b);
            }
        }
        return map;
    }

    /**
     * Which tokens the merge loop gives back whole from their own bytes: those a word can be taken
     * for without merging. Proven by merging each once, so the lookup never disagrees with the
     * loop.
     */
    private boolean[] provenFolds(byte[][] tokenBytes) {
        boolean[] folds = new boolean[tokenBytes.length];
        Scratch s = new Scratch(false);
        IdBuffer out = new IdBuffer(64);
        for (int id = 0; id < folds.length; id++) {
            byte[] bytes = tokenBytes[id];
            if (bytes == null || bytes.length == 0) {
                continue;
            }
            out.size(0);
            merge(bytes, 0, bytes.length, out, s);
            folds[id] = out.size() == 1 && out.array()[0] == id;
        }
        return folds;
    }

    @Override
    public void encode(CharSequence text, int start, int end, IdBuffer out) {
        byte[] utf8 = Utf8.encode(text, start, end, PADDING);
        encodeUtf8(utf8, 0, utf8.length - PADDING, out);
    }

    @Override
    public void encodeUtf8(byte[] utf8, int start, int end, IdBuffer out) {
        Thread thread = Thread.currentThread();
        // a virtual thread gets a scratch for the call, without a cache worth keeping
        Scratch s = thread.isVirtual() ? new Scratch(false) : scratch.get();
        encode(utf8, start, end, out, s);
        if (s.kept && s.retained() > MAX_RETAINED) {
            scratch.remove();
        }
    }

    /**
     * The batches of {@code text[start, end)}. Words are read eight bytes at a time, so the pieces
     * ending within {@link #PADDING} bytes of the array's end are encoded from a padded copy.
     */
    private void encode(byte[] text, int start, int end, IdBuffer out, Scratch s) {
        int[] ends = s.pieceEnds;
        WordCache cache = s.kept ? s.wordCache() : null;
        int cursor = start;
        while (cursor < end) {
            int n = split.split(text, cursor, end, ends);
            int inPlace = inPlace(text, ends, n);
            cursor = encodeBatch(text, cursor, ends, inPlace, n, cache, out, s);
            if (inPlace < n) {
                encode(paddedTail(text, cursor, end, s), 0, end - cursor, out, s);
                return;
            }
        }
    }

    /**
     * One batch: the first {@code count} pieces of {@code ends}, from {@code cursor}; returns the
     * cursor after them. A method of its own, called once per batch, so that the JIT compiles it
     * early and on a profile with every path in it: the text-sized loop above runs a handful of
     * times per process, and a compile taken mid-way through it speculated wrong and cost a third
     * of a pass.
     */
    private int encodeBatch(
            byte[] text,
            int cursor,
            int[] ends,
            int count,
            int n,
            WordCache cache,
            IdBuffer out,
            Scratch s) {
        int[] ids = out.reserve(n * WordCache.MAX_IDS);
        int size = out.size();
        int k = 0;
        while (k < count) {
            if (cache != null) {
                long ran = cache.run(text, ends, k, count, cursor, ids, size, singleByte);
                int next = (int) (ran >>> 32);
                size = (int) ran;
                if (next > k) {
                    cursor = ends[next - 1];
                    k = next;
                }
                if (k == count) {
                    break;
                }
            }
            int e = ends[k];
            int length = e - cursor;
            if (length == 1) {
                ids[size++] = singleByte[text[cursor] & 0xFF];
            } else {
                out.size(size);
                WordCache c = cache != null && length <= WordCache.MAX_BYTES ? cache : null;
                encodeMiss(text, cursor, e, out, s, c);
                ids = out.reserve((n - k) * WordCache.MAX_IDS);
                size = out.size();
            }
            cursor = e;
            k++;
        }
        out.size(size);
        return cursor;
    }

    /** How many of the batch's pieces can be read where they are: all but a text's last few. */
    private static int inPlace(byte[] text, int[] ends, int n) {
        int limit = text.length - PADDING;
        int k = n;
        while (k > 0 && ends[k - 1] > limit) {
            k--;
        }
        return k;
    }

    /** {@code text[cursor, end)} in the scratch, followed by readable zero bytes. */
    private static byte[] paddedTail(byte[] text, int cursor, int end, Scratch s) {
        int length = end - cursor;
        byte[] tail = s.tail(length + PADDING);
        System.arraycopy(text, cursor, tail, 0, length);
        Arrays.fill(tail, length, length + PADDING, (byte) 0);
        return tail;
    }

    /**
     * A word of two bytes or more that {@code cache} (if any) was just probed for and does not
     * hold: a token as it is, else merged; then cached when it takes few enough ids.
     */
    private void encodeMiss(
            byte[] text, int start, int end, IdBuffer out, Scratch s, WordCache cache) {
        int first = out.size();
        int token = exact.find(text, start, end - start);
        if (token >= 0) {
            out.add(token);
        } else {
            merge(text, start, end, out, s);
        }
        int count = out.size() - first;
        if (cache != null && count <= WordCache.MAX_IDS) {
            cache.insert(out.array(), first, count);
        }
    }

    /** The merge loop for a word of {@code text[start, end)}, by its length. */
    private void merge(byte[] text, int start, int end, IdBuffer out, Scratch s) {
        int length = end - start;
        if (length <= 3) {
            mergeTiny(text, start, length, out);
        } else if (length < LARGE) {
            mergeSmall(text, start, length, out, s);
        } else {
            mergeLarge(text, start, length, out, s);
        }
    }

    // ---- up to three bytes ----

    private void mergeTiny(byte[] bytes, int off, int length, IdBuffer out) {
        int t0 = singleByte[bytes[off] & 0xFF];
        if (length == 1) {
            out.add(t0);
            return;
        }
        int t1 = singleByte[bytes[off + 1] & 0xFF];
        if (length == 2) {
            long merged = merges.get(t0, t1);
            if (merged != PairMap.NONE) {
                out.add(id(merged));
            } else {
                out.add(t0);
                out.add(t1);
            }
            return;
        }
        int t2 = singleByte[bytes[off + 2] & 0xFF];
        long m01 = merges.get(t0, t1);
        long m12 = merges.get(t1, t2);
        if (m01 == PairMap.NONE && m12 == PairMap.NONE) {
            out.add(t0);
            out.add(t1);
            out.add(t2);
            return;
        }
        if (m12 == PairMap.NONE || (m01 != PairMap.NONE && rank(m01) <= rank(m12))) {
            long m012 = merges.get(id(m01), t2);
            if (m012 != PairMap.NONE) {
                out.add(id(m012));
            } else {
                out.add(id(m01));
                out.add(t2);
            }
            return;
        }
        long m012 = merges.get(t0, id(m12));
        if (m012 != PairMap.NONE) {
            out.add(id(m012));
        } else {
            out.add(t0);
            out.add(id(m12));
        }
    }

    // ---- small words: a scan over flat arrays ----

    private void mergeSmall(byte[] bytes, int off, int length, IdBuffer out, Scratch s) {
        s.ensureSmall(length);
        int[] tokens = s.tokens;
        long[] pairs = s.pairs; // the merge of tokens[i] and tokens[i + 1], or NONE
        int size = length;
        // by four, which the JIT does not do for this loop by itself
        int i = 0;
        for (; i < (size & ~3); i += 4) {
            tokens[i + 3] = singleByte[bytes[off + i + 3] & 0xFF];
            tokens[i + 2] = singleByte[bytes[off + i + 2] & 0xFF];
            tokens[i + 1] = singleByte[bytes[off + i + 1] & 0xFF];
            tokens[i] = singleByte[bytes[off + i] & 0xFF];
        }
        for (; i < size; i++) {
            tokens[i] = singleByte[bytes[off + i] & 0xFF];
        }
        for (i = 0; i + 1 < size; i++) {
            pairs[i] = merges.get(tokens[i], tokens[i + 1]);
        }
        while (size >= 2) {
            // The lowest rank and its position in one long, so the scan has no branch per pair.
            // A pair without a merge scores the maximum.
            long best = Long.MAX_VALUE;
            for (int scan = 0; scan < size - 1; scan++) {
                long rank = (int) pairs[scan];
                best = Math.min(best, rank < 0 ? Long.MAX_VALUE : (rank << 32) | scan);
            }
            if (best == Long.MAX_VALUE) {
                break;
            }
            int at = (int) best;
            tokens[at] = id(pairs[at]);
            int tail = size - at - 2;
            if (tail > 0) {
                System.arraycopy(tokens, at + 2, tokens, at + 1, tail);
                System.arraycopy(pairs, at + 1, pairs, at, tail);
            }
            size--;
            if (at > 0) {
                pairs[at - 1] = merges.get(tokens[at - 1], tokens[at]);
            }
            if (at < size - 1) {
                pairs[at] = merges.get(tokens[at], tokens[at + 1]);
            }
        }
        int[] ids = out.reserve(size);
        System.arraycopy(tokens, 0, ids, out.size(), size);
        out.size(out.size() + size);
    }

    // ---- large words: a linked list and a heap of candidates ----

    private void mergeLarge(byte[] bytes, int off, int n, IdBuffer out, Scratch s) {
        s.ensureLarge(n);
        int[] token = s.token;
        int[] prev = s.prev;
        int[] next = s.next;
        int[] edgeStamp = s.edgeStamp;
        for (int i = 0; i < n; i++) {
            token[i] = singleByte[bytes[off + i] & 0xFF];
            prev[i] = i - 1;
            next[i] = i + 1;
            edgeStamp[i] = 0;
        }
        next[n - 1] = NO_INDEX;

        s.heapSize = 0;
        for (int left = 0; left + 1 < n; left++) {
            refreshEdge(left, s);
        }
        int count = n;
        while (count > 1 && popMinCandidate(s)) {
            int left = s.popLeft;
            int right = next[left];
            if (right == NO_INDEX) {
                continue;
            }
            int leftPrev = prev[left];
            int rightNext = next[right];
            token[left] = s.popMergeId;
            next[left] = rightNext;
            if (rightNext != NO_INDEX) {
                prev[rightNext] = left;
            }
            // Detach the right node, so stale candidates that still name it are passed over.
            prev[right] = NO_INDEX;
            next[right] = NO_INDEX;
            edgeStamp[right] = 0;
            count--;
            if (leftPrev != NO_INDEX) {
                refreshEdge(leftPrev, s);
            }
            refreshEdge(left, s);
        }
        int[] ids = out.reserve(count);
        int at = out.size();
        for (int i = 0; i != NO_INDEX; i = next[i]) {
            ids[at++] = token[i];
        }
        out.size(at);
    }

    /** Pushes the merge of the node at {@code left} and its successor, if it has one. */
    private void refreshEdge(int left, Scratch s) {
        int right = s.next[left];
        if (right == NO_INDEX) {
            s.edgeStamp[left] = 0;
            return;
        }
        long merged = merges.get(s.token[left], s.token[right]);
        if (merged == PairMap.NONE) {
            s.edgeStamp[left] = 0;
            return;
        }
        int rank = rank(merged);
        int mergeId = id(merged);
        if (s.edgeStamp[left] != 0
                && s.edgeRight[left] == right
                && s.edgeRank[left] == rank
                && s.edgeMergeId[left] == mergeId) {
            return;
        }
        int stamp = s.nextStamp();
        s.edgeStamp[left] = stamp;
        s.edgeRight[left] = right;
        s.edgeRank[left] = rank;
        s.edgeMergeId[left] = mergeId;
        heapPush(rank, left, stamp, s);
    }

    /** The live candidate of lowest rank, leftmost first, into {@code s.pop*}; false if none. */
    private static boolean popMinCandidate(Scratch s) {
        while (s.heapSize > 0) {
            long node = s.heapNode[0];
            int left = (int) (node >>> 32);
            int stamp = (int) node;
            int last = --s.heapSize;
            if (last > 0) {
                s.heapRank[0] = s.heapRank[last];
                s.heapNode[0] = s.heapNode[last];
                heapifyDown(s, 0);
            }
            if (s.edgeStamp[left] != stamp) {
                continue;
            }
            s.popLeft = left;
            s.popMergeId = s.edgeMergeId[left];
            return true;
        }
        return false;
    }

    private static void heapPush(int rank, int left, int stamp, Scratch s) {
        s.ensureHeap(s.heapSize + 1);
        int i = s.heapSize++;
        long node = ((long) left << 32) | (stamp & 0xFFFFFFFFL);
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            if (!less(rank, node, s.heapRank[parent], s.heapNode[parent])) {
                break;
            }
            s.heapRank[i] = s.heapRank[parent];
            s.heapNode[i] = s.heapNode[parent];
            i = parent;
        }
        s.heapRank[i] = rank;
        s.heapNode[i] = node;
    }

    private static void heapifyDown(Scratch s, int i) {
        int rank = s.heapRank[i];
        long node = s.heapNode[i];
        int half = s.heapSize >>> 1;
        while (i < half) {
            int child = 2 * i + 1;
            if (child + 1 < s.heapSize
                    && less(
                            s.heapRank[child + 1],
                            s.heapNode[child + 1],
                            s.heapRank[child],
                            s.heapNode[child])) {
                child++;
            }
            if (!less(s.heapRank[child], s.heapNode[child], rank, node)) {
                break;
            }
            s.heapRank[i] = s.heapRank[child];
            s.heapNode[i] = s.heapNode[child];
            i = child;
        }
        s.heapRank[i] = rank;
        s.heapNode[i] = node;
    }

    /** Lower rank first, then the leftmost node. */
    private static boolean less(int rankA, long nodeA, int rankB, long nodeB) {
        return rankA != rankB ? rankA < rankB : (int) (nodeA >>> 32) < (int) (nodeB >>> 32);
    }

    private static int id(long merged) {
        return (int) (merged >>> 32);
    }

    private static int rank(long merged) {
        return (int) merged;
    }

    /** What a thread reuses from call to call. */
    private static final class Scratch {
        /** A scratch a thread keeps is worth a word cache; a one-call scratch gets none. */
        final boolean kept;

        final int[] pieceEnds = new int[BATCH];
        private WordCache wordCache;
        private byte[] tail = new byte[0];
        int[] tokens = new int[0];
        long[] pairs = new long[0];
        int[] token = new int[0];
        int[] prev = new int[0];
        int[] next = new int[0];
        int[] edgeStamp = new int[0];
        int[] edgeRight = new int[0];
        int[] edgeRank = new int[0];
        int[] edgeMergeId = new int[0];
        private int stamp = 1;
        int[] heapRank = new int[0];
        long[] heapNode = new long[0];
        int heapSize;
        int popLeft;
        int popMergeId;

        Scratch(boolean kept) {
            this.kept = kept;
        }

        WordCache wordCache() {
            if (wordCache == null) {
                wordCache = new WordCache();
            }
            return wordCache;
        }

        byte[] tail(int needed) {
            if (tail.length < needed) {
                tail = new byte[grow(tail.length, needed)];
            }
            return tail;
        }

        void ensureSmall(int length) {
            if (tokens.length < length) {
                tokens = new int[grow(tokens.length, length)];
                pairs = new long[tokens.length];
            }
        }

        void ensureLarge(int n) {
            if (token.length < n) {
                int capacity = grow(token.length, n);
                token = new int[capacity];
                prev = new int[capacity];
                next = new int[capacity];
                edgeStamp = new int[capacity];
                edgeRight = new int[capacity];
                edgeRank = new int[capacity];
                edgeMergeId = new int[capacity];
            }
            ensureHeap(Math.max(8, n * 4));
        }

        void ensureHeap(int needed) {
            if (heapRank.length < needed) {
                int capacity = grow(heapRank.length, needed);
                heapRank = Arrays.copyOf(heapRank, capacity);
                heapNode = Arrays.copyOf(heapNode, capacity);
            }
        }

        /** A stamp no live edge has; on wrapping, every edge is cleared first. */
        int nextStamp() {
            if (++stamp == 0) {
                Arrays.fill(edgeStamp, 0);
                stamp = 1;
            }
            return stamp;
        }

        int retained() {
            return Math.max(
                    Math.max(tail.length, tokens.length), Math.max(token.length, heapRank.length));
        }

        private static int grow(int current, int needed) {
            int c = Math.max(8, current);
            while (c < needed) {
                c += c >>> 1;
            }
            return c;
        }
    }

    /**
     * The tokens a word may be taken for whole, found with one probe. A word of up to seven bytes
     * is its own key, packed into a long with its length; a longer one is keyed by a hash and
     * confirmed against the token's bytes. The slot comes from the key's high bits after a
     * multiply.
     */
    private static final class ExactLookup {
        private static final int INLINE_BYTES = 7;
        private static final VarHandle LONGS =
                MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
        private static final long HASHED = Long.MIN_VALUE; // a packed key never has this bit

        // The keys alone in one array, so a probe walks a table half the size and only a hit,
        // one word in ten of those that come here, reads the id beside it.
        private final long[] keys; // 0 is an empty slot
        private final int[] ids;
        private final int shift;
        private final int maxTokenBytes;
        private final byte[][] tokenBytes;

        /** Every token, or with {@code include} only those it says. */
        ExactLookup(byte[][] tokenBytes, boolean[] include) {
            int slots = 16;
            while (slots < tokenBytes.length * 2) {
                slots <<= 1;
            }
            this.shift = Long.numberOfLeadingZeros(slots) + 1;
            this.keys = new long[slots];
            this.ids = new int[slots];
            this.tokenBytes = tokenBytes;
            int maxTokenBytes = 0;
            for (int id = 0; id < tokenBytes.length; id++) {
                byte[] bytes = tokenBytes[id];
                if (bytes == null || bytes.length == 0 || (include != null && !include[id])) {
                    continue;
                }
                maxTokenBytes = Math.max(maxTokenBytes, bytes.length);
                long key = key(Arrays.copyOf(bytes, bytes.length + 8), 0, bytes.length);
                int slot = slot(key, shift);
                while (keys[slot] != 0) {
                    if (keys[slot] == key && Arrays.equals(tokenBytes[ids[slot]], bytes)) {
                        break; // the same bytes twice: the first id stays
                    }
                    slot = (slot + 1) & (slots - 1);
                }
                if (keys[slot] == 0) {
                    keys[slot] = key;
                    ids[slot] = id;
                }
            }
            this.maxTokenBytes = maxTokenBytes;
        }

        int find(byte[] bytes, int off, int length) {
            if (length <= 0 || length > maxTokenBytes) {
                return NO_TOKEN;
            }
            long key = key(bytes, off, length);
            long[] keys = this.keys;
            int mask = keys.length - 1;
            for (int slot = slot(key, shift); ; slot = (slot + 1) & mask) {
                long k = keys[slot];
                if (k == 0) {
                    return NO_TOKEN;
                }
                if (k == key) {
                    int id = ids[slot];
                    if (length <= INLINE_BYTES
                            || Arrays.equals(tokenBytes[id], 0, length, bytes, off, off + length)) {
                        return id;
                    }
                }
            }
        }

        private static long key(byte[] bytes, int off, int length) {
            if (length <= INLINE_BYTES) {
                // One load; the bytes past the word are readable, see PADDING.
                long word = (long) LONGS.get(bytes, off) & ((1L << (8 * length)) - 1);
                return word | ((long) length << 56);
            }
            long h = 0xCBF29CE484222325L ^ length;
            for (int i = 0; i < length; i++) {
                h = (h ^ (bytes[off + i] & 0xFF)) * 0x100000001B3L;
            }
            return h | HASHED;
        }

        private static int slot(long key, int shift) {
            return (int) ((key * 0x9E3779B97F4A7C15L) >>> shift);
        }
    }
}
