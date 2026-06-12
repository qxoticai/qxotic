package com.llama4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Prefix cache of KV rows and shortconv checkpoints, keyed by the effective ingested token
 * stream (see {@link Llama#buildPrefillTokens}). A radix tree of VARIABLE-LENGTH nodes that
 * split at any divergence point (SGLang MambaRadixCache shape): KV is matchable at token
 * granularity, while shortconv state — fixed-size and updated in place — is checkpointed
 * SPARSELY, only where a request's frontier actually stopped: end of prompt (L-1), end of
 * generation, and divergence points discovered by lookup (the current request re-ingests to
 * the divergence and leaves a checkpoint there, so the NEXT request sharing that prefix
 * resumes exactly). Splitting a node sets {@code conv = null} on the prefix (a tombstone:
 * conv state cannot be split); the usable resume point is the deepest matched node end that
 * still has a conv checkpoint, at most L-1 (logits need a pending token).
 *
 * <p>Single-threaded by design: only the generation worker touches the tree (node arenas are
 * confined to it), so one in-flight request means the pinned path ({@link #pinnedDeepest})
 * can live on the cache itself. Node KV segments are layer-major: per kv layer, K rows for
 * all node tokens, then V rows. SWA layers commit/restore through their ring positions,
 * wrap-split into at most two contiguous runs; commit spans are clamped to {@link #swaStride}
 * so ring rows are still live when copied.
 */
final class PromptCache {

    final Llama.Configuration config;
    final long[] kvPrefixBytesPerToken; // per kv layer: bytes per token of all PRIOR layers (-1 = no kv)
    final long kvBytesPerToken;
    final int[] convOff;                // per layer float offset into a checkpoint array (-1 = not recurrent)
    final int convFloats;               // total floats per conv checkpoint
    final boolean anySwaKv;
    final int swaStride;                // max uncommitted span when SWA exists (ring liveness)
    final Node root;
    Node pinnedDeepest;                 // deepest node pinned by the in-flight request (root = none)
    // single-writer (generation worker); volatile so /props handler-thread reads aren't torn
    volatile long usedBytes;
    volatile int nodeCount;
    volatile long checkpointCount;
    long clock;
    volatile long lookups, hits, hitTokens;

    static final class Node {
        Node parent;                    // mutable: splits re-parent
        int[] tokens;                   // length >= 1; mutable: a split keeps the suffix here
        final List<Node> children = new ArrayList<>(2);
        Arena arena;
        MemorySegment kv;               // layer-major [K_l0 x len][V_l0 x len][K_l1 x len]...
        float[] conv;                   // shortconv states AFTER tokens[len-1]; null = tombstone
        long lastAccess;
        int refCount;
    }

    /** chain = matched nodes root-down (last may be partially matched); matchedPos = total
     *  matched positions; resumeIndex/resumePos = deepest chain node END with a conv
     *  checkpoint at most streamLength-1 (resumeIndex -1, resumePos 0 = cold). */
    record Match(List<Node> chain, int matchedPos, int resumeIndex, int resumePos) {}

    PromptCache(Llama.Configuration config) {
        this.config = config;
        boolean swa = false;
        for (int l = 0; l < config.nLayerKvFromStart; l++) {
            if (config.kvDim(l) > 0 && config.isSWA[l]) swa = true;
        }
        this.anySwaKv = swa;
        this.swaStride = swa
                ? Integer.highestOneBit(Math.min(config.slidingWindow, RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH))
                : Integer.MAX_VALUE;
        this.kvPrefixBytesPerToken = new long[config.nLayerKvFromStart];
        long perToken = 0;
        for (int l = 0; l < config.nLayerKvFromStart; l++) {
            int kvDim = config.kvDim(l);
            if (kvDim <= 0) {
                kvPrefixBytesPerToken[l] = -1;
                continue;
            }
            kvPrefixBytesPerToken[l] = perToken;
            perToken += 2L * kvDim * Float16.BYTES; // K row + V row
        }
        this.kvBytesPerToken = perToken;
        this.convOff = new int[config.numberOfLayers];
        int convTotal = 0;
        int dConv = Math.max(config.shortConvLCache - 1, 0);
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (config.isRecurrentLayer(l) && dConv > 0) {
                convOff[l] = convTotal;
                convTotal += dConv * config.embeddingLength;
            } else {
                convOff[l] = -1;
            }
        }
        this.convFloats = convTotal;
        this.root = new Node();
        this.root.tokens = new int[0];
        this.pinnedDeepest = root;
    }

    private long nodeBytes(int len, boolean hasConv) {
        return len * kvBytesPerToken + (hasConv ? convFloats * (long) Float.BYTES : 0) + 4L * len + 128;
    }

    private long kOff(int layer, int nodeLen) {
        return kvPrefixBytesPerToken[layer] * nodeLen;
    }

    private long vOff(int layer, int nodeLen) {
        return kOff(layer, nodeLen) + (long) nodeLen * config.kvDim(layer) * Float16.BYTES;
    }

    private static Node childMatching(Node node, int token) {
        for (Node child : node.children) { // radix invariant: siblings differ in their first token
            if (child.tokens[0] == token) return child;
        }
        return null;
    }

    /** Longest token-wise match plus the deepest usable resume point (conv checkpoint at a
     *  matched node end, at most streamLength-1 so a pending token remains for logits). */
    Match lookup(int[] stream, int streamLength) {
        lookups++;
        List<Node> chain = new ArrayList<>();
        Node node = root;
        int pos = 0;
        int resumePos = 0, resumeIndex = -1;
        while (pos < streamLength) {
            Node child = childMatching(node, stream[pos]);
            if (child == null) break;
            int limit = Math.min(child.tokens.length, streamLength - pos);
            int m = 1;
            while (m < limit && child.tokens[m] == stream[pos + m]) m++;
            child.lastAccess = ++clock;
            chain.add(child);
            pos += m;
            if (m < child.tokens.length) break; // divergence (or stream end) inside this node
            if (child.conv != null && pos <= streamLength - 1) {
                resumePos = pos;
                resumeIndex = chain.size() - 1;
            }
            node = child;
        }
        if (resumePos > 0) {
            hits++;
            hitTokens += resumePos;
        }
        return new Match(chain, pos, resumeIndex, resumePos);
    }

    /** Copies the resumable chain prefix's KV rows and the resume checkpoint's conv states
     *  into a fresh state. SWA layers restore only the last window before the resume point,
     *  into their ring positions. */
    void restore(Match match, Llama.State state) {
        int start = 0;
        for (int i = 0; i <= match.resumeIndex(); i++) {
            Node node = match.chain().get(i);
            int len = node.tokens.length;
            for (int l = 0; l < config.nLayerKvFromStart; l++) {
                if (kvPrefixBytesPerToken[l] < 0) continue;
                long rowBytes = (long) config.kvDim(l) * Float16.BYTES;
                MemorySegment keyDst = ((F16FloatTensor) state.keyCache[l]).memorySegment;
                MemorySegment valueDst = ((F16FloatTensor) state.valueCache[l]).memorySegment;
                int lo = start, hi = start + len;
                if (config.isSWA[l]) lo = Math.max(lo, match.resumePos() - config.slidingWindow);
                int w = config.slidingWindow;
                for (int p = lo; p < hi; ) {
                    int ring = config.kvCacheIndex(l, p);
                    int run = config.isSWA[l] ? Math.min(hi - p, w - ring) : hi - p;
                    MemorySegment.copy(node.kv, kOff(l, len) + (long) (p - start) * rowBytes,
                            keyDst, (long) ring * rowBytes, (long) run * rowBytes);
                    MemorySegment.copy(node.kv, vOff(l, len) + (long) (p - start) * rowBytes,
                            valueDst, (long) ring * rowBytes, (long) run * rowBytes);
                    p += run;
                }
            }
            start += len;
        }
        if (convFloats > 0 && match.resumeIndex() >= 0) {
            float[] conv = match.chain().get(match.resumeIndex()).conv;
            for (int l = 0; l < config.numberOfLayers; l++) {
                if (convOff[l] < 0) continue;
                F32FloatTensor dst = (F32FloatTensor) state.shortConvState[l];
                MemorySegment.copy(conv, convOff[l], dst.memorySegment, ValueLayout.JAVA_FLOAT_UNALIGNED, 0,
                        Math.toIntExact(dst.size()));
            }
        }
    }

    /** Pins the matched chain for the in-flight request (evictions skip pinned nodes). */
    void pin(Match match) {
        for (Node node : match.chain()) node.refCount++;
        pinnedDeepest = match.chain().isEmpty() ? root : match.chain().getLast();
    }

    /** Releases the request's pins (root up from the deepest committed/matched node). */
    void unpinCurrent() {
        for (Node node = pinnedDeepest; node != null && node != root; node = node.parent) node.refCount--;
        pinnedDeepest = root;
    }

    /**
     * Attaches a conv checkpoint to the node ending at position p (splitting the covering node
     * when p falls strictly inside it). The caller guarantees the ingestion frontier is exactly
     * at p — state.shortConvState IS the state after stream[0, p) — and that the path exists.
     * No-op when a checkpoint is already present (repeat divergence).
     */
    void attachCheckpoint(int[] stream, int p, Llama.State state) {
        if (convFloats == 0 || p <= 0) return;
        Node node = nodeEndingAt(stream, p);
        if (node == root || node.conv != null) return;
        snapshotConv(node, state);
        usedBytes += convFloats * (long) Float.BYTES;
        checkpointCount++;
    }

    private void snapshotConv(Node node, Llama.State state) {
        node.conv = new float[convFloats];
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (convOff[l] < 0) continue;
            F32FloatTensor src = (F32FloatTensor) state.shortConvState[l];
            MemorySegment.copy(src.memorySegment, ValueLayout.JAVA_FLOAT_UNALIGNED, 0, node.conv, convOff[l],
                    Math.toIntExact(src.size()));
        }
    }

    /** Walks from root along stream[0, p), splitting the node containing p strictly inside;
     *  returns the node ending exactly at p (root for p == 0). Caller guarantees presence. */
    private Node nodeEndingAt(int[] stream, int p) {
        Node node = root;
        int pos = 0;
        while (pos < p) {
            Node child = childMatching(node, stream[pos]);
            int len = child.tokens.length;
            if (pos + len <= p) {
                pos += len;
                node = child;
            } else {
                node = split(child, p - pos);
                pos = p;
            }
        }
        return node;
    }

    /**
     * Splits node at k (0 &lt; k &lt; len), identity-preserving: the ORIGINAL object becomes the
     * suffix (live references stay valid), a new prefix node takes rows [0, k) with
     * {@code conv = null} (tombstone — conv state cannot be split). The request pin transfers
     * to the prefix when the suffix was the pinned deepest (the suffix leaves the pinned path).
     */
    private Node split(Node node, int k) {
        assert node == pinnedDeepest || node.refCount == 0 : "splitting a foreign pinned node";
        int len = node.tokens.length;
        Node prefix = new Node();
        prefix.parent = node.parent;
        prefix.tokens = Arrays.copyOfRange(node.tokens, 0, k);
        prefix.arena = Arena.ofConfined();
        prefix.kv = prefix.arena.allocate(k * kvBytesPerToken, 64);
        copyNodeRows(node.kv, len, 0, prefix.kv, k, k);
        Arena suffixArena = Arena.ofConfined();
        MemorySegment suffixKv = suffixArena.allocate((long) (len - k) * kvBytesPerToken, 64);
        copyNodeRows(node.kv, len, k, suffixKv, len - k, len - k);
        node.arena.close();
        node.arena = suffixArena;
        node.kv = suffixKv;
        node.tokens = Arrays.copyOfRange(node.tokens, k, len);
        prefix.parent.children.remove(node);
        prefix.parent.children.add(prefix);
        prefix.children.add(node);
        node.parent = prefix;
        prefix.lastAccess = node.lastAccess;
        prefix.refCount = node.refCount;
        if (node == pinnedDeepest) {
            node.refCount--;
            pinnedDeepest = prefix;
        }
        nodeCount++;
        usedBytes += 128; // one extra node of fixed overhead; KV bytes are conserved
        return prefix;
    }

    /** Copies {@code count} per-layer K/V rows starting at src row {@code srcFrom} between two
     *  node segments with different lengths (split helper). */
    private void copyNodeRows(MemorySegment src, int srcLen, int srcFrom, MemorySegment dst, int dstLen, int count) {
        for (int l = 0; l < config.nLayerKvFromStart; l++) {
            if (kvPrefixBytesPerToken[l] < 0) continue;
            long rowBytes = (long) config.kvDim(l) * Float16.BYTES;
            MemorySegment.copy(src, kOff(l, srcLen) + srcFrom * rowBytes, dst, kOff(l, dstLen), count * rowBytes);
            MemorySegment.copy(src, vOff(l, srcLen) + srcFrom * rowBytes, dst, vOff(l, dstLen), count * rowBytes);
        }
    }

    /**
     * Commits stream[from, to) from the live state, extending the request's pinned path.
     * The frontier must be exactly at {@code to} (SWA ring rows still live: to - from is at
     * most {@link #swaStride}); {@code from} must be the pinned-deepest's end position.
     * Existing nodes are deduplicated (descended into, splitting at divergence); the final
     * node ending at {@code to} gets a conv checkpoint when {@code withConv}. Returns the
     * deepest node, or null when the budget is exhausted by pinned entries (caller stops
     * caching for this request).
     */
    Node commitSpan(int[] stream, int from, int to, Llama.State state, boolean withConv) {
        // Walk to `from`, pinning any nodes traversed BELOW the current pinned deepest: they
        // join the request's path (e.g. after a checkpoint split moved the pin up, or when the
        // span continues into content another turn already committed). The pinned path is
        // always a prefix of this walk, so a single passed-flag suffices.
        Node node = root;
        boolean passed = pinnedDeepest == root;
        int pos = 0;
        while (pos < from) {
            Node child = childMatching(node, stream[pos]);
            Node stepped;
            if (pos + child.tokens.length <= from) {
                pos += child.tokens.length;
                stepped = child;
            } else {
                stepped = split(child, from - pos); // split transfers the pin when needed
                pos = from;
            }
            if (passed) {
                descend(stepped);
            } else if (stepped == pinnedDeepest) {
                passed = true;
            }
            node = stepped;
        }
        assert node == pinnedDeepest : "commit must extend the pinned path";
        while (pos < to) {
            Node child = childMatching(node, stream[pos]);
            if (child != null) {
                int limit = Math.min(child.tokens.length, to - pos);
                int m = 1;
                while (m < limit && child.tokens[m] == stream[pos + m]) m++;
                if (m == child.tokens.length) {            // full child on our path: descend
                    descend(child);
                    pos += m;
                    node = child;
                    continue;
                }
                Node prefix = split(child, m);             // divergence or span end inside child
                descend(prefix);
                pos += m;
                node = prefix;
                continue;                                  // suffix no longer matches stream[pos]
            }
            int len = to - pos;
            boolean conv = withConv && convFloats > 0;
            if (!ensureBudget(nodeBytes(len, conv))) {
                return null;
            }
            Node fresh = new Node();
            fresh.parent = node;
            fresh.tokens = Arrays.copyOfRange(stream, pos, to);
            fresh.arena = Arena.ofConfined();
            fresh.kv = fresh.arena.allocate(len * kvBytesPerToken, 64);
            copyStateRows(state, pos, to, fresh, len);
            if (conv) snapshotConv(fresh, state);
            fresh.lastAccess = ++clock;
            fresh.refCount = 1;
            node.children.add(fresh);
            usedBytes += nodeBytes(len, conv);
            nodeCount++;
            if (conv) checkpointCount++;
            pinnedDeepest = fresh;
            node = fresh;
            pos = to;
        }
        if (withConv && convFloats > 0 && node != root && node.conv == null) {
            snapshotConv(node, state); // walk ended on an existing/split node: checkpoint it
            usedBytes += convFloats * (long) Float.BYTES;
            checkpointCount++;
        }
        return node;
    }

    private void descend(Node child) {
        child.lastAccess = ++clock;
        child.refCount++;
        pinnedDeepest = child;
    }

    /** Copies state K/V rows for stream positions [from, to) into a fresh node's segment;
     *  SWA layers read their ring positions in at most two contiguous runs. */
    private void copyStateRows(Llama.State state, int from, int to, Node node, int nodeLen) {
        for (int l = 0; l < config.nLayerKvFromStart; l++) {
            if (kvPrefixBytesPerToken[l] < 0) continue;
            long rowBytes = (long) config.kvDim(l) * Float16.BYTES;
            MemorySegment keySrc = ((F16FloatTensor) state.keyCache[l]).memorySegment;
            MemorySegment valueSrc = ((F16FloatTensor) state.valueCache[l]).memorySegment;
            int w = config.slidingWindow;
            for (int p = from; p < to; ) {
                int ring = config.kvCacheIndex(l, p);
                int run = config.isSWA[l] ? Math.min(to - p, w - ring) : to - p;
                MemorySegment.copy(keySrc, (long) ring * rowBytes,
                        node.kv, kOff(l, nodeLen) + (long) (p - from) * rowBytes, (long) run * rowBytes);
                MemorySegment.copy(valueSrc, (long) ring * rowBytes,
                        node.kv, vOff(l, nodeLen) + (long) (p - from) * rowBytes, (long) run * rowBytes);
                p += run;
            }
        }
    }

    /** Evicts least-recently-used unpinned leaves until {@code needed} more bytes fit. */
    private boolean ensureBudget(long needed) {
        while (usedBytes + needed > RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES) {
            Node victim = findLruLeaf(root, null);
            if (victim == null) {
                return false; // everything pinned: stop caching for this request
            }
            victim.parent.children.remove(victim);
            victim.arena.close();
            usedBytes -= nodeBytes(victim.tokens.length, victim.conv != null);
            nodeCount--;
            if (victim.conv != null) checkpointCount--;
        }
        return true;
    }

    private Node findLruLeaf(Node node, Node best) {
        for (Node child : node.children) {
            if (child.children.isEmpty()) {
                if (child.refCount == 0 && (best == null || child.lastAccess < best.lastAccess)) {
                    best = child;
                }
            } else {
                best = findLruLeaf(child, best);
            }
        }
        return best;
    }

    Map<String, Object> stats() {
        return Map.of(
                "enabled", true,
                "budget_bytes", RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES,
                "used_bytes", usedBytes,
                "nodes", nodeCount,
                "checkpoints", checkpointCount,
                "lookups", lookups,
                "hits", hits,
                "hit_tokens", hitTokens);
    }
}
