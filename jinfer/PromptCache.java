package com.llama4j;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Prefix cache of KV rows and shortconv state, keyed by the effective ingested token
 * stream (see {@link Llama#buildPrefillTokens}). A radix tree of VARIABLE-LENGTH nodes that
 * split at any divergence point (SGLang MambaRadixCache shape): KV is matchable at token
 * granularity, while shortconv state — fixed-size and updated in place — is resumable
 * through two tiers:
 * <ul>
 *   <li><b>Sparse F32 checkpoints</b> (bit-exact) where a request's frontier actually
 *   stopped: end of prompt (L-1), end of generation, and divergence points discovered by
 *   lookup (the current request re-ingests to the divergence and leaves a checkpoint there,
 *   so the NEXT request sharing that prefix resumes exactly). Splitting a node sets
 *   {@code conv = null} on the prefix (a tombstone: conv state cannot be split).</li>
 *   <li><b>Dense F16 bx rows</b> (near-exact): the conv state after position p is exactly
 *   the last dConv-1 per-layer conv INPUTS bx = b∘x, harvested during normal full-size
 *   ingest chunks (no chunk clamping — cold prefill speed is untouched) and retained per
 *   position. Retention: warm prompts keep EVERY row (resume anywhere, zero re-ingest);
 *   regular traffic keeps stride pairs ({@code llama.promptCacheStride}) plus a dense tail
 *   ({@code llama.promptCacheDenseTail}). Unlike checkpoints, bx rows split cleanly.</li>
 * </ul>
 * The usable resume point is the deeper of the two rules, at most L-1 (logits need a
 * pending token).
 *
 * <p>Single-threaded by design: only the generation worker touches the tree, so one
 * in-flight request means the pinned path ({@link #pinnedDeepest}) can live on the cache
 * itself. All payload bytes (KV rows, conv checkpoints) are opaque blobs owned by a
 * {@link CacheStore}; the in-memory radix index holds only tokens, topology and blob
 * references. Node KV segments are layer-major: per kv layer, K rows for all node tokens,
 * then V rows. SWA layers commit/restore through their ring positions, wrap-split into at
 * most two contiguous runs; commit spans are clamped to {@link #swaStride} so ring rows are
 * still live when copied.
 */
final class PromptCache {

    final Llama.Configuration config;
    final CacheStore store;
    final long[] kvPrefixBytesPerToken; // per kv layer: bytes per token of all PRIOR layers (-1 = no kv)
    final long kvBytesPerToken;
    final int[] convOff;                // per layer float offset into a checkpoint array (-1 = not recurrent)
    final int convFloats;               // total floats per conv checkpoint
    final int hist;                     // conv state rows (dConv - 1): bx rows needed before a resume point
    final int[] bxOff;                  // per layer byte offset within a position-major bx row (-1 = not recurrent)
    final long bxBytesPerToken;         // bytes of one retained bx row (all recurrent layers, F16)
    private final boolean anySwaKv;
    private final int swaStride;                // max uncommitted span when SWA exists (ring liveness)
    final Node root;
    Node pinnedDeepest;                 // deepest node pinned by the in-flight request (root = none)
    Harvest harvest;                    // bx staging for the in-flight request; null = sparse commits
    // single-writer (generation worker); volatile so /props handler-thread reads aren't torn
    volatile int nodeCount;
    volatile long checkpointCount;
    long clock;
    volatile long lookups, hits, hitTokens;
    volatile long denseTokens, denseHits, warmTokens;

    static final class Node {
        Node parent;                    // mutable: splits re-parent
        int[] tokens;                   // length >= 1; mutable: a split keeps the suffix here
        final List<Node> children = new ArrayList<>(2);
        MemorySegment kv;               // store blob, layer-major [K_l0 x len][V_l0 x len][K_l1 x len]...
        MemorySegment conv;             // store blob: shortconv states AFTER tokens[len-1]; null = tombstone
        int[] bxPos;                    // node-relative positions with retained bx rows, ascending; null = none
        MemorySegment bx;               // store blob: one position-major F16 row per bxPos entry
        boolean sticky;                 // warm-prompt node: exempt from LRU eviction
        long lastAccess;
        int refCount;
        Llama.State pinnedState;        // non-null: live state whose KV IS this node's data (no blob copy)
    }

    /** chain = matched nodes root-down (last may be partially matched); matchedPos = total
     *  matched positions; resumePos = deepest usable resume point at most streamLength-1
     *  (0 = cold): a node END with a conv checkpoint (denseResume false, resumeIndex = its
     *  chain index), or ANY position whose hist trailing bx rows are covered (denseResume
     *  true, resumeIndex = chain index of the node containing resumePos-1). */
    record Match(List<Node> chain, int matchedPos, int resumeIndex, int resumePos, boolean denseResume) {}

    PromptCache(Llama.Configuration config, CacheStore store) {
        this.config = config;
        this.store = store;
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
        this.hist = dConv;
        this.bxOff = new int[config.numberOfLayers];
        long bxRow = 0;
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (config.isRecurrentLayer(l) && dConv > 0) {
                bxOff[l] = Math.toIntExact(bxRow);
                bxRow += (long) config.embeddingLength * Float16.BYTES;
            } else {
                bxOff[l] = -1;
            }
        }
        this.bxBytesPerToken = bxRow;
        this.root = new Node();
        this.root.tokens = new int[0];
        this.pinnedDeepest = root;
    }

    /** Store bytes a node's blobs will occupy (pre-flight estimate for {@link #ensureBudget}). */
    private long nodeBytes(int len, boolean hasConv, int bxCount) {
        return len * kvBytesPerToken + (hasConv ? convFloats * (long) Float.BYTES : 0) + bxCount * bxBytesPerToken;
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

    /** Leading tokens of child equal to stream[pos..], capped at {@code cap} positions
     *  (child.tokens[0] is already known to match). */
    private static int matchLen(Node child, int[] stream, int pos, int cap) {
        int limit = Math.min(child.tokens.length, cap);
        int m = 1;
        while (m < limit && child.tokens[m] == stream[pos + m]) m++;
        return m;
    }

    /** Longest token-wise match plus the deepest usable resume point: a conv checkpoint at a
     *  matched node end (bit-exact), upgraded to ANY deeper matched position whose hist
     *  trailing bx rows are covered (near-exact F16), both at most streamLength-1 so a
     *  pending token remains for logits. */
    private Match lookup(int[] stream, int streamLength) {
        lookups++;
        List<Node> chain = new ArrayList<>();
        Node node = root;
        int pos = 0;
        int resumePos = 0, resumeIndex = -1;
        while (pos < streamLength) {
            Node child = childMatching(node, stream[pos]);
            if (child == null) break;
            int m = matchLen(child, stream, pos, streamLength - pos);
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
        Match match = denseUpgrade(chain, pos, streamLength, resumePos);
        if (match == null) {
            match = new Match(chain, pos, resumeIndex, resumePos, false);
        } else {
            denseHits++;
        }
        if (match.resumePos() > 0) {
            hits++;
            hitTokens += match.resumePos();
        }
        return match;
    }

    /** The deepest position p in (sparsePos, min(matchedPos, streamLength-1)] whose hist
     *  trailing bx rows (stream positions p-hist .. p-1; negatives count as covered/zero)
     *  all lie on the matched chain; null when the sparse resume point is at least as deep. */
    private Match denseUpgrade(List<Node> chain, int matchedPos, int streamLength, int sparsePos) {
        if (hist == 0 || bxBytesPerToken == 0) return null;
        int cap = Math.min(matchedPos, streamLength - 1);
        int[] starts = new int[chain.size()];
        int s = 0;
        for (int i = 0; i < chain.size(); i++) {
            starts[i] = s;
            s += chain.get(i).tokens.length;
        }
        for (int i = chain.size() - 1; i >= 0; i--) {
            Node node = chain.get(i);
            if (node.bxPos == null || starts[i] >= cap) continue;
            for (int j = node.bxPos.length - 1; j >= 0; j--) {
                int p = starts[i] + node.bxPos[j] + 1; // resume just after this newest row
                if (p > cap) continue;
                if (p <= sparsePos) return null;       // descending: can't beat sparse anymore
                boolean covered = true; // row p-1 is bxPos[j] itself; check the older ones
                for (int k = 2; k <= hist && covered; k++) {
                    covered = p - k < 0 || bxRow(chain, starts, i, p - k) >= 0;
                }
                if (covered) {
                    return new Match(chain, matchedPos, i, p, true);
                }
            }
        }
        return null;
    }

    /** Locates the retained bx row for absolute stream position q on the chain: packed
     *  {@code (chainIndex << 32) | rowIndex}, or -1 when absent. {@code hint} is a chain
     *  index at or below which q lies. */
    private static long bxRow(List<Node> chain, int[] starts, int hint, int q) {
        for (int i = Math.min(hint, chain.size() - 1); i >= 0; i--) {
            if (starts[i] > q) continue;
            Node node = chain.get(i);
            int row = node.bxPos == null ? -1 : Arrays.binarySearch(node.bxPos, q - starts[i]);
            return row < 0 ? -1 : ((long) i << 32) | row;
        }
        return -1;
    }

    /** Copies the resumable chain prefix's KV rows (up to the resume point) and the resume
     *  conv state — F32 checkpoint, or F16 bx-row decode for a dense resume — into a fresh
     *  state. SWA layers restore only the last window before the resume point, into their
     *  ring positions. */
    private void restore(Match match, Llama.State state) {
        int start = 0;
        for (int i = 0; i <= match.resumeIndex(); i++) {
            Node node = match.chain().get(i);
            store.validate(node.kv);
            int len = node.tokens.length;
            for (int l = 0; l < config.nLayerKvFromStart; l++) {
                if (kvPrefixBytesPerToken[l] < 0) continue;
                long rowBytes = (long) config.kvDim(l) * Float16.BYTES;
                MemorySegment keyDst = ((F16FloatTensor) state.keyCache[l]).memorySegment;
                MemorySegment valueDst = ((F16FloatTensor) state.valueCache[l]).memorySegment;
                int lo = start, hi = Math.min(start + len, match.resumePos());
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
            if (match.denseResume()) {
                restoreConvFromBx(match, state);
            } else {
                MemorySegment conv = match.chain().get(match.resumeIndex()).conv;
                for (int l = 0; l < config.numberOfLayers; l++) {
                    if (convOff[l] < 0) continue;
                    F32FloatTensor dst = (F32FloatTensor) state.shortConvState[l];
                    MemorySegment.copy(conv, (long) convOff[l] * Float.BYTES,
                            dst.memorySegment, 0, dst.size() * (long) Float.BYTES);
                }
            }
        }
    }

    /** Rebuilds the conv state for a dense resume at p from the F16 bx rows at stream
     *  positions p-hist .. p-1 (state row k = the k-th oldest; negatives decode to zeros). */
    private void restoreConvFromBx(Match match, Llama.State state) {
        int dim = config.embeddingLength;
        List<Node> chain = match.chain();
        int[] starts = new int[chain.size()];
        for (int i = 1; i < starts.length; i++) {
            starts[i] = starts[i - 1] + chain.get(i - 1).tokens.length;
        }
        for (int k = 0; k < hist; k++) {
            int q = match.resumePos() - hist + k;
            long packed = q < 0 ? -1 : bxRow(chain, starts, chain.size() - 1, q);
            Node node = packed < 0 ? null : chain.get((int) (packed >>> 32));
            long rowOff = packed < 0 ? 0 : (packed & 0xFFFFFFFFL) * bxBytesPerToken;
            for (int l = 0; l < config.numberOfLayers; l++) {
                if (convOff[l] < 0) continue;
                F32FloatTensor dst = (F32FloatTensor) state.shortConvState[l];
                if (node == null) {
                    dst.fillInPlace(k * dim, dim, 0f);
                    continue;
                }
                long src = rowOff + bxOff[l];
                for (int c = 0; c < dim; c++) {
                    short f16 = node.bx.get(ValueLayout.JAVA_SHORT_UNALIGNED, src + c * 2L);
                    dst.setFloat(k * dim + c, Float.float16ToFloat(f16));
                }
            }
        }
    }

    /** Pins the matched chain for the in-flight request (evictions skip pinned nodes). */
    private void pin(Match match) {
        for (Node node : match.chain()) node.refCount++;
        pinnedDeepest = match.chain().isEmpty() ? root : match.chain().getLast();
    }

    /** Releases the request's pins (root up from the deepest committed/matched node). */
    private void unpinCurrent() {
        for (Node node = pinnedDeepest; node != null && node != root; node = node.parent) node.refCount--;
        pinnedDeepest = root;
    }

    /**
     * Attaches a conv checkpoint to the node ending at position p (splitting the covering node
     * when p falls strictly inside it). The caller guarantees the ingestion frontier is exactly
     * at p — state.shortConvState IS the state after stream[0, p) — and that the path exists.
     * No-op when a checkpoint is already present (repeat divergence).
     */
    private void attachCheckpoint(int[] stream, int p, Llama.State state) {
        if (convFloats == 0 || p <= 0) return;
        Node node = nodeEndingAt(stream, p);
        if (node == root || node.conv != null) return;
        snapshotConv(node, state);
        checkpointCount++;
    }

    private void snapshotConv(Node node, Llama.State state) {
        node.conv = store.allocate(convFloats * (long) Float.BYTES);
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (convOff[l] < 0) continue;
            F32FloatTensor src = (F32FloatTensor) state.shortConvState[l];
            MemorySegment.copy(src.memorySegment, 0,
                    node.conv, (long) convOff[l] * Float.BYTES, src.size() * (long) Float.BYTES);
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
        prefix.kv = store.allocate(k * kvBytesPerToken);
        copyNodeRows(node.kv, len, 0, prefix.kv, k, k);
        MemorySegment suffixKv = store.allocate((long) (len - k) * kvBytesPerToken);
        copyNodeRows(node.kv, len, k, suffixKv, len - k, len - k);
        store.free(node.kv);
        node.kv = suffixKv;
        node.tokens = Arrays.copyOfRange(node.tokens, k, len);
        if (node.bxPos != null) { // bx rows ARE splittable: partition at k, rebase the suffix
            int cut = lowerBound(node.bxPos, node.bxPos.length, k);
            MemorySegment oldBx = node.bx;
            prefix.bxPos = cut > 0 ? Arrays.copyOfRange(node.bxPos, 0, cut) : null;
            if (cut > 0) {
                prefix.bx = store.allocate(cut * bxBytesPerToken);
                MemorySegment.copy(oldBx, 0, prefix.bx, 0, cut * bxBytesPerToken);
            }
            int rem = node.bxPos.length - cut;
            if (rem > 0) {
                int[] rebased = new int[rem];
                for (int i = 0; i < rem; i++) rebased[i] = node.bxPos[cut + i] - k;
                node.bxPos = rebased;
                node.bx = store.allocate(rem * bxBytesPerToken);
                MemorySegment.copy(oldBx, cut * bxBytesPerToken, node.bx, 0, rem * bxBytesPerToken);
            } else {
                node.bxPos = null;
                node.bx = null;
            }
            store.free(oldBx);
        }
        prefix.sticky = node.sticky;
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
     * node ending at {@code to} gets a conv checkpoint when {@code withConv}. Like
     * {@link #pinnedDeepest}, the in-flight request's harvest is cache state: staged bx rows
     * falling inside fresh nodes are consumed here (rows covered by existing nodes are
     * dropped). Returns the deepest node, or null when the budget is exhausted by pinned
     * entries (caller stops caching for this request).
     */
    private Node commitSpan(int[] stream, int from, int to, Llama.State state, boolean withConv) {
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
                int m = matchLen(child, stream, pos, to - pos);
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
            int bxFrom = 0, bxCount = 0;
            if (harvest != null) { // staged rows falling inside this fresh node's span
                bxFrom = lowerBound(harvest.positions, harvest.count, pos);
                bxCount = lowerBound(harvest.positions, harvest.count, to) - bxFrom;
            }
            if (!ensureBudget(nodeBytes(len, conv, bxCount))) {
                return null;
            }
            Node fresh = new Node();
            fresh.parent = node;
            fresh.tokens = Arrays.copyOfRange(stream, pos, to);
            fresh.kv = store.allocate(len * kvBytesPerToken);
            copyStateRows(state, pos, to, fresh, len);
            store.validate(fresh.kv);
            if (conv) snapshotConv(fresh, state);
            if (bxCount > 0) {
                fresh.bxPos = new int[bxCount];
                for (int i = 0; i < bxCount; i++) fresh.bxPos[i] = harvest.positions[bxFrom + i] - pos;
                fresh.bx = store.allocate(bxCount * bxBytesPerToken);
                MemorySegment.copy(harvest.rows, bxFrom * harvest.shortsPerRow,
                        fresh.bx, ValueLayout.JAVA_SHORT_UNALIGNED, 0, bxCount * harvest.shortsPerRow);
                fresh.sticky = harvest.warm;
                denseTokens += bxCount;
            }
            fresh.lastAccess = ++clock;
            fresh.refCount = 1;
            node.children.add(fresh);
            nodeCount++;
            if (conv) checkpointCount++;
            pinnedDeepest = fresh;
            node = fresh;
            pos = to;
        }
        if (withConv && convFloats > 0 && node != root && node.conv == null) {
            snapshotConv(node, state); // walk ended on an existing/split node: checkpoint it
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
        while (store.usedBytes() + needed > RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES) {
            Node victim = findLruLeaf(root, null);
            if (victim == null) {
                return false; // everything pinned: stop caching for this request
            }
            victim.parent.children.remove(victim);
            if (victim.kv != null) store.free(victim.kv);
            victim.pinnedState = null; // release state reference for GC
            nodeCount--;
            if (victim.conv != null) {
                store.free(victim.conv);
                checkpointCount--;
            }
            if (victim.bx != null) {
                store.free(victim.bx);
                denseTokens -= victim.bxPos.length;
            }
        }
        return true;
    }

    private Node findLruLeaf(Node node, Node best) {
        for (Node child : node.children) {
            if (child.children.isEmpty()) {
                if (child.refCount == 0 && !child.sticky
                        && (best == null || child.lastAccess < best.lastAccess)) {
                    best = child;
                }
            } else {
                best = findLruLeaf(child, best);
            }
        }
        return best;
    }

    /** First index in {@code a[0, n)} holding a value >= {@code key}. */
    private static int lowerBound(int[] a, int n, int key) {
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] < key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /**
     * Starts bx harvesting for the in-flight request: the returned observer (installed as
     * {@link Llama.State#convHarvest} around prompt-ingest chunks) F16-encodes the conv
     * inputs of RETAINED stream positions into a staging buffer that {@link #commitSpan}
     * moves into fresh nodes. Retention: warm requests keep every row; regular requests
     * keep the hist trailing rows before each multiple of {@code llama.promptCacheStride}
     * plus the last {@code llama.promptCacheDenseTail} positions of the stream. Null when
     * the model has no conv state.
     */
    private Llama.ConvHarvest beginHarvest(int streamLength, boolean warm) {
        harvest = (hist > 0 && bxBytesPerToken > 0) ? new Harvest(streamLength, warm) : null;
        return harvest;
    }

    private void endHarvest() {
        if (harvest != null && harvest.warm) {
            warmTokens += harvest.streamLength; // idempotent: the second (finally) call no-ops
        }
        harvest = null;
    }

    /** Per-chunk bx staging; chunk changes are detected from the batch's pendingPosition
     *  (== stream position: cached generation always runs the model from position 0). */
    final class Harvest implements Llama.ConvHarvest {
        final int streamLength;
        final boolean warm;
        final int shortsPerRow = Math.toIntExact(bxBytesPerToken / Float16.BYTES);
        int chunkStart = -1;
        int count;                      // retained rows staged for the current chunk
        int[] positions = new int[0];   // their absolute stream positions, ascending
        short[] rows = new short[0];    // position-major F16 rows, recurrent layers concatenated

        Harvest(int streamLength, boolean warm) {
            this.streamLength = streamLength;
            this.warm = warm;
        }

        private boolean retained(int q) {
            if (warm) return true;
            int stride = RuntimeFlags.PROMPT_CACHE_STRIDE;
            if (stride > 0 && q % stride >= stride - hist) return true;
            int tail = RuntimeFlags.PROMPT_CACHE_DENSE_TAIL;
            return tail > 0 && q >= streamLength - 1 - tail;
        }

        private void beginChunk(Llama.BatchState seq) {
            chunkStart = seq.pendingPosition;
            count = 0;
            int end = chunkStart + seq.sequenceLength;
            for (int q = chunkStart; q < end; q++) {
                if (retained(q)) count++;
            }
            if (positions.length < count) {
                positions = new int[count];
                rows = new short[count * shortsPerRow];
            }
            int i = 0;
            for (int q = chunkStart; q < end; q++) {
                if (retained(q)) positions[i++] = q;
            }
        }

        @Override
        public void layer(int layer, Llama.BatchState seq) {
            if (seq.pendingPosition != chunkStart) beginChunk(seq);
            int dim = config.embeddingLength;
            int off = bxOff[layer] / Float16.BYTES;
            for (int i = 0; i < count; i++) {
                int tmp = (positions[i] - chunkStart) * 3 * dim;
                int dst = i * shortsPerRow + off;
                for (int c = 0; c < dim; c++) {
                    // post-scan contract: shortConvScan materialized bx = B∘x over the B rows
                    rows[dst + c] = Float.floatToFloat16(seq.shortConvTmp.getFloat(tmp + c));
                }
            }
        }
    }

    /** One cached generation, plugged into {@link Engine#generate} as a {@link Llama.GenerationHooks}.
     *  Owns the whole cache policy: lookup + restore + pin on resume, checkpoint planning, SWA-clamped
     *  per-chunk commits, and the bx-harvest lifecycle — plus {@link #commitFinal} (the
     *  end-of-generation checkpoint, so multi-turn resume is exact) and {@link #cleanup} (per-request
     *  release). Created via {@link #beginGeneration}; the caller drives Engine.generate with it. */
    final class CacheRun implements Llama.GenerationHooks {
        private final Llama.State state;
        private final int[] cachedOut; // out: resumed-prefix length, for the streaming usage counters
        private final boolean warm;
        private boolean caching = true;
        private int prefillLength;
        private int matchedPos;
        private int committedTo;       // commits cover (matchedPos-or-later, committedTo]
        private final int[] checkpoints = new int[2];
        private int checkpointCount;
        private int[] stream;
        private int frontier;

        private CacheRun(Llama.State state, int[] cachedOut, boolean warm) {
            this.state = state;
            this.cachedOut = cachedOut;
            this.warm = warm;
        }

        @Override
        public int resumePosition(int[] stream, int prefillLength) {
            this.prefillLength = prefillLength;
            Match match = lookup(stream, prefillLength);
            this.matchedPos = match.matchedPos();
            this.committedTo = match.matchedPos();
            int resume = match.resumePos();
            cachedOut[0] = resume;
            restore(match, state);
            pin(match); // even on a cold resume: checkpoints attach into matched nodes
            state.convHarvest = beginHarvest(prefillLength, warm);
            for (int p : new int[]{Math.min(matchedPos, prefillLength - 1), prefillLength - 1}) {
                if (p > resume && p > 0 && (checkpointCount == 0 || checkpoints[checkpointCount - 1] != p)) {
                    checkpoints[checkpointCount++] = p;
                }
            }
            return resume;
        }

        @Override
        public int clampChunk(int position, int chunkLength) {
            int clamp = chunkLength;
            for (int i = 0; i < checkpointCount; i++) {
                if (checkpoints[i] > position) {
                    clamp = Math.min(clamp, checkpoints[i] - position);
                    break;
                }
            }
            if (caching && anySwaKv) {
                clamp = Math.min(clamp, Math.max(matchedPos, committedTo) + swaStride - position);
            }
            return clamp;
        }

        private boolean isCheckpoint(int position) {
            for (int i = 0; i < checkpointCount; i++) {
                if (checkpoints[i] == position) return true;
            }
            return false;
        }

        @Override
        public void afterIngest(int[] stream, int position) {
            this.stream = stream;
            this.frontier = position;
            if (!caching) {
                return;
            }
            if (position <= matchedPos) {
                // re-ingesting cached tokens (resume < divergence): KV is already in the
                // tree, only the conv checkpoint at the divergence is new
                if (isCheckpoint(position)) {
                    attachCheckpoint(stream, position, state);
                }
                return;
            }
            boolean commit = position <= prefillLength                                   // prefill chunk
                    || (anySwaKv && position - committedTo >= swaStride);                // SWA ring pressure
            if (commit) {
                if (commitSpan(stream, committedTo, position, state, isCheckpoint(position)) == null) {
                    caching = false;
                    return;
                }
                committedTo = position;
            }
        }

        @Override
        public void afterPrefill() {
            state.convHarvest = null; // decode chunks are never harvested
            endHarvest();
        }

        /** End-of-generation commit (success path): the frontier sits at the last ingested token
         *  (stop tokens are never ingested), checkpointed so the next turn resumes exactly there. */
        void commitFinal() {
            if (caching && frontier > committedTo) {
                commitSpan(stream, committedTo, frontier, state, true);
            }
        }

        /** Releases per-request cache state; safe on any exit path (error paths may skip afterPrefill). */
        void cleanup() {
            state.convHarvest = null;
            endHarvest();
            unpinCurrent();
        }
    }

    /** Begins a cached generation for {@code state}; the returned hooks drive Engine.generate. */
    CacheRun beginGeneration(Llama.State state, int[] cachedOut, boolean warm) {
        return new CacheRun(state, cachedOut, warm);
    }

    Map<String, Object> stats() {
        return Map.ofEntries(
                Map.entry("enabled", true),
                Map.entry("budget_bytes", RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES),
                Map.entry("used_bytes", store.usedBytes()),
                Map.entry("nodes", nodeCount),
                Map.entry("checkpoints", checkpointCount),
                Map.entry("lookups", lookups),
                Map.entry("hits", hits),
                Map.entry("hit_tokens", hitTokens),
                Map.entry("bx_bytes", denseTokens * bxBytesPerToken),
                Map.entry("dense_tokens", denseTokens),
                Map.entry("dense_hits", denseHits),
                Map.entry("warm_tokens", warmTokens));
    }
}
