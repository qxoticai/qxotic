package com.qxotic.jinfer.x.boundary;

import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * One forward call's worth of work: what to feed ({@link Input}) and which final hidden states to
 * retain ({@link Outputs}). Position-agnostic: a batch is always ingested at the state's cursor
 * ({@link ContextState#position()}), which then advances by {@link #count()}.
 */
public record Batch(Input input, Outputs outputs) {

    public Batch {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(outputs, "outputs");
    }

    /**
     * Which rows' final hidden state to retain for projection. LAST = generation; ALL = scoring.
     */
    public enum Outputs {
        LAST,
        ALL
    }

    /** Token ids, encoder-projected rows, or packed ragged multi-sequence text. */
    public sealed interface Input {
        record Tokens(int[] ids) implements Input {
            public Tokens {
                Objects.requireNonNull(ids, "ids");
            }
        }

        /**
         * Dense FP32 {@code [count, modelDim]} encoder output. A bidirectional block is one atomic
         * attention group. A null content key fingerprints row bits; a non-null key identifies the
         * source content and preprocessing options instead. The rows and their backing memory are
         * borrowed and must remain alive and unchanged through ingestion.
         */
        record Embeddings(MemoryView<?> rows, int count, boolean bidirectional, byte[] contentKey)
                implements Input {
            public Embeddings {
                Objects.requireNonNull(rows, "rows");
                if (rows.dataType() != DataType.FP32)
                    throw new IllegalArgumentException("embedding rows must be FP32");
                if (!rows.shape().isFlat() || rows.shape().rank() != 2)
                    throw new IllegalArgumentException(
                            "embedding rows must have shape [count, dim]");
                if (count <= 0 || rows.shape().flatAt(0) != count || rows.shape().flatAt(1) <= 0)
                    throw new IllegalArgumentException(
                            "embedding count " + count + " does not match shape " + rows.shape());
                if (!rows.isRowMajorContiguous())
                    throw new IllegalArgumentException("embedding rows must be dense row-major");
                if (contentKey != null) {
                    if (contentKey.length == 0)
                        throw new IllegalArgumentException("contentKey must not be empty");
                    contentKey = contentKey.clone();
                }
            }

            public Embeddings(MemoryView<?> rows, int count, boolean bidirectional) {
                this(rows, count, bidirectional, null);
            }

            @Override
            public byte[] contentKey() {
                return contentKey == null ? null : contentKey.clone();
            }
        }

        /**
         * Packed (ragged) multi-sequence text: {@code tokens.ids()} is this chunk's slice of the
         * packed stream; {@code seqLen[j]} is the FULL length of sequence j across the whole packed
         * stream (not just this chunk). Each token attends only within its own sequence, causally,
         * positions restart at 0 per sequence. Used for batched embedding (no padding).
         */
        record Sequences(Tokens tokens, int[] seqLen) implements Input {
            public Sequences {
                Objects.requireNonNull(tokens, "tokens");
                Objects.requireNonNull(seqLen, "seqLen");
            }
        }
    }

    /** Prefill a prompt span, projecting only the last row (the next-token distribution). */
    public static Batch prefill(int[] ids) {
        return new Batch(new Input.Tokens(ids), Outputs.LAST);
    }

    /** Decode one sampled token. */
    public static Batch step(int id) {
        return new Batch(new Input.Tokens(new int[] {id}), Outputs.LAST);
    }

    /** Score a span, retaining every row (e.g. perplexity / speculative verify). */
    public static Batch score(int[] ids) {
        return new Batch(new Input.Tokens(ids), Outputs.ALL);
    }

    /** Ingest a bidirectional block of encoder-projected rows. */
    public static Batch embeddings(MemoryView<?> rows, int count) {
        return embeddings(rows, count, true);
    }

    /** Ingest encoder-projected rows, optionally as one bidirectional attention group. */
    public static Batch embeddings(MemoryView<?> rows, int count, boolean bidirectional) {
        return new Batch(new Input.Embeddings(rows, count, bidirectional), Outputs.LAST);
    }

    /** As {@link #embeddings(MemoryView, int, boolean)} with a source-content cache key. */
    public static Batch embeddings(
            MemoryView<?> rows, int count, boolean bidirectional, byte[] contentKey) {
        return new Batch(
                new Input.Embeddings(rows, count, bidirectional, contentKey), Outputs.LAST);
    }

    /**
     * Pack ragged sequences into one batch (concatenate ids, record per-sequence lengths); retains
     * every row so each sequence's pooled position is addressable. No padding.
     */
    public static Batch pack(int[][] seqs) {
        int total = 0;
        for (int[] s : seqs) total += s.length;
        int[] ids = new int[total];
        int[] seqLen = new int[seqs.length];
        int off = 0;
        for (int j = 0; j < seqs.length; j++) {
            System.arraycopy(seqs[j], 0, ids, off, seqs[j].length);
            off += seqs[j].length;
            seqLen[j] = seqs[j].length;
        }
        return new Batch(new Input.Sequences(new Input.Tokens(ids), seqLen), Outputs.ALL);
    }

    /**
     * Normalizes a batch list for ingestion at {@code batchCapacity} (the old {@code
     * Batch.prepare}, output-identical): adjacent LAST-output token batches merge into the largest
     * legal prefill, oversized ones split at the capacity, anything else passes through unchanged.
     * Embedding blocks stay atomic; an oversized bidirectional block is rejected, while causal
     * embedding blocks pass through for now.
     *
     * <p>Same output, less copying than the old {@code flushRun}: a single already-legal batch
     * passes through with NO new array (ponytail: aliased, not defensively copied - the contract is
     * produce-then-ingest; revisit if a caller ever mutates between prepare and ingest), a single
     * oversized batch slices straight from the source, and only a genuine multi-batch run pays the
     * concat. Equivalence with the old algorithm is property-tested in {@code BatchTest}.
     */
    public static List<Batch> prepare(List<Batch> batches, int batchCapacity) {
        if (batchCapacity <= 0)
            throw new IllegalArgumentException("batchCapacity " + batchCapacity);
        var out = new ArrayList<Batch>(batches.size());
        var run = new ArrayList<Batch>();
        for (Batch b : batches) {
            if (b.input() instanceof Input.Tokens && b.outputs() == Outputs.LAST) {
                run.add(b);
                continue;
            }
            flushRun(run, batchCapacity, out);
            if (b.input() instanceof Input.Embeddings e
                    && e.bidirectional()
                    && e.count() > batchCapacity) {
                throw new IllegalArgumentException(
                        "bidirectional media block of "
                                + e.count()
                                + " rows exceeds batchCapacity "
                                + batchCapacity);
            }
            out.add(b);
        }
        flushRun(run, batchCapacity, out);
        return out;
    }

    private static void flushRun(List<Batch> run, int batchCapacity, List<Batch> out) {
        if (run.isEmpty()) return;
        if (run.size() == 1) {
            // the common case (one prefill): no concat pass - an already-legal batch passes
            // through whole, an oversized one slices straight from the source
            Batch b = run.get(0);
            int[] ids = ((Input.Tokens) b.input()).ids();
            run.clear();
            if (ids.length > 0 && ids.length <= batchCapacity) {
                out.add(b);
                return;
            }
            for (int from = 0; from < ids.length; from += batchCapacity) {
                out.add(
                        prefill(
                                Arrays.copyOfRange(
                                        ids, from, Math.min(from + batchCapacity, ids.length))));
            }
            return;
        }
        // a genuine merge run: the old flushRun verbatim - concat the parts, then slice
        int total = 0;
        for (Batch b : run) total += ((Input.Tokens) b.input()).ids().length;
        int[] ids = new int[total];
        int off = 0;
        for (Batch b : run) {
            int[] part = ((Input.Tokens) b.input()).ids();
            System.arraycopy(part, 0, ids, off, part.length);
            off += part.length;
        }
        run.clear();
        for (int from = 0; from < ids.length; from += batchCapacity) {
            out.add(
                    prefill(
                            Arrays.copyOfRange(
                                    ids, from, Math.min(from + batchCapacity, ids.length))));
        }
    }

    /**
     * The token ids a batch list ingests, flattened in order. Token batches only — the shared
     * currency between chat encoding, cache fingerprints and the test harnesses, so the server and
     * the testkit stay byte-compatible.
     */
    public static int[] tokenIds(List<Batch> batches) {
        int n = 0;
        for (Batch b : batches) n += ((Input.Tokens) b.input()).ids().length;
        int[] ids = new int[n];
        int i = 0;
        for (Batch b : batches) {
            int[] part = ((Input.Tokens) b.input()).ids();
            System.arraycopy(part, 0, ids, i, part.length);
            i += part.length;
        }
        return ids;
    }

    /** Total positions a batch list ingests - {@link #count} summed. */
    public static int positions(List<Batch> batches) {
        int total = 0;
        for (Batch b : batches) total += b.count();
        return total;
    }

    /** Rows this batch ingests. */
    public int count() {
        return switch (input) {
            case Input.Tokens t -> t.ids().length;
            case Input.Embeddings e -> e.count();
            case Input.Sequences s -> s.tokens().ids().length;
        };
    }
}
