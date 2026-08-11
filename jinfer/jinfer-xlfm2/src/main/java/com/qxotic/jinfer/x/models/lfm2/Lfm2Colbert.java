package com.qxotic.jinfer.x.models.lfm2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.Segments;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Model;
import com.qxotic.jinfer.x.boundary.Reranker;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryOperations;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.DoubleConsumer;

/**
 * LFM2.5-ColBERT as a reranker: late interaction. Query and documents are embedded SEPARATELY
 * through the bidirectional forward - one 128-d vector per token via the {@code dense_2} head - and
 * a pair's score is MaxSim: for every query token, the best-matching document token's cosine,
 * summed. No judge prompt, no verdict token, and no frame reuse (structurally impossible here: the
 * forward is bidirectional - frame KV attends to the previous candidate - and short-conv state is a
 * flat trailing window, not addressable by position), so this recipe owns {@link #scoreAll}
 * outright instead of the cross-encoder template.
 *
 * <p>The framing conventions come from the model's sentence-transformers config, which the GGUF
 * does not carry - they are the port's to own, like a judge prompt: {@code "[Q] "}/{@code "[D] "}
 * prefixes, queries PADDED to 32 tokens with the pad token (ColBERT's query expansion - the pad
 * rows participate in MaxSim), documents truncated to 512, and punctuation-token rows dropped from
 * DOCUMENT embeddings (the skiplist; query rows always all count). Scores are unbounded sums (a
 * 32-row query tops out at 32), not [0,1] probabilities - ranking and relative thresholds work,
 * "0.5 means yes" does not.
 *
 * <p>Buffers: {@link Lfm2#colbertRow} hands out a REUSED per-state view, so query rows (retained
 * across all documents) are copied out the jota way - the row's {@code float[]} wrapped by {@code
 * MemoryFactory.ofFloats}, filled by a {@code MemoryOperations.copy} bridge. Document rows are
 * consumed immediately, so they are never copied: each is projected ONCE and dotted against every
 * query row straight from the reused buffer (the loop inversion that lets the old recipe's
 * per-document {@code float[][]} allocation go away).
 */
public final class Lfm2Colbert implements Reranker<Lfm2.State> {

    private static final String QUERY_MARKER = "[Q] ";
    private static final String DOCUMENT_MARKER = "[D] ";
    private static final int QUERY_LENGTH = 32;
    private static final int DOCUMENT_LENGTH = 512;
    private static final String[] SKIPLIST_WORDS = {
        "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<",
        "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
    };

    private final Lfm2 model;
    private final Tokenizer tokenizer;
    private final int bos, pad, queryMarker, documentMarker;
    private final Set<Integer> skiplist;

    public Lfm2Colbert(Lfm2 model, int bos, int pad) {
        this.model = model;
        this.tokenizer = model.tokenizer();
        this.bos = bos;
        this.pad = pad;
        // the markers are ADDED tokens (the vocab's last ids), spelled with their trailing
        // space - scaffold, so they are emitted as trusted ids and only the text after them
        // goes through the plain encode (the two-domain law, and what llama.cpp produces)
        this.queryMarker = markerId(QUERY_MARKER);
        this.documentMarker = markerId(DOCUMENT_MARKER);
        this.skiplist = new HashSet<>();
        for (String word : SKIPLIST_WORDS) {
            for (int id : tokenizer.encode(word).toArray()) skiplist.add(id);
        }
    }

    /** The recipe for the checkpoint behind {@code gguf} - bos/pad read from its metadata. */
    public static Lfm2Colbert fromGguf(Lfm2 model, GGUF gguf) {
        return new Lfm2Colbert(
                model,
                gguf.getValue(int.class, "tokenizer.ggml.bos_token_id"),
                gguf.getValue(int.class, "tokenizer.ggml.padding_token_id"));
    }

    /**
     * The id whose decoded text is {@code marker}. By DECODED text, not vocabulary spelling: the
     * byte-level vocab spells a space as {@code 0}, so a name lookup misses. Scanned from the TOP
     * - added tokens live at the end of the vocabulary.
     */
    private int markerId(String marker) {
        var vocab = tokenizer.vocabulary();
        for (int id = vocab.size() - 1; id >= 0; id--) {
            if (marker.equals(tokenizer.decode(new int[] {id}))) return id;
        }
        throw new IllegalArgumentException(
                "this GGUF's vocabulary has no '"
                        + marker
                        + "' marker token - not a ColBERT"
                        + " checkpoint?");
    }

    @Override
    public String defaultInstruction() {
        return ""; // MaxSim has no instruction slot; a non-blank one is refused in scoreAll
    }

    @Override
    public Model<?, ?, Lfm2.State> model() {
        return model;
    }

    @Override
    public boolean hasInstructionSlot() {
        return false; // MaxSim has no prompt; builders that bind an instruction check this
    }

    @Override
    public int scoreAll(
            Lfm2.State state,
            String instruction,
            String query,
            List<String> documents,
            DoubleConsumer sink) {
        if (instruction != null && !instruction.isBlank()) {
            throw new UnsupportedOperationException(
                    "ColBERT scores by MaxSim over token embeddings and has no instruction slot -"
                            + " drop instruction(...), or use a judge reranker (Qwen3-Reranker)");
        }
        // the query embeds in its own pass (a <=32-token forward, negligible next to the
        // documents), so each document then scores the moment its rows are read - sink order is
        // input order by construction
        int[] querySeq = querySequence(query);
        float[][] queryRows = new float[querySeq.length][];
        int total =
                model.forEachSequence(
                        state,
                        new int[][] {querySeq},
                        (index, rowStart) -> {
                            for (int r = 0; r < queryRows.length; r++) {
                                queryRows[r] = copyRow(state, rowStart + r);
                            }
                        });
        int[][] docs = new int[documents.size()][];
        for (int i = 0; i < documents.size(); i++) docs[i] = documentSequence(documents.get(i));
        return total
                + model.forEachSequence(
                        state,
                        docs,
                        (index, rowStart) ->
                                sink.accept(maxSim(queryRows, state, rowStart, docs[index])));
    }

    private static final MemoryOperations<MemorySegment> NATIVE_OPS =
            NativeMemoryOperations.instance();

    /**
     * One projected+normalized row, copied OUT of the reused per-state buffer. The destination
     * {@code float[]} is wrapped as a heap segment ({@code MemorySegment.ofArray} → {@code
     * MemoryFactory.ofMemorySegment}), so the native domain's own {@code copy} bridges native→heap
     * - the jota idiom, without touching {@code Environment} (its global init requires a native
     * backend runtime the x cone deliberately never boots).
     */
    private float[] copyRow(Lfm2.State state, int row) {
        MemoryView<?> view = model.colbertRow(state, row);
        Views.rawF32(view, "colbertRow"); // requireDense: the F32 + contiguity gate
        int outDim = model.config().embeddingLengthOut();
        float[] dst = new float[outDim];
        MemoryView<MemorySegment> src = Views.castToSegmentBacked(view, "colbertRow");
        Memory<MemorySegment> dstMem = MemoryFactory.ofMemorySegment(MemorySegment.ofArray(dst));
        // the offset is the VIEW's byteOffset within its Memory - NOT Raw.vbase (that one is an
        // absolute address into Segments' reinterpreted global segment, for raw kernel reads)
        NATIVE_OPS.copy(src.memory(), src.byteOffset(), dstMem, 0, (long) outDim * Float.BYTES);
        return dst;
    }

    /**
     * {@code Σ_q max_d (q·d)} over L2-normalized rows; skiplisted DOCUMENT tokens drop out. Each
     * document row is projected ONCE ({@link Lfm2#colbertRow}) and dotted against every query row
     * while it sits in the reused buffer - the loop order that needs no document-side copy.
     */
    private double maxSim(float[][] queryRows, Lfm2.State state, int rowStart, int[] docIds) {
        int outDim = model.config().embeddingLengthOut();
        double[] best = new double[queryRows.length];
        Arrays.fill(best, Double.NEGATIVE_INFINITY);
        for (int d = 0; d < docIds.length; d++) {
            if (skiplist.contains(docIds[d])) continue;
            // rawF32's requireDense is the F32 gate; the dot below reads raw floats
            Views.Raw row = Views.rawF32(model.colbertRow(state, rowStart + d), "colbertRow");
            for (int q = 0; q < queryRows.length; q++) {
                float[] qr = queryRows[q];
                double dot = 0;
                for (int i = 0; i < outDim; i++) {
                    dot +=
                            qr[i]
                                    * Segments.readFloat(
                                            row.vseg(), row.vbase() + (long) i * Float.BYTES);
                }
                if (dot > best[q]) best[q] = dot;
            }
        }
        double score = 0;
        for (double b : best) score += b;
        return score;
    }

    /** {@code [BOS] [Q]-marker query}, padded to {@link #QUERY_LENGTH} (never truncated). */
    private int[] querySequence(String query) {
        int[] ids = tokenizer.encode(query).toArray();
        int n = Math.max(2 + ids.length, QUERY_LENGTH);
        int[] seq = new int[n];
        seq[0] = bos;
        seq[1] = queryMarker;
        System.arraycopy(ids, 0, seq, 2, ids.length);
        Arrays.fill(seq, 2 + ids.length, n, pad);
        return seq;
    }

    /** {@code [BOS] [D]-marker document}, truncated to {@link #DOCUMENT_LENGTH}. */
    private int[] documentSequence(String document) {
        int[] ids = tokenizer.encode(document).toArray();
        int n = Math.min(2 + ids.length, DOCUMENT_LENGTH);
        int[] seq = new int[n];
        seq[0] = bos;
        seq[1] = documentMarker;
        System.arraycopy(ids, 0, seq, 2, n - 2);
        return seq;
    }
}
