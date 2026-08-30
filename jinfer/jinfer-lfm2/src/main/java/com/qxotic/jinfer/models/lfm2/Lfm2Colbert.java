package com.qxotic.jinfer.models.lfm2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.ContextModel;
import com.qxotic.jinfer.Reranker;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
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
 * <p>Buffers: {@link Lfm2#colbertRow} hands out one reused per-state view. Query rows are retained
 * in native scratch; document rows are projected once and dotted against it immediately. No
 * per-token heap arrays sit on the scoring path.
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
        this.model = Objects.requireNonNull(model, "model");
        if (!model.configuration().isColbert() || model.weights().dense2() == null) {
            throw new IllegalArgumentException(
                    "this checkpoint is not LFM2.5-ColBERT (non-causal attention with a dense_2"
                            + " projection)");
        }
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
    public ContextModel<?, ?, Lfm2.State> model() {
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
        return state.exclusively(() -> scoreAll0(state, instruction, query, documents, sink));
    }

    private int scoreAll0(
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
        int outDim = model.configuration().embeddingLengthOut();
        MemoryView<MemorySegment> queryRows = state.embedScratch(model.configuration()).colbertRows;
        double[] best = new double[QUERY_LENGTH];
        int total =
                model.forEachSequence(
                        state,
                        new int[][] {querySeq},
                        (index, rowStart) -> {
                            for (int r = 0; r < querySeq.length; r++) {
                                MemoryView<MemorySegment> row =
                                        model.colbertRow(state, rowStart + r);
                                Convert.copyF32(row, 0, queryRows, (long) r * outDim, outDim);
                            }
                        });
        int[][] docs = new int[documents.size()][];
        for (int i = 0; i < documents.size(); i++) docs[i] = documentSequence(documents.get(i));
        return total
                + model.forEachSequence(
                        state,
                        docs,
                        (index, rowStart) ->
                                sink.accept(maxSim(queryRows, state, rowStart, docs[index], best)));
    }

    /**
     * {@code Σ_q max_d (q·d)} over L2-normalized rows; skiplisted DOCUMENT tokens drop out. Each
     * document row is projected once ({@link Lfm2#colbertRow}) and dotted against every query row
     * directly in native memory.
     */
    private double maxSim(
            MemoryView<MemorySegment> queryRows,
            Lfm2.State state,
            int rowStart,
            int[] docIds,
            double[] best) {
        int outDim = model.configuration().embeddingLengthOut();
        Arrays.fill(best, Double.NEGATIVE_INFINITY);
        for (int d = 0; d < docIds.length; d++) {
            if (skiplist.contains(docIds[d])) continue;
            MemoryView<MemorySegment> docRow = model.colbertRow(state, rowStart + d);
            for (int q = 0; q < QUERY_LENGTH; q++) {
                float dot = Ops.dot(queryRows, (long) q * outDim, docRow, 0, outDim);
                if (dot > best[q]) best[q] = dot;
            }
        }
        double score = 0;
        for (double b : best) score += b;
        return score;
    }

    /** {@code [BOS] [Q]-marker query}, padded or truncated to {@link #QUERY_LENGTH}. */
    private int[] querySequence(String query) {
        return querySequence(tokenizer.encode(query).toArray(), bos, queryMarker, pad);
    }

    static int[] querySequence(int[] ids, int bos, int marker, int pad) {
        int[] seq = new int[QUERY_LENGTH];
        seq[0] = bos;
        seq[1] = marker;
        int count = Math.min(ids.length, QUERY_LENGTH - 2);
        System.arraycopy(ids, 0, seq, 2, count);
        Arrays.fill(seq, 2 + count, seq.length, pad);
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
