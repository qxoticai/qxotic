package com.qxotic.jinfer.models.inkling2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.GatedDeltaNet;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Moe;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.Trace;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Optional;

/**
 * Thinking Machines Inkling 2 ({@code general.architecture = inkling2_loop}): a looped
 * mixture-of-experts decoder, after llama.cpp's {@code inkling2.cpp} and {@code inkling2-loop.cpp}.
 *
 * <p>The layers {@code [loopStart, loopEnd)} run {@code loopCount} times per token. Every pass
 * starts again from the output of the layers before the loop and keeps its own attention and
 * convolution history, so a layer of a later pass is another cache slot over the same weights. A
 * later pass also reads the previous one: low-rank controllers gate a message into the layer's Q,
 * K, V and feed-forward inputs.
 *
 * <p>There is one residual stream per block of {@code blockSize} layers. The first layer of a block
 * shelves the stream it was given and starts the next from its attention output; every layer input
 * is then an attention over the depth, a softmax of learned queries against the shelved streams and
 * the current one.
 *
 * <p>Attention has no rotary positions. A projection of the query onto a learned table biases each
 * score by the distance to its key, and a per-query sink takes the rest of the softmax. Keys and
 * queries are clipped to a maximum norm, values are normalized, and K, V, the attention output and
 * the feed-forward output each pass a causal short convolution whose last inputs are recurrent
 * state. One layer in {@code blockSize} attends to everything, the others through a sliding window.
 *
 * <p>Images enter as the rows of {@link Inkling2Vision}, the tower of the model's mmproj.
 */
public final class Inkling2
        implements LanguageModel<Inkling2.Configuration, Inkling2.Weights, Inkling2.State>,
                Multimodal {
    static final String ARCHITECTURE = "inkling2_loop";

    /** The weightless norm of the depth mix and the controller gates (llama.cpp's fixed 1e-6). */
    private static final float MIX_EPS = 1e-6f;

    /** The controllers of a looped layer's inputs, in tensor order. */
    private static final String[] CONTROLS = {"q", "k", "v", "ffn"};

    private final Configuration configuration;
    private final Tokenizer tokenizer;
    private final Weights weights;
    private final Inkling2Vision vision; // null without the mmproj

    Inkling2(
            Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            Inkling2Vision vision) {
        this.configuration = configuration;
        this.tokenizer = tokenizer;
        this.weights = weights;
        this.vision = vision;
    }

    @Override
    public Configuration configuration() {
        return configuration;
    }

    @Override
    public Weights weights() {
        return weights;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    @SuppressWarnings("unchecked")
    public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
        return modality == Media.Image.class
                ? Optional.ofNullable((MediaProjector<R>) vision)
                : Optional.empty();
    }

    @Override
    public Optional<CheckpointCodec<State>> checkpointCodec() {
        return Optional.of(new Inkling2CheckpointCodec(configuration));
    }

    @Override
    public State newState(
            int contextCapacity, int batchCapacity, MemoryArena<MemorySegment> arena) {
        return new State(configuration, contextCapacity, batchCapacity, arena, false);
    }

    @Override
    public State newState(int contextCapacity, int batchCapacity) {
        MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        try {
            return new State(configuration, contextCapacity, batchCapacity, arena, true);
        } catch (RuntimeException | Error failure) {
            Arenas.close(arena);
            throw failure;
        }
    }

    @Override
    public void ingest(State state, Batch batch) {
        state.exclusively(() -> forward(state, batch));
        Reference.reachabilityFence(this);
    }

    private void forward(State state, Batch batch) {
        int rows = batch.count();
        if (rows <= 0 || rows > state.batchCapacity())
            throw new IllegalArgumentException("invalid Inkling 2 batch size " + rows);
        int start = state.position();
        if (start + rows > state.contextCapacity())
            throw new IllegalArgumentException(
                    "ingest of "
                            + rows
                            + " at "
                            + start
                            + " exceeds contextCapacity "
                            + state.contextCapacity());
        Configuration c = configuration;
        int dim = c.embeddingLength;
        switch (batch.input()) {
            case Batch.Input.Tokens tokens -> {
                for (int token : tokens.ids())
                    if (token < 0 || token >= c.vocabularySize)
                        throw new IllegalArgumentException(
                                "token id outside the vocabulary: " + token);
                Views.checkAlive(weights.tokenEmbedding, "tokenEmbedding");
                Convert.gatherToF32(
                        weights.tokenEmbedding, tokens.ids(), 0, rows, state.residual, 0, dim);
                Norms.rmsnormRowsGgml(
                        state.residual, state.residual, weights.tokenNorm, rows, dim, c.rmsNormEps);
            }
            // projected image rows enter as they are: the embedding norm is the tokens'
            case Batch.Input.Embeddings embeddings -> {
                if (embeddings.bidirectional() || embeddings.rows().shape().flatAt(1) != dim)
                    throw new IllegalArgumentException(
                            "Inkling 2 takes causal embedding rows of width " + dim);
                Convert.copyF32(
                        Views.castToSegmentBacked(embeddings.rows(), "embedding rows"),
                        0,
                        state.residual,
                        0,
                        (long) rows * dim);
            }
            case Batch.Input.Sequences ignored ->
                    throw new UnsupportedOperationException(
                            "Inkling 2 does not support packed sequences");
        }
        forward(state, start, rows);
        state.advance(batch);
    }

    /** The layers over the {@code rows} embeddings in {@code state.residual}. */
    private void forward(State state, int startPos, int rows) {
        Configuration c = configuration;
        long span = (long) rows * c.embeddingLength;
        state.blockCount = 0;
        for (int layer = 0; layer < c.loopStart; layer++)
            layer(state, layer, layer, null, startPos, rows);

        // every pass starts from the prelude; a later one hears the pass before it, through the
        // blocks that pass shelved after its first and through its output
        int preludeBlocks = state.blockCount;
        Convert.copyF32(state.residual, 0, state.prelude, 0, span);
        for (int pass = 0; pass < c.loopCount; pass++) {
            if (pass > 0) {
                for (int i = 0; i < state.bank.length; i++)
                    Convert.copyF32(state.block[preludeBlocks + 1 + i], 0, state.bank[i], 0, span);
                Convert.copyF32(state.residual, 0, state.bankOut, 0, span);
                Convert.copyF32(state.prelude, 0, state.residual, 0, span);
                state.blockCount = preludeBlocks;
            }
            for (int layer = c.loopStart; layer < c.loopEnd; layer++) {
                LoopWeights loop = pass == 0 ? null : weights.loop[layer - c.loopStart];
                if (loop != null)
                    mix(
                            state,
                            state.bank,
                            state.bank.length,
                            state.bankOut,
                            loop.messageQuery,
                            CONTROLS.length,
                            state.message,
                            rows);
                layer(state, c.slot(pass, layer), layer, loop, startPos, rows);
            }
        }
        for (int layer = c.loopEnd; layer < c.numberOfLayers; layer++)
            layer(state, layer, layer, null, startPos, rows);
    }

    /**
     * Layer {@code layer} over the history of cache slot {@code slot}; {@code loop} is null except
     * in the later passes, whose inputs take the messages in {@code state.message}.
     */
    private void layer(State state, int slot, int layer, LoopWeights loop, int startPos, int rows) {
        Configuration c = configuration;
        LayerWeights w = weights.layers[layer];
        int dim = c.embeddingLength, kvDim = c.kvDim();
        boolean swa = c.isSwa[layer];
        MemoryView<MemorySegment>[] in = state.input;

        // Q, K and V read one input in a window layer, unless the messages tell them apart
        boolean split = !swa || loop != null;
        if (w.attentionQuery != null)
            mix(
                    state,
                    state.block,
                    state.blockCount,
                    state.residual,
                    w.attentionQuery,
                    swa ? 1 : 3,
                    in,
                    rows);
        for (int j = split ? 2 : 0; j >= 0; j--) { // downwards: in[0] may feed all three
            MemoryView<MemorySegment> x =
                    w.attentionQuery == null ? state.residual : in[swa ? 0 : j];
            if (loop != null) x = control(state, loop.control[j], x, state.message[j], in[j], rows);
            Norms.rmsnormRowsGgml(in[j], x, w.inputNorm[j], rows, dim, c.rmsNormEps);
        }
        attention(state, slot, w, swa, in[0], in[split ? 1 : 0], in[split ? 2 : 0], startPos, rows);

        // a block's first layer shelves the stream it was given and starts a new one
        if (layer % c.blockSize == 0) {
            MemoryView<MemorySegment> shelf = state.block[state.blockCount++];
            Parallel.forLoop(
                    rows,
                    row ->
                            Norms.rmsnormNoWeight(
                                    shelf,
                                    (long) row * dim,
                                    state.residual,
                                    (long) row * dim,
                                    dim,
                                    MIX_EPS));
            Convert.copyF32(state.branch, 0, state.residual, 0, (long) rows * dim);
        } else {
            Ops.addRows(state.residual, state.branch, rows, dim);
        }

        mix(state, state.block, state.blockCount, state.residual, w.ffnQuery, 1, in, rows);
        if (loop != null) control(state, loop.control[3], in[0], state.message[3], in[0], rows);
        Norms.rmsnormRowsGgml(in[0], in[0], w.ffnNorm, rows, dim, c.rmsNormEps);
        moe(state, w, in[0], rows);
        convolve(state, state.branch, w.ffnConv, null, slot, 2 * kvDim + dim, dim, rows);
        Ops.addRows(state.residual, state.branch, rows, dim);
        if (Trace.ENABLED) Trace.sum("l_out-" + slot, state.residual, rows * dim);
    }

    /**
     * Attention over the depth: each of the {@code queries} outputs is the softmax-weighted sum of
     * the unit-RMS states, the {@code count} shelved {@code blocks} (stored normalized) and {@code
     * current}, scored against its query at {@code 1/sqrt(dim)}.
     */
    private void mix(
            State state,
            MemoryView<MemorySegment>[] blocks,
            int count,
            MemoryView<MemorySegment> current,
            MemoryView<MemorySegment> query,
            int queries,
            MemoryView<MemorySegment>[] out,
            int rows) {
        Parallel.forLoop(
                rows, row -> mixRow(state, blocks, count, current, query, queries, out, row));
    }

    private void mixRow(
            State state,
            MemoryView<MemorySegment>[] blocks,
            int count,
            MemoryView<MemorySegment> current,
            MemoryView<MemorySegment> query,
            int queries,
            MemoryView<MemorySegment>[] out,
            int row) {
        int dim = configuration.embeddingLength;
        long at = (long) row * dim;
        float scale = 1f / (float) Math.sqrt(dim);
        Norms.rmsnormNoWeight(state.scratch, at, current, at, dim, MIX_EPS);
        float[] score = new float[count + 1];
        for (int q = 0; q < queries; q++) {
            float max = Float.NEGATIVE_INFINITY, sum = 0f;
            for (int i = 0; i <= count; i++) {
                score[i] =
                        scale
                                * Ops.dot(
                                        i < count ? blocks[i] : state.scratch,
                                        at,
                                        query,
                                        (long) q * dim,
                                        dim);
                max = Math.max(max, score[i]);
            }
            for (int i = 0; i <= count; i++) sum += score[i] = (float) Math.exp(score[i] - max);
            Ops.fillInPlace(out[q], at, dim, 0f);
            for (int i = 0; i <= count; i++)
                Ops.saxpyInPlace(
                        out[q], at, i < count ? blocks[i] : state.scratch, at, dim, score[i] / sum);
        }
    }

    /**
     * A controller mixes {@code message} into a looped layer input: {@code (1 - forget)*x +
     * gain*message}, both gates a tanh of a low-rank read of the normalized {@code x} and message.
     * Writes {@code out}, which may be {@code x}, and returns it.
     */
    private MemoryView<MemorySegment> control(
            State state,
            ControlWeights w,
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> message,
            MemoryView<MemorySegment> out,
            int rows) {
        int dim = configuration.embeddingLength, rank = configuration.gateRank;
        Parallel.forLoop(
                rows,
                row -> {
                    long at = (long) row * dim;
                    Norms.rmsnormNoWeight(state.controlIn, 2 * at, x, at, dim, MIX_EPS);
                    Norms.rmsnormNoWeight(state.controlIn, 2 * at + dim, message, at, dim, MIX_EPS);
                });
        MatMul.gemm(w.down, state.controlIn, state.controlGate, rows);
        // gain and forget are dim outputs of rank inputs: a vector update per input, not a dot
        // per output
        MemoryView<MemorySegment> gain = state.branch, forget = state.scratch;
        Parallel.forLoop(
                rows,
                row -> {
                    int at = row * dim; // softcap indexes by int
                    Ops.fillInPlace(gain, at, dim, 0f);
                    Ops.fillInPlace(forget, at, dim, 0f);
                    for (int r = 0; r < rank; r++) {
                        float gate =
                                Views.getFloat(state.controlGate, (long) row * rank + r, "gate");
                        Ops.saxpyInPlace(gain, at, w.gain, (long) r * dim, dim, gate);
                        Ops.saxpyInPlace(forget, at, w.forget, (long) r * dim, dim, gate);
                    }
                    Activations.softcap(gain, at, dim, 1f); // tanh, as a soft cap at 1
                    Activations.softcap(forget, at, dim, 1f);
                    Ops.multiplyInPlace(gain, at, message, at, dim);
                    Ops.multiplyInPlace(forget, at, x, at, dim);
                    if (out != x) Convert.copyF32(x, at, out, at, dim);
                    Ops.saxpyInPlace(out, at, forget, at, dim, -1f);
                    Ops.addInPlace(out, at, gain, at, dim);
                });
        return out;
    }

    /**
     * Grouped-query attention with a relative-position bias, leaving its output in {@code
     * state.branch}: clipped Q and K, normalized V, K and V through their convolutions, then scores
     * at {@code 1/sqrt(headSize)} plus the bias of the key's distance, against a per-query sink.
     * The output is gated per lane, projected and convolved, and the chunk's K/V committed.
     */
    private void attention(
            State state,
            int slot,
            LayerWeights w,
            boolean swa,
            MemoryView<MemorySegment> inQ,
            MemoryView<MemorySegment> inK,
            MemoryView<MemorySegment> inV,
            int startPos,
            int rows) {
        Configuration c = configuration;
        int dim = c.embeddingLength, headSize = c.headSize, heads = c.numberOfHeads;
        int kvHeads = c.numberOfKeyValueHeads, kvDim = c.kvDim(), queryDim = c.queryDim();
        int relative = c.relativeLength, extent = c.extent(swa);

        MatMul.gemm(w.query, inQ, state.query, rows);
        MatMul.gemm(w.relative, inQ, state.relative, rows);
        MatMul.gemm(w.sink, inQ, state.sinks, rows);
        MatMul.gemm(w.gate, inQ, state.gate, rows);
        // only the full layers have input-dependent taps
        MemoryView<MemorySegment> taps = swa ? null : state.taps;
        MatMul.gemm(w.key, inK, state.batchK, rows);
        if (taps != null) MatMul.gemm(w.keyTaps, inK, taps, rows);
        convolve(state, state.batchK, w.keyConv, taps, slot, 0, kvDim, rows);
        MatMul.gemm(w.value, inV, state.batchV, rows);
        if (taps != null) MatMul.gemm(w.valueTaps, inV, taps, rows);
        convolve(state, state.batchV, w.valueConv, taps, slot, kvDim, kvDim, rows);

        Parallel.forLoop(
                rows,
                row -> {
                    // the full layers stretch q and r with the logarithm of the position
                    float tau = swa ? 1f : c.logScale(startPos + row);
                    for (int head = 0; head < heads; head++) {
                        clip(
                                state.query,
                                (long) row * queryDim + (long) head * headSize,
                                headSize,
                                w.queryClip,
                                (long) head * headSize,
                                c.qkClip,
                                tau);
                        clip(
                                state.relative,
                                ((long) row * heads + head) * relative,
                                relative,
                                w.relativeClip,
                                (long) head * relative,
                                c.relativeClip,
                                tau);
                    }
                    for (int head = 0; head < kvHeads; head++) {
                        long at = (long) row * kvDim + (long) head * headSize;
                        clip(state.batchK, at, headSize, null, 0, c.qkClip, 1f);
                        Norms.rmsnormNoWeight(
                                state.batchV, at, state.batchV, at, headSize, c.rmsNormEps);
                    }
                });
        // r on the bias table: a row per (query, head) with a bias per distance in reach
        int reach = Math.min(extent, startPos + rows);
        MatMul.gemm(
                w.relativeBias,
                state.relative,
                relative,
                state.bias,
                extent,
                reach,
                rows * heads,
                relative);

        float scale = 1f / (float) Math.sqrt(headSize);
        int kvMul = heads / kvHeads, window = swa ? c.slidingWindow : 0;
        MemoryView<MemorySegment> keys = state.keyCache[slot], values = state.valueCache[slot];
        if (rows > 1)
            FlashAttention.relativePrefill(
                    state.query,
                    state.attentionOut,
                    keys,
                    values,
                    state.batchK,
                    state.batchV,
                    heads,
                    startPos,
                    rows,
                    headSize,
                    kvDim,
                    queryDim,
                    kvDim,
                    kvMul,
                    scale,
                    window,
                    0,
                    state.sinks,
                    state.bias,
                    extent);
        else
            FlashAttention.flashDecode(
                    state.query,
                    state.attentionOut,
                    keys,
                    values,
                    state.batchK,
                    state.batchV,
                    heads,
                    startPos,
                    swa ? Math.max(0, startPos - window + 1) : 0,
                    headSize,
                    kvDim,
                    kvMul,
                    scale,
                    0,
                    state.sinks,
                    state.bias,
                    extent,
                    state.decodeScratch);
        GatedDeltaNet.sigmoidMultiply(state.attentionOut, state.gate, rows, queryDim);
        MatMul.gemm(w.output, state.attentionOut, state.branch, rows);
        convolve(state, state.branch, w.attentionConv, null, slot, 2 * kvDim, dim, rows);

        Convert.f32ToF16Rows(state.batchK, keys, (long) startPos * kvDim, rows, kvDim);
        Convert.f32ToF16Rows(state.batchV, values, (long) startPos * kvDim, rows, kvDim);
    }

    /**
     * Scales {@code n} lanes by {@code scale} (nullable), then down to an L2 norm of at most {@code
     * threshold}, then by {@code tau}.
     */
    private static void clip(
            MemoryView<MemorySegment> x,
            long at,
            int n,
            MemoryView<MemorySegment> scale,
            long scaleAt,
            float threshold,
            float tau) {
        if (scale != null) Ops.multiplyInPlace(x, at, scale, scaleAt, n);
        float norm = (float) Math.sqrt(Norms.sumOfSquares(x, at, n)) + 1e-6f;
        Ops.multiplyInPlace(x, at, n, tau * Math.min(1f, threshold / norm));
    }

    /**
     * Causal short convolution with a residual, in place over {@code rows} rows of {@code channels}
     * lanes: {@code x = (x + conv(x)) / sqrt(convLength + 1)}. The inputs before the chunk are the
     * slot's recurrent state at lane {@code channel}, replaced by the chunk's last ones. {@code
     * taps} (nullable) adds input-dependent taps, one per (row, head, lag) with lag 0 the row
     * itself, and widens the divisor to {@code sqrt(2*convLength + 1)}.
     */
    private void convolve(
            State state,
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> taps,
            int slot,
            int channel,
            int channels,
            int rows) {
        Configuration c = configuration;
        int length = c.convLength, past = length - 1, headSize = c.headSize;
        long history = (long) past * channel, lead = (long) past * channels;
        long span = (long) rows * channels;
        MemoryView<MemorySegment> in = state.convIn, out = state.convOut;
        Convert.copyF32(state.conv[slot], history, in, 0, lead);
        Convert.copyF32(x, 0, in, lead, span);
        Convert.copyF32(in, span, state.conv[slot], history, lead);
        Convolutions.causalDepthwise1d(in, weight, out, past + rows, channels, length);
        if (taps != null) {
            int heads = channels / headSize;
            Parallel.forLoop(
                    rows,
                    row -> {
                        for (int head = 0; head < heads; head++)
                            for (int lag = 0; lag < length; lag++)
                                Ops.saxpyInPlace(
                                        out,
                                        lead + (long) row * channels + (long) head * headSize,
                                        in,
                                        lead
                                                + (long) (row - lag) * channels
                                                + (long) head * headSize,
                                        headSize,
                                        Views.getFloat(
                                                taps,
                                                ((long) row * heads + head) * length + lag,
                                                "taps"));
                    });
        }
        float scale = 1f / (float) Math.sqrt((taps == null ? 1 : 2) * length + 1);
        Parallel.forLoop(
                rows,
                row -> {
                    long at = (long) row * channels;
                    Ops.addInPlace(x, at, out, lead + at, channels);
                    Ops.multiplyInPlace(x, at, channels, scale);
                });
    }

    /**
     * Sigmoid-gated MoE into {@code state.branch}: the selection bias steers the top-k only, the k
     * weights are renormalized and scaled, and each expert is one packed gate|up projection, SiTU,
     * and a down projection.
     */
    private void moe(State state, LayerWeights w, MemoryView<MemorySegment> x, int rows) {
        Configuration c = configuration;
        int experts = c.expertCount, used = c.expertUsedCount, ff = c.expertFeedForwardLength;
        MatMul.gemm(w.router, x, state.moeRouter, rows);
        Ops.mapInPlace(state.moeRouter, 0, rows * experts, Activations::sigmoid);
        Convert.copyF32(state.moeRouter, 0, state.moeSelection, 0, (long) rows * experts);
        Ops.addRowBiasInPlace(state.moeSelection, 0, w.selectionBias, 0, rows, experts);
        Moe.selectTopK(
                state.moeSelection,
                state.moeRouter,
                rows,
                experts,
                used,
                state.moeRowTopE,
                state.moeRowTopP,
                state.moeExpertCounts);
        Moe.normalizeTopP(state.moeRowTopP, rows, used);
        for (int route = 0; route < rows * used; route++)
            state.moeRowTopP[route] *= c.expertWeightsScale;
        Moe.dispatch(
                state.moeRouting,
                rows,
                c.embeddingLength,
                x,
                state.moeGather,
                state.moeExpertOut,
                state.branch,
                null,
                (expert, count, gather, out) -> {
                    MatMul.gemm(w.expertGateUp[expert], gather, state.moeHidden, count);
                    Parallel.forLoop(
                            count,
                            row ->
                                    Activations.situMultiply(
                                            state.moeHidden,
                                            2L * row * ff,
                                            state.moeHidden,
                                            2L * row * ff + ff,
                                            ff,
                                            c.situBeta,
                                            c.situLinearBeta));
                    MatMul.gemm(w.expertDown[expert], state.moeHidden, out, count);
                });
    }

    @Override
    public MemoryView<?> logits(State state, int output) {
        MemoryView<?> result = state.exclusively(() -> projectLogits(state, output));
        Reference.reachabilityFence(this);
        return result;
    }

    private MemoryView<?> projectLogits(State state, int output) {
        if (output < 0 || output >= state.outputCount())
            throw new IllegalArgumentException(
                    "output " + output + " outside [0," + state.outputCount() + ")");
        Configuration c = configuration;
        int dim = c.embeddingLength, vocabulary = c.vocabularySize;
        int row = state.lastBatchSize() - state.outputCount() + output;
        long at = (long) row * dim;
        MemoryView<MemorySegment> x = state.input[0];
        mixRow(
                state,
                state.block,
                state.blockCount,
                state.residual,
                weights.outputQuery,
                1,
                state.input,
                row);
        Norms.rmsnormGgml(x, at, x, at, weights.outputNorm, dim, c.rmsNormEps);
        // the rows past the vocabulary are padding
        MatMul.mm(
                weights.output,
                0,
                dim,
                x,
                at,
                dim,
                state.logits,
                0,
                vocabulary,
                vocabulary,
                1,
                dim);
        Activations.softcap(state.logits, 0, vocabulary, c.logitSoftcap);
        return state.logits;
    }

    // === Configuration ===

    /**
     * The checkpoint's shape. {@code vocabularySize} is the tokenizer's; the embedding and output
     * tensors are padded to {@code paddedVocabularySize} rows. {@code isSwa} marks the
     * sliding-window layers.
     */
    public record Configuration(
            int embeddingLength,
            int numberOfLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int headSize,
            int vocabularySize,
            int paddedVocabularySize,
            int maxContextLength,
            float rmsNormEps,
            int expertCount,
            int expertUsedCount,
            int expertFeedForwardLength,
            float expertWeightsScale,
            float situBeta,
            float situLinearBeta,
            int slidingWindow,
            boolean[] isSwa,
            int convLength,
            int blockSize,
            int relativeLength,
            int relativeExtent,
            float qkClip,
            float relativeClip,
            int logScaleFloor,
            float logScaleAlpha,
            float logitSoftcap,
            int loopStart,
            int loopEnd,
            int loopCount,
            int gateRank)
            implements ContextConfiguration {

        int queryDim() {
            return numberOfHeads * headSize;
        }

        int kvDim() {
            return numberOfKeyValueHeads * headSize;
        }

        /** Cache slots: the layers, then the loop's layers again for each later pass. */
        int slots() {
            return numberOfLayers + (loopCount - 1) * (loopEnd - loopStart);
        }

        /** The cache slot of {@code layer} in loop pass {@code pass}. */
        int slot(int pass, int layer) {
            boolean looped = pass > 0 && layer >= loopStart && layer < loopEnd;
            return looped
                    ? numberOfLayers + (pass - 1) * (loopEnd - loopStart) + layer - loopStart
                    : layer;
        }

        /** Distances the bias table covers; a window layer's table spans its window. */
        int extent(boolean swa) {
            return swa ? slidingWindow : relativeExtent;
        }

        /** {@code 1 + alpha*log(max(1, (position + 1)/floor))}. */
        float logScale(int position) {
            return 1f
                    + logScaleAlpha
                            * (float) Math.log(Math.max(1f, (position + 1f) / logScaleFloor));
        }
    }

    // === Weights ===

    /**
     * The low-rank controller of one input of a looped layer. {@code gain} and {@code forget} are
     * held transposed, {@code [gateRank][embeddingLength]}.
     */
    public record ControlWeights(
            MemoryView<MemorySegment> down,
            MemoryView<MemorySegment> gain,
            MemoryView<MemorySegment> forget) {}

    /** A looped layer's message queries and its Q, K, V and feed-forward controllers. */
    public record LoopWeights(MemoryView<MemorySegment> messageQuery, ControlWeights[] control) {}

    /**
     * {@code inputNorm} holds the Q, K and V input norms, one weight three times for a window
     * layer; {@code keyTaps}/{@code valueTaps} are null there, and {@code attentionQuery} is null
     * for the first layer.
     */
    public record LayerWeights(
            MemoryView<MemorySegment> attentionQuery,
            MemoryView<MemorySegment>[] inputNorm,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> sink,
            MemoryView<MemorySegment> queryClip,
            MemoryView<MemorySegment> relative,
            MemoryView<MemorySegment> relativeBias,
            MemoryView<MemorySegment> relativeClip,
            MemoryView<MemorySegment> keyConv,
            MemoryView<MemorySegment> valueConv,
            MemoryView<MemorySegment> keyTaps,
            MemoryView<MemorySegment> valueTaps,
            MemoryView<MemorySegment> attentionConv,
            MemoryView<MemorySegment> ffnQuery,
            MemoryView<MemorySegment> ffnNorm,
            MemoryView<MemorySegment> router,
            MemoryView<MemorySegment> selectionBias,
            MemoryView<MemorySegment>[] expertGateUp,
            MemoryView<MemorySegment>[] expertDown,
            MemoryView<MemorySegment> ffnConv) {}

    public record Weights(
            MemoryView<MemorySegment> tokenEmbedding,
            MemoryView<MemorySegment> tokenNorm,
            MemoryView<MemorySegment> outputQuery,
            MemoryView<MemorySegment> outputNorm,
            MemoryView<MemorySegment> output,
            LayerWeights[] layers,
            LoopWeights[] loop) {}

    // === State ===

    public static final class State extends ContextState {
        final MemoryView<MemorySegment> residual, branch, scratch, prelude, bankOut, logits;
        final MemoryView<MemorySegment>[] input, message, block, bank;
        final MemoryView<MemorySegment> query, gate, attentionOut, batchK, batchV;
        final MemoryView<MemorySegment> relative, sinks, bias, taps, convIn, convOut;
        final MemoryView<MemorySegment> controlIn, controlGate;
        final MemoryView<MemorySegment> moeRouter, moeSelection, moeGather, moeExpertOut, moeHidden;
        final MemoryView<MemorySegment>[] keyCache, valueCache, conv;
        final FlashAttention.DecodeScratch decodeScratch;
        final int[] moeExpertCounts, moeRowTopE;
        final float[] moeRowTopP;
        final Moe.Routing moeRouting;
        int blockCount;

        State(
                Configuration c,
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
            if (contextCapacity > c.maxContextLength)
                throw new IllegalArgumentException(
                        "contextCapacity "
                                + contextCapacity
                                + " outside (0, "
                                + c.maxContextLength
                                + "], the model's maxContextLength");
            int rows = batchCapacity(), dim = c.embeddingLength, kvDim = c.kvDim();
            int heads = c.numberOfHeads, past = c.convLength - 1;
            residual = Views.allocateF32(memoryArena(), rows, dim);
            branch = Views.allocateF32(memoryArena(), rows, dim);
            scratch = Views.allocateF32(memoryArena(), rows, dim);
            prelude = Views.allocateF32(memoryArena(), rows, dim);
            bankOut = Views.allocateF32(memoryArena(), rows, dim);
            logits = Views.allocateF32(memoryArena(), 1, c.vocabularySize);
            input = rowsOf(3, rows, dim);
            message = rowsOf(CONTROLS.length, rows, dim);
            block = rowsOf((c.numberOfLayers + c.blockSize - 1) / c.blockSize, rows, dim);
            bank = rowsOf((c.loopEnd - c.loopStart) / c.blockSize - 1, rows, dim);
            query = Views.allocateF32(memoryArena(), rows, c.queryDim());
            gate = Views.allocateF32(memoryArena(), rows, c.queryDim());
            attentionOut = Views.allocateF32(memoryArena(), rows, c.queryDim());
            batchK = Views.allocateF32(memoryArena(), rows, kvDim);
            batchV = Views.allocateF32(memoryArena(), rows, kvDim);
            relative = Views.allocateF32(memoryArena(), rows, heads * c.relativeLength);
            sinks = Views.allocateF32(memoryArena(), rows, heads);
            bias =
                    Views.allocateF32(
                            memoryArena(),
                            rows,
                            heads * Math.max(c.slidingWindow, c.relativeExtent));
            taps = Views.allocateF32(memoryArena(), rows, c.numberOfKeyValueHeads * c.convLength);
            convIn = Views.allocateF32(memoryArena(), past + rows, dim);
            convOut = Views.allocateF32(memoryArena(), past + rows, dim);
            controlIn = Views.allocateF32(memoryArena(), rows, 2 * dim);
            controlGate = Views.allocateF32(memoryArena(), rows, c.gateRank);
            moeRouter = Views.allocateF32(memoryArena(), rows, c.expertCount);
            moeSelection = Views.allocateF32(memoryArena(), rows, c.expertCount);
            moeGather = Views.allocateF32(memoryArena(), rows, dim);
            moeExpertOut = Views.allocateF32(memoryArena(), rows, dim);
            moeHidden = Views.allocateF32(memoryArena(), rows, 2 * c.expertFeedForwardLength);
            decodeScratch = new FlashAttention.DecodeScratch(memoryArena());
            moeExpertCounts = new int[c.expertCount];
            moeRowTopE = new int[rows * c.expertUsedCount];
            moeRowTopP = new float[rows * c.expertUsedCount];
            moeRouting =
                    new Moe.Routing(moeRowTopE, moeRowTopP, moeExpertCounts, c.expertUsedCount);
            keyCache = views(c.slots());
            valueCache = views(c.slots());
            conv = views(c.slots());
            for (int slot = 0; slot < c.slots(); slot++) {
                // every slot keeps its whole history, the window layers too: a retained session
                // rewinds by moving the cursor, and a ring would have lost the rows under it
                keyCache[slot] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                valueCache[slot] = Views.allocateF16(memoryArena(), contextCapacity, kvDim);
                // the last inputs of the K, V, attention and feed-forward convolutions
                conv[slot] = Views.allocateF32(memoryArena(), past, 2 * kvDim + 2 * dim);
            }
        }

        private MemoryView<MemorySegment>[] rowsOf(int count, int rows, int width) {
            MemoryView<MemorySegment>[] views = views(count);
            for (int i = 0; i < count; i++)
                views[i] = Views.allocateF32(memoryArena(), rows, width);
            return views;
        }

        @SuppressWarnings("unchecked")
        private static MemoryView<MemorySegment>[] views(int count) {
            return new MemoryView[count];
        }

        /** The KV rows beyond the cursor are masked; the convolutions start from zeros. */
        @Override
        protected void clearHistory() {
            for (MemoryView<MemorySegment> history : conv)
                Ops.fillInPlace(history, 0, Math.toIntExact(history.shape().size()), 0f);
        }

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }
    }

    // === Loading ===

    /**
     * Loads an Inkling 2 GGUF, memory-mapping its weights into {@code arena} (which owns their
     * lifetime). The tokenizer is built from the GGUF metadata.
     *
     * @throws IllegalArgumentException if the file is not an {@code inkling2_loop} model
     */
    public static Inkling2 loadModel(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(channel, gguf, arena, null);
        }
    }

    /**
     * As {@link #loadModel(Path, Arena)} over an already-parsed GGUF header and with a
     * caller-supplied {@code tokenizer}; {@code null} builds one from the GGUF metadata.
     */
    public static Inkling2 loadModel(
            FileChannel channel, GGUF gguf, Arena arena, Tokenizer tokenizer) throws IOException {
        String arch = gguf.getString("general.architecture");
        if (!ARCHITECTURE.equals(arch))
            throw new IllegalArgumentException("unsupported architecture: " + arch);
        if (tokenizer == null)
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        Configuration c = loadConfiguration(gguf, tokenizer.vocabulary().size());
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        Weights weights = loadWeights(tensors, c, MemoryAllocators.ofArena(arena));
        return new Inkling2(c, tokenizer, weights, null);
    }

    /** A model sharing this one's weights with the vision tower of {@code mmproj} attached. */
    public Inkling2 withMedia(Path mmproj, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(mmproj, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, mmproj.toString());
            Map<String, MemoryView<MemorySegment>> tensors =
                    ModelLoader.loadTensors(channel, gguf, arena);
            Inkling2Vision tower = new Inkling2Vision(gguf, tensors, configuration.embeddingLength);
            return new Inkling2(configuration, tokenizer, weights, tower);
        }
    }

    /** Reads what llama.cpp reads, and refuses what it refuses. */
    static Configuration loadConfiguration(GGUF gguf, int vocabularySize) {
        String a = ARCHITECTURE + ".";
        int layers = gguf.getValue(int.class, a + "block_count");
        int dim = gguf.getValue(int.class, a + "embedding_length");
        int heads = uniform(gguf, a + "attention.head_count");
        int headSize = gguf.getValueOrDefault(int.class, a + "attention.key_length", dim / heads);
        boolean[] swa = gguf.getValue(boolean[].class, a + "attention.sliding_window_pattern");
        Configuration c =
                new Configuration(
                        dim,
                        layers,
                        heads,
                        uniform(gguf, a + "attention.head_count_kv"),
                        headSize,
                        vocabularySize,
                        gguf.getValue(int.class, a + "vocab_size"),
                        gguf.getValue(int.class, a + "context_length"),
                        gguf.getValue(float.class, a + "attention.layer_norm_rms_epsilon"),
                        gguf.getValue(int.class, a + "expert_count"),
                        gguf.getValue(int.class, a + "expert_used_count"),
                        uniform(gguf, a + "expert_feed_forward_length"),
                        gguf.getValueOrDefault(float.class, a + "expert_weights_scale", 1f),
                        gguf.getValue(float.class, a + "activation.situ_beta"),
                        gguf.getValue(float.class, a + "activation.situ_linear_beta"),
                        gguf.getValue(int.class, a + "attention.sliding_window"),
                        swa.clone(),
                        gguf.getValue(int.class, a + "shortconv.l_cache"),
                        gguf.getValue(int.class, a + "attn_res.block_size"),
                        gguf.getValue(int.class, a + "attention.relative_length"),
                        gguf.getValue(int.class, a + "attention.relative_extent"),
                        gguf.getValue(float.class, a + "attention.qk_clip"),
                        gguf.getValue(float.class, a + "attention.relative_clip"),
                        gguf.getValue(int.class, a + "attention.log_scale_floor"),
                        gguf.getValue(float.class, a + "attention.log_scale_alpha"),
                        gguf.getValueOrDefault(float.class, a + "final_logit_softcapping", 0f),
                        gguf.getValue(int.class, a + "loop.start"),
                        gguf.getValue(int.class, a + "loop.end"),
                        gguf.getValue(int.class, a + "loop.count"),
                        gguf.getValue(int.class, a + "loop.gate_rank"));
        require(
                dim > 0
                        && layers > 0
                        && heads > 0
                        && c.numberOfKeyValueHeads > 0
                        && heads % c.numberOfKeyValueHeads == 0
                        && headSize > 0
                        && c.maxContextLength > 0
                        && swa.length == layers,
                "invalid core dimensions");
        require(
                gguf.getValueOrDefault(int.class, a + "attention.value_length", headSize)
                        == headSize,
                "key and value widths must both equal the head size");
        require(
                vocabularySize == gguf.getValue(int.class, a + "vocab_size_unpadded")
                        && vocabularySize <= c.paddedVocabularySize,
                "tokenizer vocabulary does not match the model");
        require(
                c.slidingWindow > 0
                        && c.relativeLength > 0
                        && c.relativeExtent > 0
                        && c.logScaleFloor > 0
                        && c.convLength >= 2,
                "invalid window, relative-position or convolution dimensions");
        for (float value :
                new float[] {c.rmsNormEps, c.qkClip, c.relativeClip, c.situBeta, c.situLinearBeta})
            require(
                    Float.isFinite(value) && value > 0f,
                    "normalization, clipping and activation parameters must be finite and"
                            + " positive");
        require(
                Float.isFinite(c.logScaleAlpha) && c.logScaleAlpha >= 0f,
                "invalid logarithmic position scale");
        require(
                c.expertCount > 0
                        && c.expertUsedCount > 0
                        && c.expertUsedCount <= c.expertCount
                        && c.expertFeedForwardLength > 0,
                "invalid MoE metadata");
        require(
                c.blockSize > 0
                        && c.loopCount >= 2
                        && c.gateRank > 0
                        && c.loopStart > 0
                        && c.loopStart < c.loopEnd
                        && c.loopEnd <= layers
                        && c.loopStart % c.blockSize == 0
                        && c.loopEnd % c.blockSize == 0,
                "invalid loop [" + c.loopStart + ", " + c.loopEnd + ") x " + c.loopCount);
        return c;
    }

    /** A per-layer metadata array whose layers all agree, or the scalar it may be instead. */
    private static int uniform(GGUF gguf, String key) {
        Object value = gguf.getValue(Object.class, key);
        if (!(value instanceof int[] perLayer)) return ((Number) value).intValue();
        for (int layer : perLayer)
            require(layer == perLayer[0], key + " must be the same for every layer");
        return perLayer[0];
    }

    static Weights loadWeights(
            Map<String, MemoryView<MemorySegment>> tensors,
            Configuration c,
            MemoryArena<MemorySegment> memory) {
        int dim = c.embeddingLength, queryDim = c.queryDim(), kvDim = c.kvDim();
        int heads = c.numberOfHeads, kvHeads = c.numberOfKeyValueHeads, conv = c.convLength;
        int experts = c.expertCount, ff = c.expertFeedForwardLength, rank = c.gateRank;
        LayerWeights[] layers = new LayerWeights[c.numberOfLayers];
        for (int layer = 0; layer < layers.length; layer++) {
            String p = "blk." + layer + ".";
            boolean swa = c.isSwa[layer];
            MemoryView<MemorySegment>[] inputNorm = State.views(3);
            String[] norms = {"attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"};
            for (int j = 0; j < 3; j++)
                inputNorm[j] = f32(tensors, p + (swa ? "attn_norm" : norms[j]) + ".weight", dim);
            layers[layer] =
                    new LayerWeights(
                            layer == 0
                                    ? null
                                    : f32(
                                            tensors,
                                            p + "attn_res_query.weight",
                                            swa ? dim : 3L * dim),
                            inputNorm,
                            weight(tensors, p + "attn_q.weight", queryDim, dim),
                            weight(tensors, p + "attn_k.weight", kvDim, dim),
                            weight(tensors, p + "attn_v.weight", kvDim, dim),
                            weight(tensors, p + "attn_output.weight", dim, queryDim),
                            weight(tensors, p + "attn_gate.weight", queryDim, dim),
                            weight(tensors, p + "attn_sink_proj.weight", heads, dim),
                            f32(tensors, p + "attn_q_clip.weight", (long) heads * c.headSize),
                            weight(tensors, p + "attn_rel.weight", heads * c.relativeLength, dim),
                            weight(
                                    tensors,
                                    p + "attn_rel_bias.weight",
                                    c.extent(swa),
                                    c.relativeLength),
                            f32(
                                    tensors,
                                    p + "attn_rel_clip.weight",
                                    (long) heads * c.relativeLength),
                            f32(tensors, p + "attn_k_conv.weight", (long) kvDim * conv),
                            f32(tensors, p + "attn_v_conv.weight", (long) kvDim * conv),
                            swa
                                    ? null
                                    : weight(
                                            tensors,
                                            p + "attn_k_conv_proj.weight",
                                            kvHeads * conv,
                                            dim),
                            swa
                                    ? null
                                    : weight(
                                            tensors,
                                            p + "attn_v_conv_proj.weight",
                                            kvHeads * conv,
                                            dim),
                            f32(tensors, p + "attn_post_conv.weight", (long) dim * conv),
                            f32(tensors, p + "ffn_res_query.weight", dim),
                            f32(tensors, p + "ffn_norm.weight", dim),
                            weight(tensors, p + "ffn_gate_inp.weight", experts, dim),
                            f32(tensors, p + "exp_probs_b.bias", experts),
                            experts(tensors, p + "ffn_gate_up_exps.weight", experts, 2 * ff, dim),
                            experts(tensors, p + "ffn_down_exps.weight", experts, dim, ff),
                            f32(tensors, p + "ffn_post_conv.weight", (long) dim * conv));
        }
        LoopWeights[] loop = new LoopWeights[c.loopEnd - c.loopStart];
        for (int i = 0; i < loop.length; i++) {
            String p = "blk." + (c.loopStart + i) + ".loop_";
            ControlWeights[] control = new ControlWeights[CONTROLS.length];
            for (int j = 0; j < control.length; j++) {
                String q = p + CONTROLS[j] + "_ctrl_";
                control[j] =
                        new ControlWeights(
                                weight(tensors, q + "down.weight", rank, 2 * dim),
                                transposed(memory, weight(tensors, q + "gain.weight", dim, rank)),
                                transposed(
                                        memory, weight(tensors, q + "forget.weight", dim, rank)));
            }
            loop[i] =
                    new LoopWeights(
                            f32(tensors, p + "msg_query.weight", (long) CONTROLS.length * dim),
                            control);
        }
        return new Weights(
                weight(tensors, "token_embd.weight", c.paddedVocabularySize, dim),
                f32(tensors, "token_embd_norm.weight", dim),
                f32(tensors, "output_res_query.weight", dim),
                f32(tensors, "output_norm.weight", dim),
                weight(tensors, "output.weight", c.paddedVocabularySize, dim),
                layers,
                loop);
    }

    /** A copy of an FP32 {@code [rows][columns]} weight as {@code [columns][rows]}. */
    private static MemoryView<MemorySegment> transposed(
            MemoryArena<MemorySegment> memory, MemoryView<MemorySegment> weight) {
        int rows = Math.toIntExact(weight.shape().flatAt(0));
        int columns = Math.toIntExact(weight.shape().flatAt(1));
        MemoryView<MemorySegment> copy = Views.allocateF32(memory, columns, rows);
        Ops.transposeCopy(weight, rows, columns, copy);
        return copy;
    }

    private static MemoryView<MemorySegment>[] experts(
            Map<String, MemoryView<MemorySegment>> tensors,
            String name,
            int experts,
            int rows,
            int columns) {
        MemoryView<MemorySegment> stacked = ModelLoader.require(tensors, name);
        requireShape(stacked, name, Shape.flat(experts, rows, columns));
        return Views.sliceLeadingAxis(stacked, experts);
    }

    static MemoryView<MemorySegment> weight(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int rows, int columns) {
        MemoryView<MemorySegment> value = ModelLoader.require(tensors, name);
        requireShape(value, name, Shape.flat(rows, columns));
        return value;
    }

    /** An FP32 tensor of {@code size} elements, read flat whatever its rank. */
    static MemoryView<MemorySegment> f32(
            Map<String, MemoryView<MemorySegment>> tensors, String name, long size) {
        MemoryView<MemorySegment> value = ModelLoader.requireF32(tensors, name);
        require(value.shape().size() == size, name + " expected " + size + " elements");
        return value;
    }

    private static void requireShape(MemoryView<MemorySegment> value, String name, Shape expected) {
        Shape actual = value.dataType().logicalShape(value.shape());
        require(actual.equals(expected), name + " expected " + expected + " but was " + actual);
    }

    static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Inkling 2: " + message);
    }
}
