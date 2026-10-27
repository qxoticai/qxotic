package com.qxotic.jinfer.models.lfm3;

import static com.qxotic.jinfer.models.lfm3.Lfm3.f32;
import static com.qxotic.jinfer.models.lfm3.Lfm3.require;
import static com.qxotic.jinfer.models.lfm3.Lfm3.weight;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.FlashAttention;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jinfer.llm.SpeechOutput;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * LFM3-Omni's speech head: the backbone's hidden state becomes one 80 ms audio frame - 12 codes,
 * one per codebook - and the frame becomes 24 kHz PCM.
 *
 * <pre>
 *   hidden ─► depthformer (MaskGIT: 3 passes, 4 codebooks revealed each) ─► 12 codes
 *   codes  ─► feedback row: the sum of 12 code embeddings, the backbone's next input
 *   codes  ─► codec: FSQ lookup, x6 upsample, a causal LFM stack, log-magnitude + phase, ISTFT
 * </pre>
 *
 * The engine drives it through {@link Speaker}, as {@link SpeechOutput} describes.
 */
final class Lfm3Speech {

    static final int SAMPLE_RATE = 24_000;
    static final int CODEBOOKS = 12;
    private static final int[] SCHEDULE = {4, 4, 4}; // codebooks revealed per depthformer pass
    private static final int FRAME_STACK = 6; // codec positions per frame
    private static final int WINDOW = 30, RING = 32; // the codec's attention window, its ring
    private static final int SHORTCONV_TAPS = 7, SHORTCONV_PARTS = 3;
    private static final int[] ALL_CODEBOOKS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    private static final int FSQ_LEVELS = 4; // [8, 7, 6, 6]: one value per level per code

    /** One LFM block of either stack: exactly one of the conv and attention groups is set. */
    record Block(
            MemoryView<MemorySegment> norm1,
            MemoryView<MemorySegment> postNorm1,
            MemoryView<MemorySegment> convIn,
            MemoryView<MemorySegment> convTaps,
            MemoryView<MemorySegment> convOut,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> gate,
            MemoryView<MemorySegment> queryNorm,
            MemoryView<MemorySegment> keyNorm,
            MemoryView<MemorySegment> norm2,
            MemoryView<MemorySegment> postNorm2,
            MemoryView<MemorySegment> ffnGate,
            MemoryView<MemorySegment> ffnUp,
            MemoryView<MemorySegment> ffnDown) {}

    /** A transformer's geometry. */
    record Stack(
            int dim, int heads, int kvHeads, int headDim, int ffnDim, float eps, Block[] blocks) {
        int kvDim() {
            return kvHeads * headDim;
        }
    }

    /** Speech-code sampling; temperature 0 is argmax. The GGUF carries the model card's values. */
    record Sampling(float temperature, int topK, float repetitionPenalty, int repetitionWindow) {}

    final int modelDim, vocab; // vocab: per codebook, the last code ends the audio
    final Sampling defaults;
    private final Stack depth, codec;
    private final MemoryView<MemorySegment> projIn, projInBias, maskEmbd, depthNorm;
    private final MemoryView<MemorySegment>[] headNorms; // per codebook [depth.dim]
    private final MemoryView<MemorySegment>[] codeEmbd; // per codebook [vocab][depth.dim]
    private final MemoryView<MemorySegment> outEmbd; // [codebooks * vocab][modelDim]
    private final float[] fsq; // [vocab - 1][FSQ_LEVELS]
    private final MemoryView<MemorySegment> quantOut, quantOutBias, codecIn, codecInBias;
    private final MemoryView<MemorySegment> codecNorm, codecOut, codecOutBias;

    Lfm3Speech(GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, int modelDim) {
        require(
                "lfm3a_gen".equals(gguf.getString("clip.gen.audio.projector_type")),
                "unsupported speech head " + gguf.getString("clip.gen.audio.projector_type"));
        this.modelDim = modelDim;
        require(
                gguf.getValue(int.class, "clip.gen.audio.projection_dim") == modelDim,
                "speech head width != model width " + modelDim);
        int dim = gguf.getValue(int.class, "clip.gen.audio.embedding_length");
        int heads = gguf.getValue(int.class, "clip.gen.audio.attention.head_count");
        depth =
                stack(
                        tensors,
                        "a.gen.code.blk.",
                        gguf.getValue(int.class, "clip.gen.audio.block_count"),
                        dim,
                        heads,
                        gguf.getValue(int.class, "clip.gen.audio.attention.head_count_kv"),
                        gguf.getValue(int.class, "clip.gen.audio.feed_forward_length"),
                        gguf.getValue(float.class, "clip.gen.audio.attention.layer_norm_epsilon"));
        MemoryView<MemorySegment>[] tables =
                Views.sliceLeadingAxis(ModelLoader.require(tensors, "a.gen.code.embd.weight"));
        require(tables.length == CODEBOOKS, "expected " + CODEBOOKS + " codebooks");
        codeEmbd = tables;
        vocab = Math.toIntExact(tables[0].shape().flatAt(0));
        projIn = weight(tensors, "a.gen.code.proj_in.weight", CODEBOOKS * dim, modelDim);
        projInBias = f32(tensors, "a.gen.code.proj_in.bias", CODEBOOKS * dim);
        maskEmbd = f32(tensors, "a.gen.code.mask_embd", dim);
        depthNorm = f32(tensors, "a.gen.code.output_norm.weight", dim);
        MemoryView<MemorySegment> headNorm =
                f32(tensors, "a.gen.code.head_norm.weight", CODEBOOKS, dim);
        headNorms = rows(headNorm, CODEBOOKS);
        outEmbd = weight(tensors, "a.gen.code.out_embd.weight", CODEBOOKS * vocab, modelDim);

        // the codec: liquid-codec's decoder, its shape fixed by its config, its sizes read off the
        // tensors
        MemoryView<MemorySegment> table =
                f32(tensors, "a.gen.wav.quant.first.codebook.weight", vocab - 1, FSQ_LEVELS);
        fsq = Views.toFloatArray(table, "fsq");
        int quantDim =
                Math.toIntExact(
                        ModelLoader.require(tensors, "a.gen.wav.quant.first.out_proj.bias")
                                .shape()
                                .flatAt(0));
        quantOut =
                f32(
                        tensors,
                        "a.gen.wav.quant.first.out_proj.weight",
                        quantDim,
                        CODEBOOKS * FSQ_LEVELS);
        quantOutBias = f32(tensors, "a.gen.wav.quant.first.out_proj.bias", quantDim);
        int codecDim =
                Math.toIntExact(
                        ModelLoader.require(tensors, "a.gen.wav.tfm.in_proj.bias")
                                .shape()
                                .flatAt(0));
        codecIn = weight(tensors, "a.gen.wav.tfm.in_proj.weight", codecDim, quantDim);
        codecInBias = f32(tensors, "a.gen.wav.tfm.in_proj.bias", codecDim);
        int codecLayers = 0, codecHeadDim = 0, codecKvDim = 0;
        for (;
                tensors.containsKey("a.gen.wav.tfm.blk." + codecLayers + ".ln1.weight");
                codecLayers++) {
            String p = "a.gen.wav.tfm.blk." + codecLayers + ".";
            if (tensors.containsKey(p + "attn_q_norm.weight")) {
                codecHeadDim =
                        Math.toIntExact(tensors.get(p + "attn_q_norm.weight").shape().flatAt(0));
                codecKvDim = logical(tensors, p + "attn_k.weight", 0);
            }
        }
        require(codecHeadDim > 0, "the codec has no attention layer");
        int codecFfn = logical(tensors, "a.gen.wav.tfm.blk.0.ffn_down.weight", 1);
        codec =
                stack(
                        tensors,
                        "a.gen.wav.tfm.blk.",
                        codecLayers,
                        codecDim,
                        codecDim / codecHeadDim,
                        codecKvDim / codecHeadDim,
                        codecFfn,
                        1e-5f);
        codecNorm = f32(tensors, "a.gen.wav.tfm.output_norm.weight", codecDim);
        codecOut = weight(tensors, "a.gen.wav.tfm.out_proj.weight", 2 * Istft.BINS, codecDim);
        codecOutBias = f32(tensors, "a.gen.wav.tfm.out_proj.bias", 2 * Istft.BINS);

        defaults =
                new Sampling(
                        gguf.getValueOrDefault(float.class, "clip.gen.audio.sampling.temp", 0.4f),
                        gguf.getValueOrDefault(int.class, "clip.gen.audio.sampling.top_k", 0),
                        gguf.getValueOrDefault(
                                float.class, "clip.gen.audio.sampling.penalty_repeat", 1.2f),
                        gguf.getValueOrDefault(
                                int.class, "clip.gen.audio.sampling.penalty_last_n", 64));
    }

    /** A weight's logical size along {@code axis} (rows 0, columns 1), whatever its type. */
    private static int logical(
            Map<String, MemoryView<MemorySegment>> tensors, String name, int axis) {
        MemoryView<MemorySegment> w = ModelLoader.require(tensors, name);
        return Math.toIntExact(w.dataType().logicalShape(w.shape()).flatAt(axis));
    }

    /** The rows of a dense {@code [n][width]} view, each a {@code [1][width]} view. */
    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment>[] rows(MemoryView<MemorySegment> t, int n) {
        MemoryView<MemorySegment>[] rows = new MemoryView[n];
        for (int i = 0; i < n; i++) rows[i] = t.slice(0, i, i + 1);
        return rows;
    }

    private static Stack stack(
            Map<String, MemoryView<MemorySegment>> tensors,
            String prefix,
            int layers,
            int dim,
            int heads,
            int kvHeads,
            int ffnDim,
            float eps) {
        int headDim = dim / heads, kvDim = kvHeads * headDim;
        Block[] blocks = new Block[layers];
        for (int l = 0; l < layers; l++) {
            String p = prefix + l + ".";
            boolean conv = tensors.containsKey(p + "conv_dw.weight");
            boolean post = tensors.containsKey(p + "attn_post_norm.weight");
            blocks[l] =
                    new Block(
                            f32(tensors, p + "ln1.weight", dim),
                            post ? f32(tensors, p + "attn_post_norm.weight", dim) : null,
                            conv
                                    ? weight(
                                            tensors,
                                            p + "conv_pw1.weight",
                                            SHORTCONV_PARTS * dim,
                                            dim)
                                    : null,
                            conv ? f32(tensors, p + "conv_dw.weight", dim, SHORTCONV_TAPS) : null,
                            conv ? weight(tensors, p + "conv_pw2.weight", dim, dim) : null,
                            conv ? null : weight(tensors, p + "attn_q.weight", dim, dim),
                            conv ? null : weight(tensors, p + "attn_k.weight", kvDim, dim),
                            conv ? null : weight(tensors, p + "attn_v.weight", kvDim, dim),
                            conv ? null : weight(tensors, p + "attn_out.weight", dim, dim),
                            conv ? null : weight(tensors, p + "attn_gate.weight", heads, dim),
                            conv ? null : f32(tensors, p + "attn_q_norm.weight", headDim),
                            conv ? null : f32(tensors, p + "attn_k_norm.weight", headDim),
                            f32(tensors, p + "ln2.weight", dim),
                            post ? f32(tensors, p + "ffn_post_norm.weight", dim) : null,
                            weight(tensors, p + "ffn_gate.weight", ffnDim, dim),
                            weight(tensors, p + "ffn_up.weight", ffnDim, dim),
                            weight(tensors, p + "ffn_down.weight", dim, ffnDim));
        }
        return new Stack(dim, heads, kvHeads, headDim, ffnDim, eps, blocks);
    }

    /**
     * Buffers for one stack over up to {@code rows} positions: bidirectional without state (the
     * depthformer), or causal with its window and conv state kept across calls (the codec).
     */
    private static final class Pass {
        final Stack stack;
        final boolean causal;
        final RoPE.Schedule rope;
        final MemoryView<MemorySegment> x, normed, mixed, wide, query, key, value, gate, attn;
        final MemoryView<MemorySegment> hidden, hidden2, cos, sin;
        final MemoryView<MemorySegment>[] keyRing, valueRing, convState; // codec state, per block

        @SuppressWarnings("unchecked")
        Pass(Stack stack, int rows, boolean causal, MemoryArena<MemorySegment> arena) {
            this.stack = stack;
            this.causal = causal;
            this.rope = RoPE.plain(stack.headDim, 1e6); // the converter pins both rope bases
            int dim = stack.dim, kvDim = stack.kvDim();
            x = Views.allocateF32(arena, rows, dim);
            normed = Views.allocateF32(arena, rows, dim);
            mixed = Views.allocateF32(arena, rows, dim);
            wide = Views.allocateF32(arena, rows, SHORTCONV_PARTS * dim);
            query = Views.allocateF32(arena, rows, dim);
            key = Views.allocateF32(arena, rows, kvDim);
            value = Views.allocateF32(arena, rows, kvDim);
            gate = Views.allocateF32(arena, rows, stack.heads);
            attn = Views.allocateF32(arena, rows, dim);
            hidden = Views.allocateF32(arena, rows, stack.ffnDim);
            hidden2 = Views.allocateF32(arena, rows, stack.ffnDim);
            cos = Views.allocateF32(arena, rows, stack.headDim / 2);
            sin = Views.allocateF32(arena, rows, stack.headDim / 2);
            int n = stack.blocks.length;
            keyRing = new MemoryView[n];
            valueRing = new MemoryView[n];
            convState = new MemoryView[n];
            if (!causal) return;
            for (int l = 0; l < n; l++) {
                if (stack.blocks[l].convIn != null) {
                    convState[l] = Views.allocateF32(arena, (long) (SHORTCONV_TAPS - 1) * dim);
                } else {
                    keyRing[l] = Views.allocateF32(arena, RING, kvDim);
                    valueRing[l] = Views.allocateF32(arena, RING, kvDim);
                }
            }
        }
    }

    /**
     * The stack over {@code p.x[0, rows)} in place: pre-norm blocks with optional sandwich norms.
     * Bidirectional without state (the depthformer); causal over a {@link #WINDOW}-position window
     * continuing at {@code start} (the codec).
     */
    private static void run(Pass p, int rows, int start) {
        Stack s = p.stack;
        int dim = s.dim;
        RoPE.fill(p.cos, p.sin, start, rows, s.headDim / 2, p.rope);
        for (int l = 0; l < s.blocks.length; l++) {
            Block b = s.blocks[l];
            Norms.rmsnormRowsGgml(p.normed, p.x, b.norm1, rows, dim, s.eps);
            if (b.convIn != null) {
                MatMul.gemm(b.convIn, p.normed, p.wide, rows);
                Convolutions.shortConvScan(
                        b.convTaps,
                        p.convState[l],
                        p.wide,
                        p.attn,
                        rows,
                        dim,
                        SHORTCONV_TAPS,
                        SHORTCONV_PARTS);
                MatMul.gemm(b.convOut, p.attn, p.mixed, rows);
            } else {
                attention(p, b, l, rows, start);
            }
            if (b.postNorm1 != null)
                Norms.rmsnormRowsGgml(p.mixed, p.mixed, b.postNorm1, rows, dim, s.eps);
            Ops.addRows(p.x, p.mixed, rows, dim);
            Norms.rmsnormRowsGgml(p.normed, p.x, b.norm2, rows, dim, s.eps);
            Towers.ffn(
                    b.ffnGate, b.ffnUp, b.ffnDown, p.normed, p.hidden, p.hidden2, p.mixed, rows,
                    s.ffnDim);
            if (b.postNorm2 != null)
                Norms.rmsnormRowsGgml(p.mixed, p.mixed, b.postNorm2, rows, dim, s.eps);
            Ops.addRows(p.x, p.mixed, rows, dim);
        }
    }

    /** GQA with per-head Q/K RMS norms, NeoX RoPE and the per-head sigmoid gate; F32 throughout. */
    private static void attention(Pass p, Block b, int l, int rows, int start) {
        Stack s = p.stack;
        int dim = s.dim, kvDim = s.kvDim(), headDim = s.headDim;
        MatMul.gemm(b.query, p.normed, p.query, rows);
        MatMul.gemm(b.key, p.normed, p.key, rows);
        MatMul.gemm(b.value, p.normed, p.value, rows);
        MatMul.gemm(b.gate, p.normed, p.gate, rows);
        Parallel.forLoop(
                rows,
                r -> {
                    Towers.headNormRope(
                            p.query,
                            (long) r * dim,
                            s.heads,
                            headDim,
                            b.queryNorm,
                            s.eps,
                            r,
                            p.cos,
                            p.sin);
                    Towers.headNormRope(
                            p.key,
                            (long) r * kvDim,
                            s.kvHeads,
                            headDim,
                            b.keyNorm,
                            s.eps,
                            r,
                            p.cos,
                            p.sin);
                });
        float scale = 1f / (float) Math.sqrt(headDim);
        if (p.causal) {
            FlashAttention.slidingWindowPrefill(
                    p.query,
                    p.attn,
                    p.keyRing[l],
                    p.valueRing[l],
                    p.key,
                    p.value,
                    s.heads,
                    start,
                    rows,
                    headDim,
                    kvDim,
                    dim,
                    kvDim,
                    s.heads / s.kvHeads,
                    scale,
                    WINDOW,
                    RING - 1,
                    null);
            for (int r = 0; r < rows; r++) {
                long at = (long) ((start + r) & (RING - 1)) * kvDim;
                Convert.copyF32(p.key, (long) r * kvDim, p.keyRing[l], at, kvDim);
                Convert.copyF32(p.value, (long) r * kvDim, p.valueRing[l], at, kvDim);
            }
        } else {
            FlashAttention.bidirectionalPrefill(
                    p.query,
                    p.attn,
                    p.key,
                    p.value,
                    s.heads,
                    rows,
                    headDim,
                    kvDim,
                    dim,
                    s.heads / s.kvHeads,
                    scale);
        }
        long gateStride = p.gate.stride().flatAt(0);
        Parallel.forLoop(
                rows,
                r -> {
                    for (int h = 0; h < s.heads; h++)
                        Ops.multiplyInPlace(
                                p.attn,
                                (long) r * dim + (long) h * headDim,
                                headDim,
                                Activations.sigmoid(
                                        Views.getFloat(p.gate, r * gateStride + h, "gate")));
                });
        MatMul.gemm(b.output, p.attn, p.mixed, rows);
    }

    /**
     * One utterance's state: the depthformer's buffers, the codec's window and ISTFT, the recent
     * codes the repetition penalty reads. Everything a frame touches is allocated here, once; close
     * it when the speech ends.
     */
    final class Stream implements AutoCloseable {
        private final MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        private final Pass depthPass = new Pass(depth, CODEBOOKS, false, arena);
        private final Pass codecPass = new Pass(codec, FRAME_STACK, true, arena);
        private final MemoryView<MemorySegment> projected, logits, quantIn, quant;
        private final MemoryView<MemorySegment> codecRow, spectra;
        private final MemoryView<MemorySegment>[] depthRows, logitRows;

        /** The last frame's feedback row, the backbone's next input. */
        final MemoryView<MemorySegment> feedback;

        /** {@link #feedback} as the keyed batch the backbone ingests. */
        private Batch row;

        /** Where the caller puts the backbone's hidden state before each frame. */
        final MemoryView<MemorySegment> hidden;

        private final Feedback feedbacks = new Feedback(arena);
        private final int[] history;
        private final float[] logitRow = new float[vocab], spectrum = new float[2 * Istft.BINS];
        private final float[] quantValues = new float[CODEBOOKS * FSQ_LEVELS];
        private final double[] probabilities = new double[vocab];
        private final boolean[] seen = new boolean[vocab];
        private final Istft istft = new Istft();
        private int frames, position;

        private Stream(int historyFrames) {
            int dim = depth.dim;
            projected = Views.allocateF32(arena, 1, CODEBOOKS * dim);
            logits = Views.allocateF32(arena, CODEBOOKS, vocab);
            feedback = feedbacks.row;
            hidden = Views.allocateF32(arena, 1, modelDim);
            quantIn = Views.allocateF32(arena, 1, CODEBOOKS * FSQ_LEVELS);
            quant = Views.allocateF32(arena, 1, quantOutBias.shape().flatAt(0));
            codecRow = codecPass.normed.slice(0, 0, 1);
            spectra = Views.allocateF32(arena, FRAME_STACK, 2 * Istft.BINS);
            depthRows = rows(depthPass.normed, CODEBOOKS);
            logitRows = rows(logits, CODEBOOKS);
            history = new int[Math.max(1, historyFrames) * CODEBOOKS];
        }

        /**
         * Samples the frame for {@code hiddenState} (the backbone's final-normed last row) into
         * {@code codes}, revealing the codebooks in {@code order}, 4 per depthformer pass, and sets
         * {@link #feedback}. False when a codebook picked the end code: the speech is over, and the
         * frame is the end frame - every code the end code, its feedback set too.
         */
        boolean frame(
                MemoryView<MemorySegment> hiddenState,
                int[] order,
                Sampling sampling,
                RandomGenerator rng,
                int[] codes) {
            Arrays.fill(codes, -1);
            project(hiddenState);
            int k = 0;
            for (int pick : SCHEDULE) {
                pass(codes, order, k, k + pick);
                for (int j = 0; j < pick; j++, k++) {
                    int cb = order[k];
                    Views.copyToArray(logits, (long) cb * vocab, logitRow, 0, vocab, "logits");
                    penalize(logitRow, cb, sampling);
                    codes[cb] = sample(logitRow, sampling, rng);
                }
            }
            for (int c : codes) {
                if (c == vocab - 1) {
                    Arrays.fill(codes, vocab - 1);
                    feedback(codes);
                    return false;
                }
            }
            System.arraycopy(
                    codes,
                    0,
                    history,
                    (frames % (history.length / CODEBOOKS)) * CODEBOOKS,
                    CODEBOOKS);
            frames++;
            feedback(codes);
            return true;
        }

        /**
         * The depthformer's logits {@code [codebook][vocab]} for the codes known so far (-1
         * masked), every codebook's.
         */
        MemoryView<MemorySegment> codeLogits(MemoryView<MemorySegment> hiddenState, int[] codes) {
            project(hiddenState);
            pass(codes, ALL_CODEBOOKS, 0, CODEBOOKS);
            return logits;
        }

        /** proj_in of the hidden state, once per frame: every pass starts from it. */
        private void project(MemoryView<MemorySegment> hiddenState) {
            MatMul.gemv(projIn, hiddenState, projected);
            Ops.addInPlace(projected, 0, projInBias, 0, CODEBOOKS * depth.dim);
        }

        /**
         * One depthformer pass over the codes known so far, its logits for the codebooks {@code
         * heads[from, to)} alone: the others are not read before the next pass recomputes them.
         */
        private void pass(int[] codes, int[] heads, int from, int to) {
            int dim = depth.dim;
            Convert.copyF32(projected, 0, depthPass.x, 0, (long) CODEBOOKS * dim);
            for (int cb = 0; cb < CODEBOOKS; cb++) {
                if (codes[cb] >= 0) {
                    Convert.copyToF32(
                            codeEmbd[cb], (long) codes[cb] * dim, depthPass.normed, 0, dim);
                    Ops.addInPlace(depthPass.x, (long) cb * dim, depthPass.normed, 0, dim);
                } else {
                    Ops.addInPlace(depthPass.x, (long) cb * dim, maskEmbd, 0, dim);
                }
            }
            run(depthPass, CODEBOOKS, 0);
            // the stack's norm, then the codebook's own, then its tied embedding table
            for (int i = from; i < to; i++) {
                long row = (long) heads[i] * dim;
                Norms.rmsnormGgml(
                        depthPass.normed, row, depthPass.x, row, depthNorm, dim, depth.eps);
                Norms.rmsnormGgml(
                        depthPass.normed,
                        row,
                        depthPass.normed,
                        row,
                        headNorms[heads[i]],
                        dim,
                        depth.eps);
                MatMul.gemv(codeEmbd[heads[i]], depthRows[heads[i]], logitRows[heads[i]]);
            }
        }

        /** {@link #feedback} for {@code codes}, and {@link #row} with it. */
        void feedback(int[] codes) {
            row = feedbacks.batch(codes);
        }

        /** The frame's audio: {@value #FRAME_STACK} ISTFT hops, less the start padding at first. */
        float[] decode(int[] codes) {
            for (int cb = 0; cb < CODEBOOKS; cb++)
                System.arraycopy(
                        fsq, codes[cb] * FSQ_LEVELS, quantValues, cb * FSQ_LEVELS, FSQ_LEVELS);
            Views.copyFromArray(quantIn, 0, quantValues, 0, quantValues.length, "fsq");
            MatMul.gemv(quantOut, quantIn, quant);
            Ops.addInPlace(
                    quant, 0, quantOutBias, 0, Math.toIntExact(quantOutBias.shape().flatAt(0)));
            int dim = codec.dim;
            MatMul.gemv(codecIn, quant, codecRow);
            Ops.addInPlace(codecPass.normed, 0, codecInBias, 0, dim);
            // nearest upsampling: every frame is FRAME_STACK codec positions
            for (int r = 0; r < FRAME_STACK; r++)
                Convert.copyF32(codecPass.normed, 0, codecPass.x, (long) r * dim, dim);
            run(codecPass, FRAME_STACK, position);
            position += FRAME_STACK;
            Norms.rmsnormRowsGgml(
                    codecPass.normed, codecPass.x, codecNorm, FRAME_STACK, dim, codec.eps);
            MatMul.gemm(codecOut, codecPass.normed, spectra, FRAME_STACK);
            Ops.addRowBiasInPlace(spectra, 0, codecOutBias, 0, FRAME_STACK, 2 * Istft.BINS);
            float[] pcm = new float[FRAME_STACK * Istft.HOP];
            int n = 0;
            for (int r = 0; r < FRAME_STACK; r++) {
                Views.copyToArray(
                        spectra,
                        (long) r * 2 * Istft.BINS,
                        spectrum,
                        0,
                        spectrum.length,
                        "spectrum");
                n += istft.frame(spectrum, pcm, n);
            }
            return n == pcm.length ? pcm : Arrays.copyOf(pcm, n);
        }

        /** The ISTFT's held-back tail, less its padding: the end of the audio. */
        float[] flush() {
            return istft.flush();
        }

        /** The CTRL penalty on the codes this codebook took in the recent frames. */
        private void penalize(float[] l, int cb, Sampling s) {
            if (s.repetitionPenalty() <= 1f || s.repetitionWindow() <= 0) return;
            int capacity = history.length / CODEBOOKS;
            int n = Math.min(frames, Math.min(capacity, s.repetitionWindow()));
            Arrays.fill(seen, false);
            for (int f = frames - n; f < frames; f++) {
                int c = history[(f % capacity) * CODEBOOKS + cb];
                if (seen[c]) continue;
                seen[c] = true;
                l[c] = l[c] > 0 ? l[c] / s.repetitionPenalty() : l[c] * s.repetitionPenalty();
            }
        }

        /** Argmax at temperature 0, else top-k (ties kept) then a tempered softmax draw. */
        private int sample(float[] l, Sampling s, RandomGenerator rng) {
            int best = 0;
            for (int v = 1; v < l.length; v++) if (l[v] > l[best]) best = v;
            if (s.temperature() <= 0f) return best;
            float min = Float.NEGATIVE_INFINITY;
            if (s.topK() > 0 && s.topK() < l.length) {
                float[] sorted =
                        l.clone(); // ponytail: a sort per draw, top-k is off in Omni's card
                Arrays.sort(sorted);
                min = sorted[l.length - s.topK()];
            }
            double sum = 0;
            for (int v = 0; v < l.length; v++)
                sum +=
                        probabilities[v] =
                                l[v] < min ? 0 : Math.exp((l[v] - l[best]) / s.temperature());
            double r = rng.nextDouble() * sum;
            for (int v = 0; v < l.length; v++) if ((r -= probabilities[v]) <= 0) return v;
            return l.length - 1;
        }

        @Override
        public void close() {
            Arenas.close(arena);
        }
    }

    Stream newStream(Sampling sampling) {
        return new Stream(sampling.repetitionWindow());
    }

    /**
     * A frame's feedback row, the backbone's input for it: the sum of its code embeddings,
     * accumulated in double. The replay of a spoken turn computes the same rows the live reply fed.
     */
    final class Feedback {
        private final int[] ids = new int[CODEBOOKS];
        private final MemoryView<MemorySegment> rows;
        private final float[] values = new float[modelDim];
        private final double[] sum = new double[modelDim];

        /** The row {@link #of} writes, {@code [1][modelDim]}. */
        final MemoryView<MemorySegment> row;

        Feedback(MemoryArena<MemorySegment> arena) {
            rows = Views.allocateF32(arena, CODEBOOKS, modelDim);
            row = Views.allocateF32(arena, 1, modelDim);
        }

        /**
         * The row as the batch a reply ingests and its replay re-emits, keyed by the codes - the
         * one spelling the prompt cache matches the two by.
         */
        Batch batch(int[] codes) {
            return Batch.embeddings(of(codes), 1, false, frameKey(codes));
        }

        MemoryView<MemorySegment> of(int[] codes) {
            for (int cb = 0; cb < CODEBOOKS; cb++) ids[cb] = codes[cb] + cb * vocab;
            Convert.gatherToF32(outEmbd, ids, 0, CODEBOOKS, rows, 0, modelDim);
            Arrays.fill(sum, 0);
            for (int cb = 0; cb < CODEBOOKS; cb++) {
                Views.copyToArray(rows, (long) cb * modelDim, values, 0, modelDim, "feedback");
                for (int i = 0; i < modelDim; i++) sum[i] += values[i];
            }
            for (int i = 0; i < modelDim; i++) values[i] = (float) sum[i];
            Views.copyFromArray(row, 0, values, 0, modelDim, "feedback");
            return row;
        }
    }

    /** The cache identity of a frame's row: its codes, which fix the row. */
    private static ContentKey frameKey(int[] codes) {
        ByteBuffer bytes = ByteBuffer.allocate(FRAME_DOMAIN.length + 4 * codes.length);
        bytes.put(FRAME_DOMAIN);
        for (int code : codes) bytes.putInt(code);
        return ContentKey.sha256(bytes.array());
    }

    private static final byte[] FRAME_DOMAIN =
            "lfm3-speech-frame\0".getBytes(StandardCharsets.US_ASCII);

    Speaker speaker(
            Lfm3 model, Lfm3.State state, boolean transcript, RandomGenerator random, int textEnd) {
        return new Speaker(model, state, transcript, random, textEnd);
    }

    /**
     * One spoken reply's lanes, as LiquidAI's {@code generate()} and llama.cpp's speech helper run
     * them. Speech alone speaks to its end frame, then text follows. Interleaved, the reply
     * alternates {@value #TEXT_SLOTS} text tokens and {@value #AUDIO_SLOTS} frames; {@code
     * <|text_end|>} ends the transcript and the speech runs on to its end frame. After it, {@code
     * <|text_end|>} ends the reply. Frames reveal their codebooks in a fresh random order each, as
     * the model was trained, sampled with the GGUF's settings.
     */
    final class Speaker implements SpeechOutput.Speaker {
        private static final int TEXT_SLOTS = 6, AUDIO_SLOTS = 12;
        private final Lfm3 model;
        private final Lfm3.State state;
        private final RandomGenerator random;
        private final int textEnd;
        private final Stream stream = newStream(defaults);
        private final int[] order = new int[CODEBOOKS], codes = new int[CODEBOOKS];
        private boolean interleaving, speaking, transcriptDone, flushed;
        private int slots = TEXT_SLOTS;

        private Speaker(
                Lfm3 model,
                Lfm3.State state,
                boolean transcript,
                RandomGenerator random,
                int textEnd) {
            this.model = model;
            this.state = state;
            this.random = random;
            this.textEnd = textEnd;
            interleaving = transcript;
            speaking = !transcript;
        }

        @Override
        public boolean textNext() {
            return !speaking;
        }

        @Override
        public SpeechOutput.Lane text(int token) {
            if (!interleaving)
                return token == textEnd ? SpeechOutput.Lane.END : SpeechOutput.Lane.TEXT;
            if (token == textEnd) {
                transcriptDone = true;
                speaking = true;
                return SpeechOutput.Lane.CONTROL; // the switch itself takes no slot
            }
            slot();
            return SpeechOutput.Lane.TEXT;
        }

        @Override
        public SpeechOutput.Frame frame() {
            model.hiddenState(state, stream.hidden);
            for (int i = 0; i < CODEBOOKS; i++) order[i] = i;
            for (int i = CODEBOOKS - 1; i > 0; i--) {
                int j = random.nextInt(i + 1), t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            boolean more = stream.frame(stream.hidden, order, defaults, random, codes);
            float[] pcm;
            if (more) {
                pcm = stream.decode(codes);
                slot();
            } else {
                pcm = flush(); // the codec's tail closes the speech; text follows
                interleaving = false;
                speaking = false;
            }
            return new SpeechOutput.Frame(codes.clone(), clamp(pcm), stream.row);
        }

        /** An interleaved slot taken: the block flips when its count runs out. */
        private void slot() {
            if (interleaving && !transcriptDone && --slots == 0) {
                speaking = !speaking;
                slots = speaking ? AUDIO_SLOTS : TEXT_SLOTS;
            }
        }

        @Override
        public float[] flush() {
            if (flushed) return new float[0];
            flushed = true;
            return clamp(stream.flush());
        }

        @Override
        public void close() {
            stream.close();
        }
    }

    /** PCM within [-1, 1]: the ISTFT can overshoot by a hair, as any WAV writer clips. */
    private static float[] clamp(float[] pcm) {
        for (int i = 0; i < pcm.length; i++) pcm[i] = Math.max(-1f, Math.min(1f, pcm[i]));
        return pcm;
    }

    /**
     * llama.cpp's streaming ISTFT: a periodic Hann window, overlap-add normalized by the summed
     * squared window, the first {@code (N_FFT - HOP) / 2} samples dropped as padding and as many
     * held back until {@link #flush}. Its inverse FFT is llama.cpp's own: radix-2 levels scaling by
     * 1/2 over an odd-length inverse DFT leaf.
     */
    static final class Istft {
        static final int N_FFT = 1280, HOP = 320;
        static final int BINS = N_FFT / 2 + 1;
        private static final int PADDING = (N_FFT - HOP) / 2;
        private static final float[] SIN = new float[N_FFT],
                COS = new float[N_FFT],
                HANN = new float[N_FFT];

        static {
            for (int i = 0; i < N_FFT; i++) {
                double theta = 2 * Math.PI * i / N_FFT;
                SIN[i] = (float) Math.sin((float) theta);
                COS[i] = (float) Math.cos((float) theta);
                HANN[i] = (float) (0.5 * (1.0 - COS[i]));
            }
        }

        private final float[] overlap = new float[N_FFT], windowSum = new float[N_FFT];
        private final float[] in = new float[N_FFT * 8], out = new float[N_FFT * 8];
        private int padding = PADDING;

        /**
         * One STFT frame - log magnitudes, then phases - into {@code pcm[at]}; returns samples
         * written.
         */
        int frame(float[] logMagnitudePhase, float[] pcm, int at) {
            for (int j = 0; j < BINS; j++) {
                float magnitude = (float) Math.exp(logMagnitudePhase[j]);
                float phase = logMagnitudePhase[BINS + j];
                in[2 * j] = magnitude * (float) Math.cos(phase);
                in[2 * j + 1] = magnitude * (float) Math.sin(phase);
            }
            for (int j = 1; j < BINS - 1; j++) {
                in[2 * (N_FFT - j)] = in[2 * j];
                in[2 * (N_FFT - j) + 1] = -in[2 * j + 1];
            }
            inverse(in, 0, N_FFT, out, 0);
            for (int j = 0; j < N_FFT; j++) {
                windowSum[j] += HANN[j] * HANN[j];
                overlap[j] += out[2 * j] * HANN[j];
            }
            int n = 0;
            for (int i = 0; i < HOP; i++) {
                if (padding > 0) {
                    padding--;
                    continue;
                }
                pcm[at + n++] = windowSum[i] > 1e-8f ? overlap[i] / windowSum[i] : overlap[i];
            }
            shift(HOP);
            return n;
        }

        /** The overlap still held, less the trailing padding. */
        float[] flush() {
            float[] tail = new float[N_FFT - HOP];
            for (int i = 0; i < tail.length; i++)
                tail[i] = windowSum[i] > 1e-8f ? overlap[i] / windowSum[i] : overlap[i];
            shift(N_FFT - HOP);
            return Arrays.copyOf(tail, tail.length - PADDING);
        }

        private void shift(int n) {
            System.arraycopy(overlap, n, overlap, 0, N_FFT - n);
            Arrays.fill(overlap, N_FFT - n, N_FFT, 0f);
            System.arraycopy(windowSum, n, windowSum, 0, N_FFT - n);
            Arrays.fill(windowSum, N_FFT - n, N_FFT, 0f);
        }

        /** llama.cpp's fft_impl<Inverse, complex input>, op for op. */
        private static void inverse(float[] in, int inAt, int n, float[] out, int outAt) {
            if (n == 1) {
                out[outAt] = in[inAt];
                out[outAt + 1] = in[inAt + 1];
                return;
            }
            int step = N_FFT / n;
            if (n % 2 == 1) {
                float scale = 1f / n;
                for (int k = 0; k < n; k++) {
                    float re = 0, im = 0;
                    for (int j = 0; j < n; j++) {
                        int index = k * j * step % N_FFT;
                        float c = COS[index], s = SIN[index];
                        float a = in[inAt + 2 * j], b = in[inAt + 2 * j + 1];
                        re += a * c - b * s;
                        im += a * s + b * c;
                    }
                    out[outAt + 2 * k] = re * scale;
                    out[outAt + 2 * k + 1] = im * scale;
                }
                return;
            }
            int half = n / 2, child = inAt + 2 * n;
            for (int i = 0; i < half; i++) {
                in[child + 2 * i] = in[inAt + 4 * i];
                in[child + 2 * i + 1] = in[inAt + 4 * i + 1];
            }
            int even = outAt + 2 * n;
            inverse(in, child, half, out, even);
            for (int i = 0; i < half; i++) {
                in[child + 2 * i] = in[inAt + 4 * i + 2];
                in[child + 2 * i + 1] = in[inAt + 4 * i + 3];
            }
            int odd = even + n;
            inverse(in, child, half, out, odd);
            for (int k = 0; k < half; k++) {
                float re = COS[k * step], im = SIN[k * step];
                float oddRe = out[odd + 2 * k], oddIm = out[odd + 2 * k + 1];
                float evenRe = out[even + 2 * k], evenIm = out[even + 2 * k + 1];
                out[outAt + 2 * k] = 0.5f * (evenRe + re * oddRe - im * oddIm);
                out[outAt + 2 * k + 1] = 0.5f * (evenIm + re * oddIm + im * oddRe);
                out[outAt + 2 * (k + half)] = 0.5f * (evenRe - re * oddRe + im * oddIm);
                out[outAt + 2 * (k + half) + 1] = 0.5f * (evenIm - re * oddIm - im * oddRe);
            }
        }
    }
}
