// Gemma 4 Conformer audio encoder (projector_type = gemma4a, e.g. gemma-4-E2B/E4B).
//
// Reference: llama.cpp tools/mtmd/models/gemma4a.cpp (clip_graph_gemma4a::build) plus the
// host-side inputs in clip.cpp (blocked causal mask, sinusoidal RPE); AudioPreprocess supplies
// the log-mel chunks. Parity is pinned stage-wise against llama-mtmd-cli MTMD_DEBUG_GRAPH traces
// (test-fixtures/audio/oracle).
//
// Pipeline per 30 s mel chunk [128 mel x T frames]:
//   1. subsampling: two 3x3 stride-2 conv2d over (freq, time) with per-channel LayerNorm + ReLU
//      (freq 128 -> 64 -> 32, channels 1 -> 128 -> 32, time T -> ceil(T/2) -> ceil(T/4)),
//      flatten [ch * freq = 1024] per timestep, linear a.input_projection (1024 -> 1024)
//   2. 12 Conformer blocks, each: half-residual macaron FFN | chunked local attention with
//      sinusoidal RPE (chunk 12, past 12, per-dim Q scale, log2-base scaling, tanh softcap 50)
//      | conv module (pointwise GLU -> causal depthwise k=5 -> RMS -> SiLU -> pointwise) |
//      half-residual FFN' | RMS out-norm
//   3. tail: a.pre_encode.out (1024 -> 1536, bias) -> weightless RMS -> mm.a.input_projection
//      (1536 -> 1536)
// Each 30 s chunk encodes independently (no cross-chunk attention).
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Embedder;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FastMath;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.Norms;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.kernels.GGMLTensorEntry;
import com.qxotic.jinfer.kernels.ModelLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public final class Gemma4Conformer implements Embedder<Media.Audio> {

    // chunked local attention geometry (fixed in the reference)
    static final int CHUNK = 12; // queries per block
    static final int PAST = 12; // max past horizon
    static final int CONTEXT = CHUNK + PAST; // 24 context slots per block
    static final int RPE = PAST + 1; // 13 relative positions

    record Block(
            F32FloatTensor ffNorm,
            Clamped ffUp,
            Clamped ffDown,
            F32FloatTensor ffPostNorm,
            F32FloatTensor attnPreNorm,
            Clamped q,
            Clamped k,
            Clamped v,
            Clamped o,
            F32FloatTensor attnPostNorm,
            F32FloatTensor rel, // kRel @ posEmb, [RPE x dim], precomputed at load
            F32FloatTensor perDimScale,
            F32FloatTensor convPreNorm,
            Clamped convPw1,
            F32FloatTensor convDw,
            F32FloatTensor convPostNorm,
            Clamped convPw2,
            F32FloatTensor ffNorm1,
            Clamped ffUp1,
            Clamped ffDown1,
            F32FloatTensor ffPostNorm1,
            F32FloatTensor ln2) {}

    final int dim; // 1024
    final int heads; // 8
    final int headDim; // 128
    final int ffDim; // 4096
    final int nMel; // 128
    final int outDim; // 1536
    final float eps;
    final AudioPreprocess preprocess;

    final float[] conv0; // [3*3*1*128] taps, layout ((oc*ic + ic)*ky)*3 + kx
    final F32FloatTensor norm0; // [128]
    final float[] conv1; // [3*3*128*32]
    final F32FloatTensor norm1; // [32]
    final Clamped inputProj; // [1024 -> 1024]
    final Block[] blocks;
    final Clamped outProj; // [1024 -> 1536]
    final F32FloatTensor outProjBias; // [1536]
    final Clamped mmProj; // [1536 -> 1536]

    private Gemma4Conformer(
            int dim,
            int heads,
            int ffDim,
            int nMel,
            int outDim,
            float eps,
            float[] conv0,
            F32FloatTensor norm0,
            float[] conv1,
            F32FloatTensor norm1,
            Clamped inputProj,
            Block[] blocks,
            Clamped outProj,
            F32FloatTensor outProjBias,
            Clamped mmProj) {
        this.dim = dim;
        this.heads = heads;
        this.headDim = dim / heads;
        this.ffDim = ffDim;
        this.nMel = nMel;
        this.outDim = outDim;
        this.eps = eps;
        this.preprocess = new AudioPreprocess(nMel);
        this.conv0 = conv0;
        this.norm0 = norm0;
        this.conv1 = conv1;
        this.norm1 = norm1;
        this.inputProj = inputProj;
        this.blocks = blocks;
        this.outProj = outProj;
        this.outProjBias = outProjBias;
        this.mmProj = mmProj;
    }

    // Sinusoidal RPE table for positions [PAST .. 0]: emb[p][i] = sin(pos/ts_i),
    // emb[p][i + dim/2] = cos(pos/ts_i) (clip.cpp's gemma4a input build).
    private static float[] buildPosEmb(int dim) {
        int half = dim / 2;
        float logInc = (float) (Math.log(10000.0) / Math.max(half - 1, 1));
        float[] emb = new float[RPE * dim];
        for (int p = 0; p < RPE; p++) {
            float position = PAST - p;
            for (int i = 0; i < half; i++) {
                float scaled = position * (float) Math.exp(-i * logInc);
                emb[p * dim + i] = (float) Math.sin(scaled);
                emb[p * dim + i + half] = (float) Math.cos(scaled);
            }
        }
        return emb;
    }

    @Override
    public void embed(Media.Audio audio, int maxChunkSize, Consumer<FloatTensor> sink) {
        try (Arena scratch = Arena.ofShared()) {
            sink.accept(encode(audio, scratch));
        }
    }

    /** Output tokens for {@code audio}: per 30 s chunk, mel frames through two stride-2 convs. */
    @Override
    public int positions(Media.Audio audio) {
        int monoLen = AudioPreprocess.mono16kLength(audio);
        int total = 0;
        for (int off = 0; off < monoLen; off += AudioPreprocess.CHUNK_SAMPLES) {
            int len = Math.min(AudioPreprocess.CHUNK_SAMPLES, monoLen - off);
            total += tokensForFrames(AudioPreprocess.framesFor(len));
        }
        return Math.max(1, total);
    }

    static int tokensForFrames(int frames) {
        int t2 = (frames - 1) / 2 + 1; // stride-2, pad-1, k=3: ceil(frames/2)
        return (t2 - 1) / 2 + 1;
    }

    /** Encode one clip: all chunks concatenated, rows [tokens x outDim], caller-owned copy. */
    public FloatTensor encode(Media.Audio audio) {
        try (Arena scratch = Arena.ofShared()) {
            FloatTensor rows = encode(audio, scratch);
            FloatTensor out = FloatTensor.allocateF32(Arena.ofAuto(), Math.toIntExact(rows.size()));
            rows.copyTo(0, out, 0, (int) rows.size());
            return out;
        }
    }

    private FloatTensor encode(Media.Audio audio, Arena scratch) {
        List<AudioPreprocess.MelChunk> mels = preprocess.logMel(audio);
        int totalTokens = 0;
        int maxFrames = 1;
        for (AudioPreprocess.MelChunk mel : mels) {
            totalTokens += tokensForFrames(mel.frames());
            maxFrames = Math.max(maxFrames, mel.frames());
        }
        FloatTensor rows =
                FloatTensor.allocateF32(scratch, Math.toIntExact((long) totalTokens * outDim));
        Scratch sc = Scratch.allocate(scratch, maxFrames, dim, ffDim, outDim, nMel);
        int at = 0;
        for (AudioPreprocess.MelChunk mel : mels) {
            int tokens = tokensForFrames(mel.frames());
            forward(mel, rows, (long) at * outDim, sc);
            at += tokens;
        }
        return rows;
    }

    // === the tower ===

    /** Per-encode scratch, sized once for the largest chunk and reused across chunks. */
    private record Scratch(
            float[] c0,
            float[] c1,
            FloatTensor flat,
            FloatTensor x,
            FloatTensor clampBuf,
            FloatTensor norm,
            FloatTensor ff,
            FloatTensor ffOut,
            FloatTensor qT,
            FloatTensor kT,
            FloatTensor vT,
            FloatTensor attnOut,
            FloatTensor pw,
            FloatTensor glu,
            FloatTensor proj,
            FloatTensor projOut) {
        static Scratch allocate(Arena a, int maxFrames, int dim, int ffDim, int outDim, int nMel) {
            int t2 = (maxFrames - 1) / 2 + 1;
            int n = tokensForFrames(maxFrames);
            return new Scratch(
                    new float[128 * t2 * (nMel / 2)],
                    new float[32 * n * (nMel / 4)],
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * ffDim),
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * ffDim),
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * dim * 2),
                    FloatTensor.allocateF32(a, n * dim),
                    FloatTensor.allocateF32(a, n * outDim),
                    FloatTensor.allocateF32(a, n * outDim));
        }
    }

    private void forward(
            AudioPreprocess.MelChunk mel, FloatTensor rowsOut, long outOff, Scratch sc) {
        int frames = mel.frames();
        int t2 = (frames - 1) / 2 + 1;
        int n = tokensForFrames(frames); // encoder positions

        // 1. subsampling: the mel chunk is already time-major [t][freq], conv2d's input layout
        float[] c0 = sc.c0();
        conv2d(mel.data(), frames, nMel, 1, conv0, 128, c0); // [128ch][t2][64]
        layerNormChannelsRelu(c0, 128, t2 * (nMel / 2), norm0);
        float[] c1 = sc.c1();
        conv2d(c0, t2, nMel / 2, 128, conv1, 32, c1); // [32ch][n][32]
        int f4 = nMel / 4;
        layerNormChannelsRelu(c1, 32, n * f4, norm1);

        // flatten [freq * ch] per timestep - ggml's permute(1,2,0,3) leaves CHANNELS fastest
        // (feature = f*32 + c), pinned by the node_34 trace anchor; then the input projection
        FloatTensor flat = sc.flat();
        for (int c = 0; c < 32; c++) {
            for (int t = 0; t < n; t++) {
                for (int f = 0; f < f4; f++) {
                    flat.setFloat((long) t * dim + f * 32 + c, c1[(c * n + t) * f4 + f]);
                }
            }
        }
        FloatTensor x = sc.x();
        inputProj.gemm(flat, dim, x, dim, n, sc.clampBuf());

        for (Block b : blocks) {
            halfFfn(x, b.ffNorm(), b.ffUp(), b.ffDown(), b.ffPostNorm(), n, sc);
            attention(x, b, n, sc);
            convModule(x, b, n, sc);
            halfFfn(x, b.ffNorm1(), b.ffUp1(), b.ffDown1(), b.ffPostNorm1(), n, sc);
            Parallel.forRows(
                    n, t -> Norms.rmsnorm(x, (long) t * dim, x, (long) t * dim, b.ln2(), dim, eps));
        }

        // 3. tail: out projection (+bias) -> weightless RMS -> mm projection
        FloatTensor proj = sc.proj();
        outProj.gemm(x, dim, proj, outDim, n, sc.clampBuf());
        Parallel.forRows(
                n,
                t -> {
                    long row = (long) t * outDim;
                    proj.addInPlace(row, outProjBias, 0, outDim);
                    Norms.rmsnormNoWeight(proj, row, proj, row, outDim, eps);
                });
        FloatTensor projOut = sc.projOut();
        mmProj.gemm(proj, outDim, projOut, outDim, n, sc.clampBuf());
        projOut.copyTo(0, rowsOut, outOff, Math.toIntExact((long) n * outDim));
    }

    /** 3x3 stride-2 pad-1 conv2d over [inCh][timeIn][freqIn] -> [outCh][ceil(t/2)][ceil(f/2)]. */
    private static void conv2d(
            float[] in, int timeIn, int freqIn, int inCh, float[] taps, int outCh, float[] out) {
        int timeOut = (timeIn - 1) / 2 + 1;
        int freqOut = (freqIn - 1) / 2 + 1;
        Parallel.parallelFor(
                0,
                outCh,
                oc -> {
                    for (int ot = 0; ot < timeOut; ot++) {
                        for (int of = 0; of < freqOut; of++) {
                            float sum = 0f;
                            for (int ic = 0; ic < inCh; ic++) {
                                int tapBase = ((oc * inCh + ic) * 3) * 3;
                                for (int ky = 0; ky < 3; ky++) {
                                    int it = 2 * ot - 1 + ky;
                                    if (it < 0 || it >= timeIn) continue;
                                    for (int kx = 0; kx < 3; kx++) {
                                        int ifr = 2 * of - 1 + kx;
                                        if (ifr < 0 || ifr >= freqIn) continue;
                                        sum +=
                                                taps[tapBase + ky * 3 + kx]
                                                        * in[(ic * timeIn + it) * freqIn + ifr];
                                    }
                                }
                            }
                            out[(oc * timeOut + ot) * freqOut + of] = sum;
                        }
                    }
                });
    }

    /** Per-position LayerNorm across channels (weight, no bias), then ReLU, in place. */
    private static void layerNormChannelsRelu(
            float[] x, int channels, int positions, F32FloatTensor weight) {
        Parallel.parallelFor(
                0,
                positions,
                p -> {
                    double mean = 0;
                    for (int c = 0; c < channels; c++) mean += x[c * positions + p];
                    mean /= channels;
                    double var = 0;
                    for (int c = 0; c < channels; c++) {
                        double d = x[c * positions + p] - mean;
                        var += d * d;
                    }
                    float inv = (float) (1.0 / Math.sqrt(var / channels + 1e-6));
                    for (int c = 0; c < channels; c++) {
                        float v =
                                (float) ((x[c * positions + p] - mean) * inv) * weight.getFloat(c);
                        x[c * positions + p] = Math.max(0f, v);
                    }
                });
    }

    /** Macaron half-step: x += 0.5 * postNorm(down(silu(up(rmsNorm(x))))). */
    private void halfFfn(
            FloatTensor x,
            F32FloatTensor preNorm,
            Clamped up,
            Clamped down,
            F32FloatTensor postNorm,
            int n,
            Scratch sc) {
        FloatTensor norm = sc.norm();
        FloatTensor ff = sc.ff();
        FloatTensor ffOut = sc.ffOut();
        Parallel.forRows(
                n, t -> Norms.rmsnorm(norm, (long) t * dim, x, (long) t * dim, preNorm, dim, eps));
        up.gemm(norm, dim, ff, ffDim, n, sc.clampBuf());
        Parallel.forRows(n, t -> ff.siluInPlace((long) t * ffDim, ffDim));
        down.gemm(ff, ffDim, ffOut, dim, n, sc.clampBuf());
        Parallel.forRows(
                n,
                t -> {
                    long row = (long) t * dim;
                    Norms.rmsnorm(ffOut, row, ffOut, row, postNorm, dim, eps);
                    x.saxpyInPlace(row, ffOut, row, dim, 0.5f);
                });
    }

    /** Chunked local attention with sinusoidal RPE; full residual onto x. */
    private void attention(FloatTensor x, Block b, int n, Scratch sc) {
        FloatTensor norm = sc.norm();
        FloatTensor qT = sc.qT();
        FloatTensor kT = sc.kT();
        FloatTensor vT = sc.vT();
        FloatTensor attnOut = sc.attnOut();
        Parallel.forRows(
                n,
                t ->
                        Norms.rmsnorm(
                                norm,
                                (long) t * dim,
                                x,
                                (long) t * dim,
                                b.attnPreNorm(),
                                dim,
                                eps));
        b.q().gemm(norm, dim, qT, dim, n, sc.clampBuf());
        b.k().gemm(norm, dim, kT, dim, n, sc.clampBuf());
        b.v().gemm(norm, dim, vT, dim, n, sc.clampBuf());

        float qScale = (float) ((1.0 / Math.sqrt(headDim)) / Math.log(2.0)); // (1/sqrt(d))/ln2
        float kScale = (float) (Math.log1p(Math.exp(1.0)) / Math.log(2.0)); // softplus(1)/ln2
        float softcap = 50.0f;

        // scale Q (with per-dim scale) and K in place
        Parallel.forRows(
                n,
                t -> {
                    long row = (long) t * dim;
                    for (int h = 0; h < heads; h++) {
                        for (int d = 0; d < headDim; d++) {
                            int lane = h * headDim + d;
                            qT.setFloat(
                                    row + lane,
                                    qT.getFloat(row + lane) * qScale * b.perDimScale().getFloat(d));
                            kT.setFloat(row + lane, kT.getFloat(row + lane) * kScale);
                        }
                    }
                });

        F32FloatTensor rel = b.rel(); // kRel @ posEmb, precomputed at load
        int numBlocks = (n + CHUNK - 1) / CHUNK;
        Parallel.parallelFor(
                0,
                numBlocks * heads,
                unit -> {
                    int blk = unit / heads;
                    int h = unit % heads;
                    int hBase = h * headDim;
                    float[] scores = new float[CONTEXT];
                    for (int c = 0; c < CHUNK; c++) {
                        int gq = blk * CHUNK + c;
                        if (gq >= n) break;
                        long qRow = (long) gq * dim + hBase;
                        for (int s = 0; s < CONTEXT; s++) {
                            int gk = blk * CHUNK - PAST + s;
                            boolean valid = gk >= 0 && gk < n && gk <= gq && (gq - gk) < PAST;
                            if (!valid) {
                                scores[s] = -1e9f;
                                continue;
                            }
                            float ac = qT.dot(qRow, kT, (long) gk * dim + hBase, headDim);
                            int p = s - c;
                            float bd =
                                    p >= 0 && p < RPE
                                            ? qT.dot(qRow, rel, (long) p * dim + hBase, headDim)
                                            : 0f;
                            scores[s] = softcap * FastMath.tanh((ac + bd) / softcap);
                        }
                        // softmax over the context slots, normalized in place
                        float max = Float.NEGATIVE_INFINITY;
                        for (int s = 0; s < CONTEXT; s++) max = Math.max(max, scores[s]);
                        float sum = 0f;
                        for (int s = 0; s < CONTEXT; s++) {
                            scores[s] = FastMath.expNeg(scores[s] - max);
                            sum += scores[s];
                        }
                        float inv = 1f / sum;
                        long outRow = (long) gq * dim + hBase;
                        attnOut.fillInPlace(outRow, headDim, 0f);
                        for (int s = 0; s < CONTEXT; s++) {
                            int gk = blk * CHUNK - PAST + s;
                            if (gk < 0 || gk >= n || scores[s] == 0f) continue;
                            attnOut.saxpyInPlace(
                                    outRow, vT, (long) gk * dim + hBase, headDim, scores[s] * inv);
                        }
                    }
                });

        // o projection, post-norm, full residual
        b.o().gemm(attnOut, dim, norm, dim, n, sc.clampBuf());
        Parallel.forRows(
                n,
                t -> {
                    long row = (long) t * dim;
                    Norms.rmsnorm(norm, row, norm, row, b.attnPostNorm(), dim, eps);
                    x.addInPlace(row, norm, row, dim);
                });
    }

    /** Conv module: rms -> pw1 -> GLU -> causal depthwise k=5 -> rms -> SiLU -> pw2; residual. */
    private void convModule(FloatTensor x, Block b, int n, Scratch sc) {
        FloatTensor norm = sc.norm();
        FloatTensor pw = sc.pw();
        FloatTensor glu = sc.glu();
        Parallel.forRows(
                n,
                t ->
                        Norms.rmsnorm(
                                norm,
                                (long) t * dim,
                                x,
                                (long) t * dim,
                                b.convPreNorm(),
                                dim,
                                eps));
        b.convPw1().gemm(norm, dim, pw, dim * 2, n, sc.clampBuf());
        Parallel.forRows(
                n,
                t -> {
                    long src = (long) t * dim * 2;
                    long dst = (long) t * dim;
                    for (int d = 0; d < dim; d++) {
                        float gate = FastMath.sigmoid(pw.getFloat(src + dim + d));
                        glu.setFloat(dst + d, pw.getFloat(src + d) * gate);
                    }
                });
        // causal depthwise conv over time: out[t][c] = sum_k w[c][k] * glu[t-4+k][c]
        F32FloatTensor dw = b.convDw();
        Parallel.forRows(
                n,
                t -> {
                    long dst = (long) t * dim;
                    for (int c = 0; c < dim; c++) {
                        float sum = 0f;
                        for (int k = 0; k < 5; k++) {
                            int src = t - 4 + k;
                            if (src < 0) continue;
                            sum += dw.getFloat(c * 5 + k) * glu.getFloat((long) src * dim + c);
                        }
                        norm.setFloat(dst + c, sum);
                    }
                });
        Parallel.forRows(
                n,
                t -> {
                    long row = (long) t * dim;
                    Norms.rmsnorm(norm, row, norm, row, b.convPostNorm(), dim, eps);
                    norm.siluInPlace(row, dim);
                });
        b.convPw2().gemm(norm, dim, glu, dim, n, sc.clampBuf());
        Parallel.forRows(
                n,
                t -> {
                    long row = (long) t * dim;
                    x.addInPlace(row, glu, row, dim);
                });
    }

    // === loader ===

    // === loader ===

    public static Gemma4Conformer loadModel(Path mmprojPath, Arena arena) throws IOException {
        try (FileChannel fc = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            var gguf = ModelLoader.readGguf(fc, mmprojPath.toString());
            return loadModel(mmprojPath, gguf, ModelLoader.loadTensors(fc, gguf, arena), arena);
        }
    }

    /**
     * As {@link #loadModel(Path, Arena)} over an mmproj ALREADY parsed and mapped: one header read
     * and one mapping serve both towers of a sidecar carrying vision AND audio. {@code mmprojPath}
     * is a label for messages here - nothing is read from it.
     */
    public static Gemma4Conformer loadModel(
            Path mmprojPath, GGUF gguf, Map<String, GGMLTensorEntry> t, Arena arena)
            throws IOException {
        {
            int dim = gguf.getValue(int.class, "clip.audio.embedding_length");
            int heads = gguf.getValue(int.class, "clip.audio.attention.head_count");
            int ffDim = gguf.getValue(int.class, "clip.audio.feed_forward_length");
            int nBlocks = gguf.getValue(int.class, "clip.audio.block_count");
            int nMel = gguf.getValue(int.class, "clip.audio.num_mel_bins");
            int outDim = gguf.getValue(int.class, "clip.audio.projection_dim");
            float eps =
                    gguf.getValueOrDefault(
                            float.class, "clip.audio.attention.layer_norm_epsilon", 1e-6f);
            validateArchitecture(mmprojPath, dim, heads, nMel);

            // the sinusoidal RPE table projected through each block's kRel, once at load - the
            // forward pass then reads a constant
            FloatTensor posRows = FloatTensor.allocateF32(arena, RPE * dim);
            float[] posEmb = buildPosEmb(dim);
            for (int i = 0; i < RPE * dim; i++) posRows.setFloat(i, posEmb[i]);

            Block[] blocks = new Block[nBlocks];
            for (int i = 0; i < nBlocks; i++) {
                String p = "a.blk." + i + ".";
                blocks[i] =
                        new Block(
                                f32(t, p + "ffn_norm.weight", dim),
                                Clamped.load(t, p + "ffn_up", (long) dim * ffDim),
                                Clamped.load(t, p + "ffn_down", (long) dim * ffDim),
                                f32(t, p + "ffn_post_norm.weight", dim),
                                f32(t, p + "attn_pre_norm.weight", dim),
                                Clamped.load(t, p + "attn_q", (long) dim * dim),
                                Clamped.load(t, p + "attn_k", (long) dim * dim),
                                Clamped.load(t, p + "attn_v", (long) dim * dim),
                                Clamped.load(t, p + "attn_out", (long) dim * dim),
                                f32(t, p + "attn_post_norm.weight", dim),
                                relProjection(
                                        w(t, p + "attn_k_rel.weight", (long) dim * dim),
                                        posRows,
                                        dim,
                                        arena),
                                f32(t, p + "per_dim_scale.weight", dim / heads),
                                f32(t, p + "conv_norm.weight", dim),
                                Clamped.load(t, p + "conv_pw1", (long) dim * dim * 2),
                                f32(t, p + "conv_dw.weight", dim * 5),
                                f32(t, p + "norm_conv.weight", dim),
                                Clamped.load(t, p + "conv_pw2", (long) dim * dim),
                                f32(t, p + "ffn_norm_1.weight", dim),
                                Clamped.load(t, p + "ffn_up_1", (long) dim * ffDim),
                                Clamped.load(t, p + "ffn_down_1", (long) dim * ffDim),
                                f32(t, p + "ffn_post_norm_1.weight", dim),
                                f32(t, p + "ln2.weight", dim));
            }
            return new Gemma4Conformer(
                    dim,
                    heads,
                    ffDim,
                    nMel,
                    outDim,
                    eps,
                    taps(t, "a.conv1d.0.weight", 3 * 3 * 1 * 128),
                    f32(t, "a.conv1d.0.norm.weight", 128),
                    taps(t, "a.conv1d.1.weight", 3 * 3 * 128 * 32),
                    f32(t, "a.conv1d.1.norm.weight", 32),
                    Clamped.load(t, "a.input_projection", (long) dim * dim),
                    blocks,
                    Clamped.load(t, "a.pre_encode.out", (long) dim * outDim),
                    f32(t, "a.pre_encode.out.bias", outDim),
                    Clamped.load(t, "mm.a.input_projection", (long) outDim * outDim));
        }
    }

    /**
     * The geometry this port implements (the E2B/E4B family): the subsampling ladder (1 -> 128 ->
     * 32 channels over two stride-2 convs) must flatten to exactly the encoder width, heads must
     * divide it, and the RPE sinusoid needs it even. A gemma4a variant violating any of these would
     * otherwise produce silently wrong rows (a partial flatten still feeds shape-compatible GEMMs),
     * so it refuses loudly instead. Package-visible for direct testing.
     */
    static void validateArchitecture(Path mmproj, int dim, int heads, int nMel) {
        int flattened = (nMel / 4) * 32; // freq/4 after two stride-2 convs, 32 output channels
        if (flattened != dim || dim % heads != 0) {
            throw new IllegalArgumentException(
                    "'"
                            + mmproj.getFileName()
                            + "' declares a gemma4a geometry this port does not implement (mel"
                            + " bins "
                            + nMel
                            + " -> flattened "
                            + flattened
                            + ", encoder width "
                            + dim
                            + ", heads "
                            + heads
                            + ") - supported is the E2B/E4B family shape (128 mels, width 1024,"
                            + " heads dividing the width); a newer jinfer may support this"
                            + " variant");
        }
    }

    private static F32FloatTensor relProjection(
            FloatTensor kRel, FloatTensor posRows, int dim, Arena arena) {
        FloatTensor rel = FloatTensor.allocateF32(arena, RPE * dim);
        kRel.gemm(posRows, dim, rel, dim, RPE, dim, dim);
        return (F32FloatTensor) rel;
    }

    private static FloatTensor w(Map<String, GGMLTensorEntry> t, String name, long expected) {
        return Clamped.require(t, name, expected);
    }

    private static F32FloatTensor f32(Map<String, GGMLTensorEntry> t, String name, long expected) {
        FloatTensor w = w(t, name, expected);
        if (!(w instanceof F32FloatTensor f)) {
            throw new IllegalStateException(name + ": expected F32, got " + w.getClass());
        }
        return f;
    }

    private static float[] taps(Map<String, GGMLTensorEntry> t, String name, int expected) {
        float[] out = new float[expected];
        w(t, name, expected).copyRow(0, out, 0, expected);
        return out;
    }
}
