// Gemma 4 vision encoder (gemma4v SigLIP-style ViT) against the new com.qxotic.jinfer.models API,
// with a
// BATCHED
// prefill: all patches flow through the 16 transformer blocks as one GEMM-batched pass (vs the
// per-patch
// float[][] reference in gemma4.java/Gemma4Vision.java). Backs Gemma4's MultiModal
// Embedder<Media.Image>.
//
// Architecture (reverse-engineered from the working reference + llama.cpp clip_graph_gemma4v):
//   preprocess: resize→[0,1] HWC, scale_bias px*2-1, patchify (patch=16) -> nPatches=(W/16)*(H/16)
//   patch embed: token = patchEmbd @ patchPixels  (conv-as-matmul, [visionDim, 3*16*16])
//   + factorized 2D position: token += posX[px] + posY[py]   (posEmbd [visionDim, posSize, 2])
//   tower x16 (Gemma sandwich norm, RMSNorm eps 1e-6):
//     cur += postNorm(attnPost, attn( rms(cur,ln1) ));  cur += postNorm(ffnPost, ffn( rms(cur,ln2)
// ))
//   attention: q/k/v = W@x (clamped); per-head RMSNorm on q(qNorm)/k(kNorm)/v(no-weight);
//     2D NeoX RoPE theta=100 (x on dims[0:hd/2], y on [hd/2:hd]); BIDIRECTIONAL full attn, NO
// 1/sqrt scale; out=Wo@.
//   ffn: GeGLU with gelu_quick: down( geluQuick(gate@x) * (up@x) ),  ffnDim=3072
//   pool: merge×merge avg over the patch grid, * sqrt(visionDim)
//   project: clamp -> mm.input_projection (visionDim->modelDim) -> clamp -> RMSNorm(no weight)
// NOTE: activation clamps come from the per-tensor calibration tensors (input_max/min,
// output_max/min);
//       required for parity with the quantized reference. This file loads F32 weights.
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Embedder;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FlashAttention;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.Norms;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.kernels.*;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Batched gemma4v vision tower → projected model-dim rows. Implements {@link Embedder} over images.
 */
public final class Gemma4Vision implements Embedder<Media.Image>, VisionBudget {

    // --- config (from clip.vision.* metadata) ---
    final int imageSize,
            patchSize,
            visionDim,
            nHead,
            headDim,
            nLayer,
            ffnDim,
            modelDim,
            merge,
            posSize;
    final float normEps, ropeTheta;
    // --- weights ---
    final FloatTensor patchEmbd; // [visionDim, 3*patch*patch]  (conv as matmul)
    final FloatTensor posEmbd; // [visionDim, posSize, 2] flattened (x-table then y-table)
    final Clamped mmProj; // [modelDim, visionDim], with calibration clamps
    final Layer[] layers;

    record Layer(
            F32FloatTensor ln1,
            F32FloatTensor ln2,
            F32FloatTensor attnPostNorm,
            F32FloatTensor ffnPostNorm,
            F32FloatTensor qNorm,
            F32FloatTensor kNorm,
            Clamped wq,
            Clamped wk,
            Clamped wv,
            Clamped wo,
            Clamped ffnGate,
            Clamped ffnUp,
            Clamped ffnDown) {}

    Gemma4Vision(
            int imageSize,
            int patchSize,
            int visionDim,
            int nHead,
            int nLayer,
            int ffnDim,
            int modelDim,
            int merge,
            int posSize,
            float normEps,
            float ropeTheta,
            FloatTensor patchEmbd,
            FloatTensor posEmbd,
            Clamped mmProj,
            Layer[] layers) {
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.visionDim = visionDim;
        this.nHead = nHead;
        this.headDim = visionDim / nHead;
        this.nLayer = nLayer;
        this.ffnDim = ffnDim;
        this.modelDim = modelDim;
        this.merge = Math.max(1, merge);
        this.posSize = posSize;
        this.normEps = normEps;
        this.ropeTheta = ropeTheta;
        this.patchEmbd = patchEmbd;
        this.posEmbd = posEmbd;
        this.mmProj = mmProj;
        this.layers = layers;
    }

    // === Embedder seam ===

    @Override
    public void embed(Media.Image image, int maxChunkSize, Consumer<FloatTensor> sink) {
        // Everything this encode touches (~340 MB/image at defaults), the projected rows
        // included, lives in one owned arena freed on exit - per-call ofAuto here was the 51GB
        // pathology for vision workloads (GC never runs in a native-heavy JVM). NOTHING escapes:
        // the sink sees an ephemeral view (the Embedder contract) and must copy what it keeps.
        try (Arena scratch = Arena.ofShared()) {
            sink.accept(encode(image, scratch)); // [nTokens, modelDim]
        }
    }

    /**
     * Encode one image → projected rows (nTokens × modelDim), all patches batched through the
     * tower, returned as a caller-owned GC-managed copy (the {@link #embed} seam skips the copy:
     * its rows die with the per-encode scratch). Contract: at most one encode at a time per
     * pipeline (the state's serial-pipeline law covers its media encodes); the tower itself is
     * stateless - every mutable buffer is per-encode scratch - so many pipelines may share it.
     */
    public FloatTensor encode(Media.Image image) {
        return encode(image, VisionPreprocess.budget(280));
    }

    /** The per-call budget entry: video frames ride this tower at the video budget. */
    public FloatTensor encode(Media.Image image, int budgetTokens) {
        try (Arena scratch = Arena.ofShared()) {
            FloatTensor rows = encode(image, scratch, budgetTokens);
            FloatTensor out = FloatTensor.allocateF32(Arena.ofAuto(), (int) rows.size());
            rows.copyTo(0, out, 0, (int) rows.size());
            return out;
        }
    }

    private FloatTensor encode(Media.Image image, Arena scratch) {
        return encode(image, scratch, VisionPreprocess.budget(280));
    }

    private FloatTensor encode(Media.Image image, Arena scratch, int budgetTokens) {
        // 1. preprocess + patch-embed (+ 2D position) → patches: [nPatch, visionDim]
        Patches p = patchify(image, scratch, budgetTokens);
        int n = p.count;
        FloatTensor cur = p.tokens; // [n, visionDim]
        FloatTensor tmp = FloatTensor.allocateF32(scratch, n * visionDim); // rmsAddResidual
        FloatTensor clampTmp =
                FloatTensor.allocateF32(scratch, n * ffnDim); // Clamped.gemm input scratch
        FloatTensor xb = FloatTensor.allocateF32(scratch, n * visionDim);
        FloatTensor q = FloatTensor.allocateF32(scratch, n * visionDim),
                k = FloatTensor.allocateF32(scratch, n * visionDim),
                v = FloatTensor.allocateF32(scratch, n * visionDim);
        FloatTensor attn = FloatTensor.allocateF32(scratch, n * visionDim);
        FloatTensor g = FloatTensor.allocateF32(scratch, n * ffnDim),
                u = FloatTensor.allocateF32(scratch, n * ffnDim);
        // K/V kept in F16 (like the LLM KV cache) for the rolling flash attention (no materialized
        // scores)
        FloatTensor kF16 = FloatTensor.allocateF16(scratch, n, visionDim),
                vF16 = FloatTensor.allocateF16(scratch, n, visionDim);

        // 2. tower (batched)
        for (Layer l : layers) {
            rms(xb, cur, l.ln1(), n, visionDim); // n1 = rms(cur, ln1)
            attention(xb, q, k, v, attn, kF16, vF16, l, p.px, p.py, n, clampTmp);
            rmsAddResidual(cur, attn, l.attnPostNorm(), n, visionDim, tmp);
            rms(xb, cur, l.ln2(), n, visionDim); // n2 = rms(cur, ln2)
            feedForward(xb, g, u, attn, l, n, clampTmp); // ffn(n2) -> attn (reused)
            rmsAddResidual(cur, attn, l.ffnPostNorm(), n, visionDim, tmp);
        }

        // 3. pool (merge x merge avg * sqrt(visionDim)) then project + final RMSNorm
        return projectPooled(cur, p.px, p.py, scratch, clampTmp);
    }

    // === batched tower steps ===

    private void attention(
            FloatTensor x,
            FloatTensor q,
            FloatTensor k,
            FloatTensor v,
            FloatTensor out,
            FloatTensor kF16,
            FloatTensor vF16,
            Layer l,
            int px,
            int py,
            int n,
            FloatTensor clampTmp) {
        l.wq().gemm(x, visionDim, q, visionDim, n, clampTmp);
        l.wk().gemm(x, visionDim, k, visionDim, n, clampTmp);
        l.wv().gemm(x, visionDim, v, visionDim, n, clampTmp);
        // per-head RMS norms (q,k with weight; v no weight) + 2D RoPE on q,k
        Parallel.forRows(
                n,
                t -> {
                    for (int h = 0; h < nHead; h++) {
                        int off = t * visionDim + h * headDim;
                        headRms(q, off, l.qNorm());
                        headRms(k, off, l.kNorm());
                        headRms(v, off, null);
                    }
                    int posX = t % px, posY = t / px;
                    for (int h = 0; h < nHead; h++) {
                        int base = t * visionDim + h * headDim;
                        rope2d(q, base, posX, posY);
                        rope2d(k, base, posX, posY);
                    }
                });
        // K/V → F16, then rolling BIDIRECTIONAL flash attention: online softmax, no materialized
        // [n,n] score
        // matrix (the memory-bound bottleneck), no 1/√d scale (scale=1, matches the gemma4v
        // reference).
        k.copyTo(0, kF16, 0, n * visionDim);
        v.copyTo(0, vF16, 0, n * visionDim);
        FlashAttention.bidirectionalPrefill(
                q, out, kF16, vF16, nHead, n, headDim, visionDim, visionDim, 1, 1.0f);
        // output projection in place: x <- Wo @ out  (reuse x as scratch is unsafe; write to q)
        l.wo().gemm(out, visionDim, q, visionDim, n, clampTmp);
        q.copyTo(0, out, 0, n * visionDim);
    }

    private void feedForward(
            FloatTensor x,
            FloatTensor g,
            FloatTensor u,
            FloatTensor out,
            Layer l,
            int n,
            FloatTensor clampTmp) {
        l.ffnGate().gemm(x, visionDim, g, ffnDim, n, clampTmp);
        l.ffnUp().gemm(x, visionDim, u, ffnDim, n, clampTmp);
        Parallel.forRows(
                n,
                t -> { // gelu_quick(gate)*up
                    int base = t * ffnDim;
                    for (int d = 0; d < ffnDim; d++) {
                        float gg = g.getFloat(base + d);
                        float gq = gg / (1f + (float) Math.exp(-1.702f * gg));
                        g.setFloat(base + d, gq * u.getFloat(base + d));
                    }
                });
        l.ffnDown().gemm(g, ffnDim, out, visionDim, n, clampTmp);
    }

    private FloatTensor projectPooled(
            FloatTensor cur, int px, int py, Arena scratch, FloatTensor clampTmp) {
        int outX = Math.max(1, px / merge), outY = Math.max(1, py / merge);
        int nTok = outX * outY;
        float scale = (float) Math.sqrt(visionDim);
        FloatTensor pooled = FloatTensor.allocateF32(scratch, nTok * visionDim);
        for (int oy = 0; oy < outY; oy++)
            for (int ox = 0; ox < outX; ox++) {
                int dst = (oy * outX + ox) * visionDim, cnt = 0;
                for (int my = 0; my < merge; my++) {
                    int p = oy * merge + my;
                    if (p >= py) continue;
                    for (int mx = 0; mx < merge; mx++) {
                        int q = ox * merge + mx;
                        if (q >= px) continue;
                        int src = (p * px + q) * visionDim;
                        for (int d = 0; d < visionDim; d++)
                            pooled.setFloat(
                                    dst + d, pooled.getFloat(dst + d) + cur.getFloat(src + d));
                        cnt++;
                    }
                }
                float inv = cnt > 0 ? scale / cnt : scale;
                for (int d = 0; d < visionDim; d++)
                    pooled.setFloat(dst + d, pooled.getFloat(dst + d) * inv);
            }
        // mm_soft_emb_norm: RMSNorm the pooled 768-dim features BEFORE the projection (llama.cpp
        // order) —
        // confirmed by the embedding diff: this bounds the output range (max ~4.6) to match
        // llama.cpp,
        // vs project→RMSNorm which let outliers to ~7.4 and drowned small-object features.
        Parallel.forRows(nTok, t -> rmsNoWeight(pooled, t * visionDim, visionDim));
        FloatTensor projected = FloatTensor.allocateF32(scratch, nTok * modelDim);
        mmProj.gemm(pooled, visionDim, projected, modelDim, nTok, clampTmp);
        return projected;
    }

    // === norms / rope helpers ===

    private void rms(FloatTensor out, FloatTensor x, F32FloatTensor w, int n, int dim) {
        Parallel.forRows(
                n, t -> Norms.rmsnorm(out, (long) t * dim, x, (long) t * dim, w, dim, normEps));
    }

    private void rmsAddResidual(
            FloatTensor residual,
            FloatTensor x,
            F32FloatTensor w,
            int n,
            int dim,
            FloatTensor tmp) {
        Parallel.forRows(
                n,
                t -> {
                    Norms.rmsnorm(tmp, (long) t * dim, x, (long) t * dim, w, dim, normEps);
                    residual.addInPlace((long) t * dim, tmp, (long) t * dim, dim);
                });
    }

    private void headRms(FloatTensor x, int off, F32FloatTensor w) {
        float ss = 0f;
        for (int d = 0; d < headDim; d++) {
            float vv = x.getFloat(off + d);
            ss += vv * vv;
        }
        float inv = (float) (1.0 / Math.sqrt(ss / headDim + normEps));
        for (int d = 0; d < headDim; d++)
            x.setFloat(off + d, x.getFloat(off + d) * inv * (w == null ? 1f : w.getFloat(d)));
    }

    private void rmsNoWeight(FloatTensor x, int off, int dim) {
        float ss = 0f;
        for (int d = 0; d < dim; d++) {
            float vv = x.getFloat(off + d);
            ss += vv * vv;
        }
        float inv = (float) (1.0 / Math.sqrt(ss / dim + normEps));
        for (int d = 0; d < dim; d++) x.setFloat(off + d, x.getFloat(off + d) * inv);
    }

    private void rope2d(FloatTensor x, int base, int posX, int posY) {
        int halfHead = headDim / 2, ropePairs = halfHead / 2;
        rotate(x, base, posX, ropePairs, halfHead);
        rotate(x, base + halfHead, posY, ropePairs, halfHead);
    }

    private void rotate(FloatTensor x, int base, int pos, int ropePairs, int ropeDim) {
        for (int i = 0; i < ropePairs; i++) {
            int d0 = base + i, d1 = base + i + ropePairs;
            float v0 = x.getFloat(d0), v1 = x.getFloat(d1);
            float invFreq = (float) Math.pow(ropeTheta, -(2.0 * i) / ropeDim);
            float a = pos * invFreq, c = (float) Math.cos(a), s = (float) Math.sin(a);
            x.setFloat(d0, v0 * c - v1 * s);
            x.setFloat(d1, v0 * s + v1 * c);
        }
    }

    // === preprocess + patch embed (resize/im2col live in VisionPreprocess) ===

    private record Patches(FloatTensor tokens, int count, int px, int py) {}

    /** The plan's row count for {@code image} - patchify's grid arithmetic, no tower run. */
    @Override
    public int positions(Media.Image image) {
        int ps = patchSize, curMerge = Math.max(1, merge), factor = ps * curMerge, tw, th;
        if (VisionPreprocess.SMART_RESIZE) {
            int maxPixels = VisionPreprocess.budget(280) * factor * factor;
            int[] wh =
                    VisionPreprocess.smartResize(
                            image.width(), image.height(), factor, factor * factor, maxPixels);
            tw = wh[0];
            th = wh[1];
        } else {
            tw = th = imageSize;
        }
        int px = tw / ps, py = th / ps;
        return Math.max(1, px / curMerge) * Math.max(1, py / curMerge);
    }

    private Patches patchify(Media.Image image, Arena scratch, int budgetTokens) {
        int ps = patchSize, factor = ps * Math.max(1, merge), tw, th;
        if (VisionPreprocess.SMART_RESIZE) {
            int maxPixels = budgetTokens * factor * factor, minPixels = factor * factor;
            int[] wh =
                    VisionPreprocess.smartResize(
                            image.width(), image.height(), factor, minPixels, maxPixels);
            tw = wh[0];
            th = wh[1];
        } else {
            tw = th = imageSize; // fixed procSize square (gemma4.java-reference parity)
        }
        int px = tw / ps, py = th / ps, n = px * py, patchVec = 3 * ps * ps;
        FloatTensor flat = VisionPreprocess.im2col(image, tw, th, ps, scratch);
        FloatTensor tokens = FloatTensor.allocateF32(scratch, n * visionDim);
        patchEmbd.gemm(flat, patchVec, tokens, visionDim, n, visionDim, patchVec); // conv as matmul
        for (int gy = 0; gy < py; gy++)
            for (int gx = 0; gx < px; gx++) { // + factorized 2D position
                int tok = (gy * px + gx) * visionDim,
                        xb = visionDim * gx,
                        yb = visionDim * (gy + posSize);
                for (int d = 0; d < visionDim; d++)
                    tokens.setFloat(
                            tok + d,
                            tokens.getFloat(tok + d)
                                    + posEmbd.getFloat(xb + d)
                                    + posEmbd.getFloat(yb + d));
            }
        return new Patches(tokens, n, px, py);
    }

    // === loader ===

    /**
     * Metadata/tensor agreement for the conv patch embedding: it must be [visionDim, 3*patch*patch]
     * or the patchify silently mis-embeds. Shared by both vision towers.
     */
    static FloatTensor checkPatchEmbd(
            Path mmproj, FloatTensor patchEmbd, int visionDim, int patchSize) {
        long expected = (long) visionDim * 3 * patchSize * patchSize;
        if (patchEmbd.size() != expected) {
            throw new IllegalArgumentException(
                    "'"
                            + mmproj.getFileName()
                            + "': v.patch_embd has "
                            + patchEmbd.size()
                            + " elements but patch size "
                            + patchSize
                            + " and embedding_length "
                            + visionDim
                            + " imply "
                            + expected
                            + " - the sidecar's metadata and tensors disagree");
        }
        return patchEmbd;
    }

    public static Gemma4Vision loadModel(Path mmprojPath, Arena arena) throws IOException {
        try (FileChannel fc = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            var gguf = ModelLoader.readGguf(fc, mmprojPath.toString());
            Map<String, GGMLTensorEntry> t = ModelLoader.loadTensors(fc, gguf, arena);
            int imageSize = gguf.getValueOrDefault(int.class, "clip.vision.image_size", 224);
            int patchSize = gguf.getValueOrDefault(int.class, "clip.vision.patch_size", 16);
            int visionDim = gguf.getValueOrDefault(int.class, "clip.vision.embedding_length", 768);
            int nHead = gguf.getValueOrDefault(int.class, "clip.vision.attention.head_count", 12);
            int nLayer = gguf.getValueOrDefault(int.class, "clip.vision.block_count", 16);
            int ffnDim = gguf.getValueOrDefault(int.class, "clip.vision.feed_forward_length", 3072);
            int merge = gguf.getValueOrDefault(int.class, "clip.vision.proj_scale_factor", 3);
            // smartResize canonical processing size: a multiple of (patch*merge) whose area fits
            // image_max_pixels
            // (default 280*(patch*merge)^2). For patch16/merge3 -> 768 -> 48x48 patches -> merge ->
            // 16x16 = 256 tokens.
            int curMerge = Math.max(1, merge), factor = patchSize * curMerge;
            int maxPixels =
                    VisionPreprocess.IMAGE_TOKEN_BUDGET > 0
                            ? VisionPreprocess.IMAGE_TOKEN_BUDGET * factor * factor
                            : gguf.getValueOrDefault(
                                    int.class,
                                    "clip.vision.image_max_pixels",
                                    280 * factor * factor);
            int procSize = (int) (Math.sqrt(maxPixels) / factor) * factor;
            int modelDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", 1536);
            float eps =
                    gguf.getValueOrDefault(
                            float.class, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
            // the geometry invariants a variant file would break silently: heads must divide
            // the width, the patch embedding must match [visionDim, 3*patch*patch], and the
            // position table must split evenly - each otherwise mis-embeds without an error
            if (visionDim % nHead != 0) {
                throw new IllegalArgumentException(
                        "'"
                                + mmprojPath.getFileName()
                                + "': vision head_count "
                                + nHead
                                + " does not divide embedding_length "
                                + visionDim
                                + " - unsupported gemma4v geometry");
            }
            FloatTensor patchEmbd =
                    checkPatchEmbd(
                            mmprojPath,
                            ModelLoader.loadQuantized(t.get("v.patch_embd.weight")),
                            visionDim,
                            patchSize);
            FloatTensor posEmbd = ModelLoader.loadQuantized(t.get("v.position_embd.weight"));
            if (posEmbd.size() % (visionDim * 2L) != 0) {
                throw new IllegalArgumentException(
                        "'"
                                + mmprojPath.getFileName()
                                + "': v.position_embd has "
                                + posEmbd.size()
                                + " elements, not a multiple of 2*embedding_length - the"
                                + " sidecar's metadata and tensors disagree");
            }
            int posSize = (int) (posEmbd.size() / (visionDim * 2L));
            Clamped mmProj = Clamped.load(t, "mm.input_projection", (long) visionDim * modelDim);

            Layer[] layers = new Layer[nLayer];
            for (int i = 0; i < nLayer; i++) {
                String p = "v.blk." + i + ".";
                layers[i] =
                        new Layer(
                                ModelLoader.f32OrNull(t, p + "ln1.weight"),
                                ModelLoader.f32OrNull(t, p + "ln2.weight"),
                                ModelLoader.f32OrNull(t, p + "attn_post_norm.weight"),
                                ModelLoader.f32OrNull(t, p + "ffn_post_norm.weight"),
                                ModelLoader.f32OrNull(t, p + "attn_q_norm.weight"),
                                ModelLoader.f32OrNull(t, p + "attn_k_norm.weight"),
                                Clamped.load(t, p + "attn_q", (long) visionDim * visionDim),
                                Clamped.load(t, p + "attn_k", (long) visionDim * visionDim),
                                Clamped.load(t, p + "attn_v", (long) visionDim * visionDim),
                                Clamped.load(t, p + "attn_out", (long) visionDim * visionDim),
                                Clamped.load(t, p + "ffn_gate", (long) visionDim * ffnDim),
                                Clamped.load(t, p + "ffn_up", (long) visionDim * ffnDim),
                                Clamped.load(t, p + "ffn_down", (long) visionDim * ffnDim));
            }
            return new Gemma4Vision(
                    procSize, patchSize, visionDim, nHead, nLayer, ffnDim, modelDim, merge, posSize,
                    eps, 100.0f, patchEmbd, posEmbd, mmProj, layers);
        }
    }
}
