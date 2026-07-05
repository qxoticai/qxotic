// Gemma 4 vision encoder (gemma4v SigLIP-style ViT) against the new com.qxotic.llm API, with a BATCHED
// prefill: all patches flow through the 16 transformer blocks as one GEMM-batched pass (vs the per-patch
// float[][] reference in gemma4.java/Gemma4Vision.java). Backs Gemma4's MultiModal Embedder<Media.Image>.
//
// Architecture (reverse-engineered from the working reference + llama.cpp clip_graph_gemma4v):
//   preprocess: resize→[0,1] HWC, scale_bias px*2-1, patchify (patch=16) -> nPatches=(W/16)*(H/16)
//   patch embed: token = patchEmbd @ patchPixels  (conv-as-matmul, [visionDim, 3*16*16])
//   + factorized 2D position: token += posX[px] + posY[py]   (posEmbd [visionDim, posSize, 2])
//   tower x16 (Gemma sandwich norm, RMSNorm eps 1e-6):
//     cur += postNorm(attnPost, attn( rms(cur,ln1) ));  cur += postNorm(ffnPost, ffn( rms(cur,ln2) ))
//   attention: q/k/v = W@x (clamped); per-head RMSNorm on q(qNorm)/k(kNorm)/v(no-weight);
//     2D NeoX RoPE theta=100 (x on dims[0:hd/2], y on [hd/2:hd]); BIDIRECTIONAL full attn, NO 1/sqrt scale; out=Wo@.
//   ffn: GeGLU with gelu_quick: down( geluQuick(gate@x) * (up@x) ),  ffnDim=3072
//   pool: merge×merge avg over the patch grid, * sqrt(visionDim)
//   project: clamp -> mm.input_projection (visionDim->modelDim) -> clamp -> RMSNorm(no weight)
// NOTE: activation clamps come from the per-tensor calibration tensors (input_max/min, output_max/min);
//       required for parity with the quantized reference. This file loads F32 weights.
package com.qxotic.llm;

import com.qxotic.jinfer.Embedder;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.FlashAttention;
import com.qxotic.jinfer.GGMLTensorEntry;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.ModelLoader;
import com.qxotic.jinfer.Norms;
import com.qxotic.jinfer.Parallel;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.function.Consumer;

/** Batched gemma4v vision tower → projected model-dim rows. Implements {@link Embedder} over images. */
public final class Gemma4Vision implements Embedder<Media.Image> {

    // --- config (from clip.vision.* metadata) ---
    final int imageSize, patchSize, visionDim, nHead, headDim, nLayer, ffnDim, modelDim, merge, posSize;
    final float normEps, ropeTheta;
    // --- weights ---
    final FloatTensor patchEmbd;     // [visionDim, 3*patch*patch]  (conv as matmul)
    final FloatTensor posEmbd;       // [visionDim, posSize, 2] flattened (x-table then y-table)
    final FloatTensor mmProj;        // [modelDim, visionDim]
    final ClampInfo mmProjClamp;
    final Layer[] layers;

    /** Per-tensor activation calibration ranges (from the *.input_min/max + *.output_min/max tensors):
     *  clamp the matmul INPUT to [inpMin,inpMax] and the OUTPUT to [outMin,outMax]. Required for parity. */
    record ClampInfo(float inpMin, float inpMax, float outMin, float outMax) {}

    record Layer(F32FloatTensor ln1, F32FloatTensor ln2, F32FloatTensor attnPostNorm, F32FloatTensor ffnPostNorm,
                 F32FloatTensor qNorm, F32FloatTensor kNorm,
                 FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
                 FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown,
                 ClampInfo wqClamp, ClampInfo wkClamp, ClampInfo wvClamp, ClampInfo woClamp,
                 ClampInfo ffnGateClamp, ClampInfo ffnUpClamp, ClampInfo ffnDownClamp) {}

    Gemma4Vision(int imageSize, int patchSize, int visionDim, int nHead, int nLayer, int ffnDim, int modelDim,
                 int merge, int posSize, float normEps, float ropeTheta,
                 FloatTensor patchEmbd, FloatTensor posEmbd, FloatTensor mmProj, ClampInfo mmProjClamp, Layer[] layers) {
        this.imageSize = imageSize; this.patchSize = patchSize; this.visionDim = visionDim;
        this.nHead = nHead; this.headDim = visionDim / nHead; this.nLayer = nLayer; this.ffnDim = ffnDim;
        this.modelDim = modelDim; this.merge = Math.max(1, merge); this.posSize = posSize;
        this.normEps = normEps; this.ropeTheta = ropeTheta;
        this.patchEmbd = patchEmbd; this.posEmbd = posEmbd; this.mmProj = mmProj; this.mmProjClamp = mmProjClamp; this.layers = layers;
    }

    // === Embedder seam ===

    @Override
    public void embed(Media.Image image, int maxChunkSize, Consumer<FloatTensor> sink) {
        FloatTensor rows = encode(image);   // [nTokens, modelDim]
        sink.accept(rows);
    }

    /** Encode one image → projected rows (nTokens × modelDim), all patches batched through the tower. */
    public FloatTensor encode(Media.Image image) {
        // 1. preprocess + patch-embed (+ 2D position) → patches: [nPatch, visionDim]
        Patches p = patchify(image);
        int n = p.count;
        FloatTensor cur = p.tokens;        // [n, visionDim]
        FloatTensor xb = FloatTensor.allocateF32(n * visionDim);
        FloatTensor q = FloatTensor.allocateF32(n * visionDim), k = FloatTensor.allocateF32(n * visionDim), v = FloatTensor.allocateF32(n * visionDim);
        FloatTensor attn = FloatTensor.allocateF32(n * visionDim);
        FloatTensor g = FloatTensor.allocateF32(n * ffnDim), u = FloatTensor.allocateF32(n * ffnDim);
        // K/V kept in F16 (like the LLM KV cache) for the rolling flash attention (no materialized scores)
        FloatTensor kF16 = FloatTensor.allocateF16(n, visionDim), vF16 = FloatTensor.allocateF16(n, visionDim);

        // 2. tower (batched)
        for (Layer l : layers) {
            rms(xb, cur, l.ln1(), n, visionDim);                 // n1 = rms(cur, ln1)
            attention(xb, q, k, v, attn, kF16, vF16, l, p.px, p.py, n);  // attn(n1) -> attn
            rmsAddResidual(cur, attn, l.attnPostNorm(), n, visionDim);  // cur += rms(attn, attnPost)
            rms(xb, cur, l.ln2(), n, visionDim);                 // n2 = rms(cur, ln2)
            feedForward(xb, g, u, attn, l, n);                   // ffn(n2) -> attn (reused)
            rmsAddResidual(cur, attn, l.ffnPostNorm(), n, visionDim);   // cur += rms(ffn, ffnPost)
        }

        // 3. pool (merge x merge avg * sqrt(visionDim)) then project + final RMSNorm
        return projectPooled(cur, p.px, p.py);
    }

    // === batched tower steps ===

    private void attention(FloatTensor x, FloatTensor q, FloatTensor k, FloatTensor v, FloatTensor out,
                           FloatTensor kF16, FloatTensor vF16, Layer l, int px, int py, int n) {
        clampedGemm(x, q, n, visionDim, visionDim, l.wq(), l.wqClamp());
        clampedGemm(x, k, n, visionDim, visionDim, l.wk(), l.wkClamp());
        clampedGemm(x, v, n, visionDim, visionDim, l.wv(), l.wvClamp());
        // per-head RMS norms (q,k with weight; v no weight) + 2D RoPE on q,k
        Parallel.forRows(n, t -> {
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
        // K/V → F16, then rolling BIDIRECTIONAL flash attention: online softmax, no materialized [n,n] score
        // matrix (the memory-bound bottleneck), no 1/√d scale (scale=1, matches the gemma4v reference).
        k.copyTo(0, kF16, 0, n * visionDim);
        v.copyTo(0, vF16, 0, n * visionDim);
        FlashAttention.bidirectionalPrefill(q, out, kF16, vF16, nHead, n, headDim, visionDim, visionDim, 1, 1.0f);
        // output projection in place: x <- Wo @ out  (reuse x as scratch is unsafe; write to q)
        clampedGemm(out, q, n, visionDim, visionDim, l.wo(), l.woClamp());
        q.copyTo(0, out, 0, n * visionDim);
    }

    private FloatTensor clampScratch;   // reused input-clamp buffer (lazily grown); avoids a per-call allocation

    /** Clamped matmul: clamp input to [inpMin,inpMax], gemm, clamp output to [outMin,outMax] (reference parity). */
    private void clampedGemm(FloatTensor in, FloatTensor out, int n, int inDim, int outDim, FloatTensor w, ClampInfo c) {
        FloatTensor src = in;
        if (c != null) {
            int need = n * inDim;
            if (clampScratch == null || clampScratch.size() < need) clampScratch = FloatTensor.allocateF32(need);
            src = clampScratch;
            in.copyTo(0, src, 0, need);                        // vectorized copy ...
            src.clampInPlace(0, need, c.inpMin(), c.inpMax());  // ... then SIMD clamp (was scalar getFloat/setFloat)
        }
        w.gemm(src, inDim, out, outDim, n, outDim, inDim);
        if (c != null) out.clampInPlace(0, n * outDim, c.outMin(), c.outMax());
    }

    private void feedForward(FloatTensor x, FloatTensor g, FloatTensor u, FloatTensor out, Layer l, int n) {
        clampedGemm(x, g, n, visionDim, ffnDim, l.ffnGate(), l.ffnGateClamp());
        clampedGemm(x, u, n, visionDim, ffnDim, l.ffnUp(), l.ffnUpClamp());
        Parallel.forRows(n, t -> {                      // gelu_quick(gate)*up
            int base = t * ffnDim;
            for (int d = 0; d < ffnDim; d++) {
                float gg = g.getFloat(base + d);
                float gq = gg / (1f + (float) Math.exp(-1.702f * gg));
                g.setFloat(base + d, gq * u.getFloat(base + d));
            }
        });
        clampedGemm(g, out, n, ffnDim, visionDim, l.ffnDown(), l.ffnDownClamp());
    }

    private FloatTensor projectPooled(FloatTensor cur, int px, int py) {
        int outX = Math.max(1, px / merge), outY = Math.max(1, py / merge);
        int nTok = outX * outY;
        float scale = (float) Math.sqrt(visionDim);
        FloatTensor pooled = FloatTensor.allocateF32(nTok * visionDim);
        for (int oy = 0; oy < outY; oy++) for (int ox = 0; ox < outX; ox++) {
            int dst = (oy * outX + ox) * visionDim, cnt = 0;
            for (int my = 0; my < merge; my++) { int p = oy * merge + my; if (p >= py) continue;
                for (int mx = 0; mx < merge; mx++) { int q = ox * merge + mx; if (q >= px) continue;
                    int src = (p * px + q) * visionDim;
                    for (int d = 0; d < visionDim; d++) pooled.setFloat(dst + d, pooled.getFloat(dst + d) + cur.getFloat(src + d));
                    cnt++; } }
            float inv = cnt > 0 ? scale / cnt : scale;
            for (int d = 0; d < visionDim; d++) pooled.setFloat(dst + d, pooled.getFloat(dst + d) * inv);
        }
        // mm_soft_emb_norm: RMSNorm the pooled 768-dim features BEFORE the projection (llama.cpp order) —
        // confirmed by the embedding diff: this bounds the output range (max ~4.6) to match llama.cpp,
        // vs project→RMSNorm which let outliers to ~7.4 and drowned small-object features.
        Parallel.forRows(nTok, t -> rmsNoWeight(pooled, t * visionDim, visionDim));
        FloatTensor projected = FloatTensor.allocateF32(nTok * modelDim);
        clampedGemm(pooled, projected, nTok, visionDim, modelDim, mmProj, mmProjClamp);
        return projected;
    }

    // === norms / rope helpers ===

    private void rms(FloatTensor out, FloatTensor x, F32FloatTensor w, int n, int dim) {
        Parallel.forRows(n, t -> Norms.rmsnorm(out, (long) t * dim, x, (long) t * dim, w, dim, normEps));
    }

    private void rmsAddResidual(FloatTensor residual, FloatTensor x, F32FloatTensor w, int n, int dim) {
        FloatTensor tmp = FloatTensor.allocateF32(n * dim);
        Parallel.forRows(n, t -> {
            Norms.rmsnorm(tmp, (long) t * dim, x, (long) t * dim, w, dim, normEps);
            residual.addInPlace((long) t * dim, tmp, (long) t * dim, dim);
        });
    }

    private void headRms(FloatTensor x, int off, F32FloatTensor w) {
        float ss = 0f;
        for (int d = 0; d < headDim; d++) { float vv = x.getFloat(off + d); ss += vv * vv; }
        float inv = (float) (1.0 / Math.sqrt(ss / headDim + normEps));
        for (int d = 0; d < headDim; d++) x.setFloat(off + d, x.getFloat(off + d) * inv * (w == null ? 1f : w.getFloat(d)));
    }

    private void rmsNoWeight(FloatTensor x, int off, int dim) {
        float ss = 0f;
        for (int d = 0; d < dim; d++) { float vv = x.getFloat(off + d); ss += vv * vv; }
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

    private Patches patchify(Media.Image image) {
        int ps = patchSize, factor = ps * Math.max(1, merge), tw, th;
        if (VisionPreprocess.SMART_RESIZE) {
            int maxPixels = VisionPreprocess.budget(280) * factor * factor, minPixels = factor * factor;
            int[] wh = VisionPreprocess.smartResize(image.width(), image.height(), factor, minPixels, maxPixels);
            tw = wh[0]; th = wh[1];
        } else {
            tw = th = imageSize;   // fixed procSize square (gemma4.java-reference parity)
        }
        int px = tw / ps, py = th / ps, n = px * py, patchVec = 3 * ps * ps;
        FloatTensor flat = VisionPreprocess.im2col(image, tw, th, ps);
        FloatTensor tokens = FloatTensor.allocateF32(n * visionDim);
        patchEmbd.gemm(flat, patchVec, tokens, visionDim, n, visionDim, patchVec);   // conv as matmul
        for (int gy = 0; gy < py; gy++) for (int gx = 0; gx < px; gx++) {            // + factorized 2D position
            int tok = (gy * px + gx) * visionDim, xb = visionDim * gx, yb = visionDim * (gy + posSize);
            for (int d = 0; d < visionDim; d++) tokens.setFloat(tok + d, tokens.getFloat(tok + d) + posEmbd.getFloat(xb + d) + posEmbd.getFloat(yb + d));
        }
        return new Patches(tokens, n, px, py);
    }

    // === loader ===

    public static Gemma4Vision loadModel(Path mmprojPath) throws IOException {
        try (FileChannel fc = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            var gguf = ModelLoader.readGguf(fc, mmprojPath.toString());
            Map<String, GGMLTensorEntry> t = ModelLoader.loadTensors(fc, gguf);
            int imageSize = gguf.getValueOrDefault(int.class, "clip.vision.image_size", 224);
            int patchSize = gguf.getValueOrDefault(int.class, "clip.vision.patch_size", 16);
            int visionDim = gguf.getValueOrDefault(int.class, "clip.vision.embedding_length", 768);
            int nHead = gguf.getValueOrDefault(int.class, "clip.vision.attention.head_count", 12);
            int nLayer = gguf.getValueOrDefault(int.class, "clip.vision.block_count", 16);
            int ffnDim = gguf.getValueOrDefault(int.class, "clip.vision.feed_forward_length", 3072);
            int merge = gguf.getValueOrDefault(int.class, "clip.vision.proj_scale_factor", 3);
            // smartResize canonical processing size: a multiple of (patch*merge) whose area fits image_max_pixels
            // (default 280*(patch*merge)^2). For patch16/merge3 -> 768 -> 48x48 patches -> merge -> 16x16 = 256 tokens.
            int curMerge = Math.max(1, merge), factor = patchSize * curMerge;
            int maxPixels = VisionPreprocess.IMAGE_TOKEN_BUDGET > 0 ? VisionPreprocess.IMAGE_TOKEN_BUDGET * factor * factor : gguf.getValueOrDefault(int.class, "clip.vision.image_max_pixels", 280 * factor * factor);
            int procSize = (int) (Math.sqrt(maxPixels) / factor) * factor;
            int modelDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", 1536);
            float eps = gguf.getValueOrDefault(float.class, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
            FloatTensor patchEmbd = ModelLoader.loadQuantized(t.get("v.patch_embd.weight"));
            FloatTensor posEmbd = ModelLoader.loadQuantized(t.get("v.position_embd.weight"));
            int posSize = (int) (posEmbd.size() / (visionDim * 2L));
            FloatTensor mmProj = ModelLoader.loadQuantized(t.get("mm.input_projection.weight"));
            ClampInfo mmProjClamp = readClampInfo(t, "mm.input_projection");
            Layer[] layers = new Layer[nLayer];
            for (int i = 0; i < nLayer; i++) {
                String p = "v.blk." + i + ".";
                layers[i] = new Layer(
                        ModelLoader.f32OrNull(t, p + "ln1.weight"), ModelLoader.f32OrNull(t, p + "ln2.weight"),
                        ModelLoader.f32OrNull(t, p + "attn_post_norm.weight"), ModelLoader.f32OrNull(t, p + "ffn_post_norm.weight"),
                        ModelLoader.f32OrNull(t, p + "attn_q_norm.weight"), ModelLoader.f32OrNull(t, p + "attn_k_norm.weight"),
                        ModelLoader.loadQuantized(t.get(p + "attn_q.weight")), ModelLoader.loadQuantized(t.get(p + "attn_k.weight")),
                        ModelLoader.loadQuantized(t.get(p + "attn_v.weight")), ModelLoader.loadQuantized(t.get(p + "attn_out.weight")),
                        ModelLoader.loadQuantized(t.get(p + "ffn_gate.weight")), ModelLoader.loadQuantized(t.get(p + "ffn_up.weight")),
                        ModelLoader.loadQuantized(t.get(p + "ffn_down.weight")),
                        readClampInfo(t, p + "attn_q"), readClampInfo(t, p + "attn_k"),
                        readClampInfo(t, p + "attn_v"), readClampInfo(t, p + "attn_out"),
                        readClampInfo(t, p + "ffn_gate"), readClampInfo(t, p + "ffn_up"), readClampInfo(t, p + "ffn_down"));
            }
            return new Gemma4Vision(procSize, patchSize, visionDim, nHead, nLayer, ffnDim, modelDim, merge, posSize, eps, 100.0f,
                    patchEmbd, posEmbd, mmProj, mmProjClamp, layers);
        }
    }

    /** Per-tensor activation clamp ranges from the calibration tensors (each a single F32 scalar); null if absent. */
    private static ClampInfo readClampInfo(Map<String, GGMLTensorEntry> t, String base) {
        F32FloatTensor inMin = ModelLoader.f32OrNull(t, base + ".input_min"), inMax = ModelLoader.f32OrNull(t, base + ".input_max");
        F32FloatTensor outMin = ModelLoader.f32OrNull(t, base + ".output_min"), outMax = ModelLoader.f32OrNull(t, base + ".output_max");
        if (inMin == null || inMax == null || outMin == null || outMax == null) return null;
        return new ClampInfo(inMin.getFloat(0), inMax.getFloat(0), outMin.getFloat(0), outMax.getFloat(0));
    }

}
