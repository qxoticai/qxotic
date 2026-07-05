// Gemma 4 "unified" vision embedder (projector_type = gemma4uv, e.g. gemma-4-12b).
//
// Reference: llama.cpp tools/mtmd/models/gemma4uv.cpp (clip_graph_gemma4uv::build).
// Unlike the gemma4v full SigLIP ViT (Gemma4Vision), the unified embedder is MINIMAL: there is no
// transformer at all - no attention, no FFN, no per-block norms. The 3x3 token merge is baked directly
// into a bigger conv patch: the base patch (16) is multiplied by n_merge (3) to a 48x48 conv, so each
// token already spans a 48x48 region (48*48*3 = 6912 input features) and n_merge becomes 1.
//
// Graph (all patches batched):
//   im2col 48x48            -> [n, 6912]      (channel-outer [c, ky, kx] ordering, pixels scaled to [-1,1])
//   patch_norm.1 (LayerNorm over 6912)
//   patch_embd  (6912->3840) + patch_embd.bias
//   patch_norm.2 (LayerNorm over 3840)
//   + factorized 2D position (posEmbd[gx] for x, posEmbd[gy] for y; direct index, no interpolation)
//   patch_norm.3 (LayerNorm over 3840; "pos_norm")
//   RMSNorm (no weight; embedding_pre_projection_norm)
//   mm.input_projection (3840->3840)
//
// The three patch norms are PyTorch LayerNorm (mean-subtracting, weight+bias, eps 1e-5); only the final
// pre-projection norm is RMSNorm (eps = clip.vision.attention.layer_norm_epsilon, ~1e-6). Resize is the
// same aspect-preserving smart_resize as gemma4v (BILINEAR, factor 48), so a 640x480 image yields the
// same 130 tokens. Backs Gemma4's MultiModal Embedder<Media.Image> when the mmproj is gemma4uv.
package com.qxotic.llm;

import com.qxotic.jinfer.Embedder;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.GGMLTensorEntry;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.ModelLoader;
import com.qxotic.jinfer.Parallel;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.function.Consumer;

public final class Gemma4VisionUnified implements Embedder<Media.Image> {

    static final float LN_EPS = 1e-5f;          // PyTorch LayerNorm eps (hardcoded in gemma4uv.cpp)

    final int patchSize;      // effective conv patch (48 = base 16 * merge 3)
    final int visionDim;      // n_embd = 3840
    final int modelDim;       // projection_dim = 3840
    final int posSize;        // pos-table columns (1120)
    final int patchVec;       // 3 * patchSize * patchSize = 6912
    final float rmsEps;       // pre-projection RMSNorm eps (~1e-6)

    final FloatTensor patchEmbd;   // [visionDim, patchVec] conv-as-matmul
    final FloatTensor patchBias;   // [visionDim]
    final FloatTensor posEmbd;     // [visionDim, posSize, 2] flattened (x-table then y-table)
    final FloatTensor mmProj;      // [modelDim, visionDim]
    final F32FloatTensor ln1w, ln1b, ln2w, ln2b, ln3w, ln3b;

    Gemma4VisionUnified(int patchSize, int visionDim, int modelDim, int posSize, float rmsEps,
                        FloatTensor patchEmbd, FloatTensor patchBias, FloatTensor posEmbd, FloatTensor mmProj,
                        F32FloatTensor ln1w, F32FloatTensor ln1b, F32FloatTensor ln2w, F32FloatTensor ln2b,
                        F32FloatTensor ln3w, F32FloatTensor ln3b) {
        this.patchSize = patchSize; this.visionDim = visionDim; this.modelDim = modelDim; this.posSize = posSize;
        this.patchVec = 3 * patchSize * patchSize; this.rmsEps = rmsEps;
        this.patchEmbd = patchEmbd; this.patchBias = patchBias; this.posEmbd = posEmbd; this.mmProj = mmProj;
        this.ln1w = ln1w; this.ln1b = ln1b; this.ln2w = ln2w; this.ln2b = ln2b; this.ln3w = ln3w; this.ln3b = ln3b;
    }

    @Override
    public void embed(Media.Image image, int maxChunkSize, Consumer<FloatTensor> sink) {
        sink.accept(encode(image));
    }

    /** Encode one image -> projected rows (nTokens x modelDim). */
    public FloatTensor encode(Media.Image image) {
        int ps = patchSize, factor = ps;                 // merge already baked into the conv patch
        int maxPixels = VisionPreprocess.budget(280) * factor * factor, minPixels = 40 * factor * factor;
        int[] wh = VisionPreprocess.SMART_RESIZE
                ? VisionPreprocess.smartResize(image.width(), image.height(), factor, minPixels, maxPixels)
                : new int[]{ 16 * factor, 16 * factor };  // fixed-square fallback (256 tokens)
        int tw = wh[0], th = wh[1], px = tw / ps, n = px * (th / ps);

        // 1. im2col (48x48, channel-outer, [-1,1]) -> patch_norm.1 (LayerNorm)
        FloatTensor flat = VisionPreprocess.im2col(image, tw, th, ps);
        layerNorm(flat, n, patchVec, ln1w, ln1b);

        // 2. conv patch-embed + bias, LayerNorm
        FloatTensor cur = FloatTensor.allocateF32(n * visionDim);
        patchEmbd.gemm(flat, patchVec, cur, visionDim, n, visionDim, patchVec);
        addBias(cur, n, visionDim, patchBias);
        layerNorm(cur, n, visionDim, ln2w, ln2b);

        // 3. + factorized 2D position, then pos-norm (LayerNorm)
        Parallel.forRows(n, gi -> {
            int gy = gi / px, gx = gi % px, tok = gi * visionDim, xb = visionDim * gx, yb = visionDim * (gy + posSize);
            for (int d = 0; d < visionDim; d++)
                cur.setFloat((long) tok + d, cur.getFloat((long) tok + d) + posEmbd.getFloat(xb + d) + posEmbd.getFloat(yb + d));
        });
        layerNorm(cur, n, visionDim, ln3w, ln3b);

        // 4. pre-projection RMSNorm (no weight) then mm projection
        Parallel.forRows(n, t -> rmsNoWeight(cur, (long) t * visionDim, visionDim, rmsEps));
        FloatTensor rows = FloatTensor.allocateF32(n * modelDim);
        mmProj.gemm(cur, visionDim, rows, modelDim, n, modelDim, visionDim);
        return rows;
    }

    /** PyTorch LayerNorm over each row: (x - mean) / sqrt(var + eps) * w + b. */
    private static void layerNorm(FloatTensor x, int n, int dim, F32FloatTensor w, F32FloatTensor b) {
        Parallel.forRows(n, t -> {
            long off = (long) t * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += x.getFloat(off + d);
            mean /= dim;
            float var = 0f;
            for (int d = 0; d < dim; d++) { float v = x.getFloat(off + d) - mean; var += v * v; }
            float inv = (float) (1.0 / Math.sqrt(var / dim + LN_EPS));
            for (int d = 0; d < dim; d++)
                x.setFloat(off + d, (x.getFloat(off + d) - mean) * inv * w.getFloat(d) + b.getFloat(d));
        });
    }

    private static void addBias(FloatTensor x, int n, int dim, FloatTensor bias) {
        Parallel.forRows(n, t -> {
            long off = (long) t * dim;
            for (int d = 0; d < dim; d++) x.setFloat(off + d, x.getFloat(off + d) + bias.getFloat(d));
        });
    }

    private static void rmsNoWeight(FloatTensor x, long off, int dim, float eps) {
        float ss = 0f;
        for (int d = 0; d < dim; d++) { float v = x.getFloat(off + d); ss += v * v; }
        float inv = (float) (1.0 / Math.sqrt(ss / dim + eps));
        for (int d = 0; d < dim; d++) x.setFloat(off + d, x.getFloat(off + d) * inv);
    }

    // === loader ===

    public static Gemma4VisionUnified loadModel(Path mmprojPath) throws IOException {
        try (FileChannel fc = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            var gguf = ModelLoader.readGguf(fc, mmprojPath.toString());
            Map<String, GGMLTensorEntry> t = ModelLoader.loadTensors(fc, gguf);
            int basePatch = gguf.getValueOrDefault(int.class, "clip.vision.patch_size", 16);
            int merge = gguf.getValueOrDefault(int.class, "clip.vision.proj_scale_factor", 3);
            int patchSize = basePatch * Math.max(1, merge);   // unified bakes the merge into the conv patch
            int visionDim = gguf.getValueOrDefault(int.class, "clip.vision.embedding_length", 3840);
            int modelDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", visionDim);
            float rmsEps = gguf.getValueOrDefault(float.class, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
            FloatTensor patchEmbd = ModelLoader.loadQuantized(t.get("v.patch_embd.weight"));
            FloatTensor patchBias = ModelLoader.loadQuantized(t.get("v.patch_embd.bias"));
            FloatTensor posEmbd = ModelLoader.loadQuantized(t.get("v.position_embd.weight"));
            int posSize = (int) (posEmbd.size() / (visionDim * 2L));
            FloatTensor mmProj = ModelLoader.loadQuantized(t.get("mm.input_projection.weight"));
            return new Gemma4VisionUnified(patchSize, visionDim, modelDim, posSize, rmsEps,
                    patchEmbd, patchBias, posEmbd, mmProj,
                    ModelLoader.f32OrNull(t, "v.patch_norm.1.weight"), ModelLoader.f32OrNull(t, "v.patch_norm.1.bias"),
                    ModelLoader.f32OrNull(t, "v.patch_norm.2.weight"), ModelLoader.f32OrNull(t, "v.patch_norm.2.bias"),
                    ModelLoader.f32OrNull(t, "v.patch_norm.3.weight"), ModelLoader.f32OrNull(t, "v.patch_norm.3.bias"));
        }
    }
}
