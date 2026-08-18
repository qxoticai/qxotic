package com.qxotic.jinfer.models.qwen35;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Segments;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.boundary.Arenas;
import com.qxotic.jinfer.boundary.Media;
import com.qxotic.jinfer.boundary.MediaProjector;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Qwen3-VL vision tower and 2x2 spatial merger ({@code projector_type=qwen3vl_merger}) carried by
 * the Qwen3.5 mmproj: a fused-QKV ViT with a DUAL summed patch convolution, an absolute
 * position table bilinearly resized to the target grid, vision M-RoPE ({@code [yyyyxxxx]} per
 * head), tanh-GELU FFNs, and a final {mm.0 → GELU → mm.2} MLP over each 2x2 patch block.
 *
 * <p>Ported from llama.cpp's {@code clip_graph_qwen3vl}. The tower emits tokens in the 2x2-block
 * major order its mrope position table implies, so the merger is a plain contiguous copy.
 */
public final class Qwen35Vision implements MediaProjector<Media.Image> {
    private final int patchSize,
            patchVector,
            visionDim,
            modelDim,
            headCount,
            headDim,
            ffnDim,
            merge,
            positionSide;
    private final float normEps;
    private final MemoryView<MemorySegment> patch0, patch1, patchBias;
    private final MemoryView<MemorySegment> positionEmbedding;
    private final MemoryView<MemorySegment> postLnW, postLnB;
    private final Linear mm0, mm2;
    private final Layer[] layers;
    private final float[] invFreq;

    record Linear(
            MemoryView<MemorySegment> weight,
            MemoryView<MemorySegment> bias,
            int outputDim,
            int inputDim) {}

    record Layer(
            MemoryView<MemorySegment> ln1W,
            MemoryView<MemorySegment> ln1B,
            MemoryView<MemorySegment> qkvW,
            MemoryView<MemorySegment> qkvB,
            MemoryView<MemorySegment> attnOutW,
            MemoryView<MemorySegment> attnOutB,
            MemoryView<MemorySegment> ln2W,
            MemoryView<MemorySegment> ln2B,
            MemoryView<MemorySegment> ffnUpW,
            MemoryView<MemorySegment> ffnUpB,
            MemoryView<MemorySegment> ffnDownW,
            MemoryView<MemorySegment> ffnDownB) {}

    Qwen35Vision(
            int patchSize,
            int visionDim,
            int modelDim,
            int headCount,
            int ffnDim,
            int merge,
            int positionSide,
            float normEps,
            MemoryView<MemorySegment> patch0,
            MemoryView<MemorySegment> patch1,
            MemoryView<MemorySegment> patchBias,
            MemoryView<MemorySegment> positionEmbedding,
            MemoryView<MemorySegment> postLnW,
            MemoryView<MemorySegment> postLnB,
            Linear mm0,
            Linear mm2,
            Layer[] layers) {
        if (patchSize <= 0
                || visionDim <= 0
                || modelDim <= 0
                || headCount <= 0
                || ffnDim <= 0
                || merge <= 1
                || positionSide <= 0)
            throw new IllegalArgumentException("vision dimensions must be positive");
        if (visionDim % headCount != 0)
            throw new IllegalArgumentException(
                    "head_count " + headCount + " does not divide embedding_length " + visionDim);
        int headDim = visionDim / headCount;
        if (headDim % 4 != 0)
            throw new IllegalArgumentException(
                    "head_dim " + headDim + " must be divisible by 4 (vision M-RoPE sections)");

        this.patchSize = patchSize;
        this.patchVector = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        this.visionDim = visionDim;
        this.modelDim = modelDim;
        this.headCount = headCount;
        this.headDim = headDim;
        this.ffnDim = ffnDim;
        this.merge = merge;
        this.positionSide = positionSide;
        this.normEps = normEps;
        // The dual conv reads the two kernel planes as raw FP32 (the kernel is not a plain gemm
        // over patch rows), so patch weights must be FP32 even though the linear weights below may
        // be quantized.
        this.patch0 = requirePatchF32(patch0, "v.patch_embd.weight", visionDim, patchVector);
        this.patch1 = requirePatchF32(patch1, "v.patch_embd.weight.1", visionDim, patchVector);
        this.patchBias = requireF32(patchBias, "v.patch_embd.bias", Shape.flat(visionDim));
        this.positionEmbedding =
                requireF32(
                        positionEmbedding,
                        "v.position_embd.weight",
                        Shape.flat(Math.multiplyExact(positionSide, positionSide), visionDim));
        this.postLnW = requireF32(postLnW, "v.post_ln.weight", Shape.flat(visionDim));
        this.postLnB = requireF32(postLnB, "v.post_ln.bias", Shape.flat(visionDim));
        this.mm0 = requireLinear(mm0, "mm.0");
        this.mm2 = requireLinear(mm2, "mm.2");
        if (this.mm0.inputDim() != Math.multiplyExact(Math.multiplyExact(merge, merge), visionDim)
                || this.mm2.inputDim() != this.mm0.outputDim()
                || this.mm2.outputDim() != modelDim)
            throw new IllegalArgumentException("invalid qwen3vl_merger projection geometry");
        this.layers = Objects.requireNonNull(layers, "layers").clone();
        for (int i = 0; i < this.layers.length; i++) validateLayer(this.layers[i], i);

        // ponytail: ggml_rope_multi receives n_dims = headDim/2, so theta_scale is
        // base^(-2/(headDim/2)) and each M-RoPE section restarts its theta at base^0. Section 0
        // (j < headDim/4) reads tokenY, section 1 reads tokenX - the [yyyyxxxx] layout.
        this.invFreq = new float[headDim / 4];
        for (int j = 0; j < invFreq.length; j++)
            invFreq[j] = (float) Math.pow(10000.0, -2.0 * j / (headDim / 2.0));
    }

    @Override
    public int positions(Media.Image image) {
        return Qwen35VisionPreprocess.positions(
                Objects.requireNonNull(image, "image"), patchSize, merge);
    }

    @Override
    public String planId() {
        return "qwen3vl patch="
                + patchSize
                + " merge="
                + merge
                + " pos="
                + positionSide
                + " tokens="
                + Qwen35VisionPreprocess.MIN_IMAGE_TOKENS
                + ".."
                + Qwen35VisionPreprocess.MAX_IMAGE_TOKENS;
    }

    @Override
    public void project(Media.Image image, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(image, "image");
        Objects.requireNonNull(sink, "sink");
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        int[] size =
                Qwen35VisionPreprocess.smartResize(
                        image.width(),
                        image.height(),
                        Math.multiplyExact(patchSize, merge),
                        Qwen35VisionPreprocess.MIN_IMAGE_TOKENS,
                        Qwen35VisionPreprocess.MAX_IMAGE_TOKENS);
        int patchesX = size[0] / patchSize, patchesY = size[1] / patchSize;
        int merged = Math.multiplyExact(patchesX, patchesY) / (merge * merge);
        if (merged > maxChunkSize)
            throw new IllegalArgumentException(
                    "vision block has " + merged + " rows, exceeding maxChunkSize " + maxChunkSize);
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            sink.accept(encode(image, scratch, size, patchesX, patchesY));
        } finally {
            Arenas.close(scratch);
        }
    }

    private MemoryView<MemorySegment> encode(
            Media.Image image,
            MemoryArena<MemorySegment> scratch,
            int[] size,
            int patchesX,
            int patchesY) {
        int nPos = Math.multiplyExact(patchesX, patchesY);
        int merged = nPos / (merge * merge);
        float[] pixels = Qwen35VisionPreprocess.normalize(image, size[0], size[1]);
        int plane = Math.multiplyExact(size[0], size[1]);

        // ponytail: the dual conv writes tokens directly in the 2x2-block-major order llama's
        // spatial-merge permute implies; do not "fix" this to raster - it is what makes the merger
        // below a single contiguous copy.
        MemoryView<MemorySegment> tokens = Views.allocateF32(scratch, nPos, visionDim);
        MemoryView<MemorySegment> positions = resizePositions(scratch, patchesX, patchesY);
        Parallel.forRows(
                nPos,
                t -> {
                    int py = tokenY(t, patchesX), px = tokenX(t, patchesX);
                    long row = (long) t * visionDim;
                    int base = (py * patchSize) * size[0] + px * patchSize;
                    for (int c = 0; c < visionDim; c++) {
                        long kBase = (long) c * patchVector;
                        float sum = 0f;
                        for (int ky = 0; ky < patchSize; ky++) {
                            int rowOff = base + ky * size[0];
                            long kRow = kBase + (long) ky * patchSize;
                            for (int kx = 0; kx < patchSize; kx++) {
                                float r = pixels[rowOff + kx];
                                float g = pixels[plane + rowOff + kx];
                                float b = pixels[2 * plane + rowOff + kx];
                                long i0 = kRow + kx;
                                long i1 = i0 + (long) patchSize * patchSize;
                                long i2 = i1 + (long) patchSize * patchSize;
                                sum +=
                                        getF(patch0, i0) * r
                                                + getF(patch0, i1) * g
                                                + getF(patch0, i2) * b
                                                + getF(patch1, i0) * r
                                                + getF(patch1, i1) * g
                                                + getF(patch1, i2) * b;
                            }
                        }
                        putF(tokens, row + c, sum);
                    }
                    long posRow = (long) py * patchesX + px;
                    for (int c = 0; c < visionDim; c++) {
                        float v = getF(tokens, row + c);
                        v += getF(patchBias, c);
                        v += getF(positions, posRow * visionDim + c);
                        putF(tokens, row + c, v);
                    }
                });

        // Tower layers: ln1 -> fused QKV -> vision M-RoPE -> full attention -> +bias -> +residual
        // -> ln2 -> tanh-GELU FFN -> +residual.
        MemoryView<MemorySegment> hidden = Views.allocateF32(scratch, nPos, visionDim);
        MemoryView<MemorySegment> qkv = Views.allocateF32(scratch, nPos, 3 * visionDim);
        MemoryView<MemorySegment> attn = Views.allocateF32(scratch, nPos, visionDim);
        MemoryView<MemorySegment> scores = Views.allocateF32(scratch, nPos, nPos);
        MemoryView<MemorySegment> q = Views.allocateF32(scratch, nPos, headDim);
        MemoryView<MemorySegment> k = Views.allocateF32(scratch, nPos, headDim);
        MemoryView<MemorySegment> vT = Views.allocateF32(scratch, headDim, nPos);
        MemoryView<MemorySegment> o = Views.allocateF32(scratch, nPos, headDim);
        MemoryView<MemorySegment> ffn = Views.allocateF32(scratch, nPos, ffnDim);
        for (Layer layer : layers) {
            Norms.layerNorm(hidden, tokens, layer.ln1W(), layer.ln1B(), visionDim, nPos, normEps);
            MatMul.gemm(
                    layer.qkvW(),
                    hidden,
                    visionDim,
                    qkv,
                    3 * visionDim,
                    3 * visionDim,
                    nPos,
                    visionDim);
            Ops.addRowBiasInPlace(qkv, 0, layer.qkvB(), 0, nPos, 3 * visionDim);
            attention(qkv, attn, scores, q, k, vT, o, nPos, patchesX);
            MatMul.gemm(
                    layer.attnOutW(),
                    attn,
                    visionDim,
                    hidden,
                    visionDim,
                    visionDim,
                    nPos,
                    visionDim);
            Ops.addRowBiasInPlace(hidden, 0, layer.attnOutB(), 0, nPos, visionDim);
            Ops.addInPlace(tokens, 0, hidden, 0, nPos * visionDim);

            Norms.layerNorm(hidden, tokens, layer.ln2W(), layer.ln2B(), visionDim, nPos, normEps);
            MatMul.gemm(layer.ffnUpW(), hidden, visionDim, ffn, ffnDim, ffnDim, nPos, visionDim);
            Ops.addRowBiasInPlace(ffn, 0, layer.ffnUpB(), 0, nPos, ffnDim);
            geluTanhInPlace(ffn, nPos, ffnDim);
            MatMul.gemm(layer.ffnDownW(), ffn, ffnDim, hidden, visionDim, visionDim, nPos, ffnDim);
            Ops.addRowBiasInPlace(hidden, 0, layer.ffnDownB(), 0, nPos, visionDim);
            Ops.addInPlace(tokens, 0, hidden, 0, nPos * visionDim);
        }

        // Post-layernorm then the merger. The tower's block-major layout already places each 2x2
        // block's four rows consecutively, so the merger input is a single contiguous copy.
        Norms.layerNorm(tokens, tokens, postLnW, postLnB, visionDim, nPos, normEps);
        MemoryView<MemorySegment> mergedRows = Views.allocateF32(scratch, merged, 4 * visionDim);
        MemoryView<MemorySegment> out = Views.allocateF32(scratch, merged, modelDim);
        Convert.copyF32(tokens, 0, mergedRows, 0, (long) nPos * visionDim);
        MatMul.gemm(
                mm0.weight(),
                mergedRows,
                mm0.inputDim(),
                hidden,
                mm0.outputDim(),
                mm0.outputDim(),
                merged,
                mm0.inputDim());
        Ops.addRowBiasInPlace(hidden, 0, mm0.bias(), 0, merged, mm0.outputDim());
        geluTanhInPlace(hidden, merged, mm0.outputDim());
        MatMul.gemm(
                mm2.weight(),
                hidden,
                mm2.inputDim(),
                out,
                modelDim,
                modelDim,
                merged,
                mm2.inputDim());
        Ops.addRowBiasInPlace(out, 0, mm2.bias(), 0, merged, modelDim);
        return out;
    }

    /**
     * Fused-QKV attention for one tower layer. Q/K are gathered per head, vision M-RoPE'd, then
     * attended over all tokens. V is gathered transposed so the OV gemm contracts tokens.
     *
     * <p>ponytail: this keeps the hand-rolled no-mask softmax instead of FlashAttention - the flash
     * kernel's online softmax differs in last-ulp from llama.cpp's flash-disabled reference this
     * tower is matched against, so switching to flash is a correctness regression, not an
     * optimization.
     */
    private void attention(
            MemoryView<MemorySegment> qkv,
            MemoryView<MemorySegment> attn,
            MemoryView<MemorySegment> scores,
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> k,
            MemoryView<MemorySegment> vT,
            MemoryView<MemorySegment> o,
            int nPos,
            int patchesX) {
        float scale = (float) (1.0 / Math.sqrt(headDim));
        for (int h = 0; h < headCount; h++) {
            long hb = (long) h * headDim;
            for (int t = 0; t < nPos; t++) {
                long src = (long) t * 3 * visionDim + hb;
                int py = tokenY(t, patchesX), px = tokenX(t, patchesX);
                for (int j = 0; j < headDim; j++) {
                    putF(q, (long) t * headDim + j, getF(qkv, src + j));
                    putF(k, (long) t * headDim + j, getF(qkv, src + visionDim + j));
                    putF(vT, (long) j * nPos + t, getF(qkv, src + 2 * visionDim + j));
                }
                rope(q, (long) t * headDim, py, px);
                rope(k, (long) t * headDim, py, px);
            }
            // C[s][row] = W[row] · A[s] with W=k, A=q puts s=query, row=key.
            MatMul.gemm(k, 0, q, headDim, scores, nPos, nPos, nPos, headDim);
            for (int i = 0; i < nPos; i++) {
                long row = (long) i * nPos;
                for (int j = 0; j < nPos; j++) putF(scores, row + j, getF(scores, row + j) * scale);
                Ops.softmaxInPlace(scores, row, nPos);
            }
            // o[t][d] = Σ_j vT[d][j] · scores[t][j].
            MatMul.gemm(vT, 0, scores, nPos, o, headDim, headDim, nPos, nPos);
            for (int t = 0; t < nPos; t++) {
                long dst = (long) t * visionDim + hb;
                for (int j = 0; j < headDim; j++)
                    putF(attn, dst + j, getF(o, (long) t * headDim + j));
            }
        }
    }

    private void rope(MemoryView<MemorySegment> t, long rowBase, int posY, int posX) {
        for (int j = 0; j < headDim / 2; j++) {
            int section = j < headDim / 4 ? 0 : 1;
            int pos = section == 0 ? posY : posX;
            float theta = pos * invFreq[section == 0 ? j : j - headDim / 4];
            float cos = (float) Math.cos(theta), sin = (float) Math.sin(theta);
            long a = rowBase + j, b = rowBase + j + headDim / 2;
            float x0 = getF(t, a);
            float x1 = getF(t, b);
            putF(t, a, x0 * cos - x1 * sin);
            putF(t, b, x0 * sin + x1 * cos);
        }
    }

    private int tokenY(int t, int patchesX) {
        int yb = t / (4 * (patchesX / 2));
        return yb * 2 + (t % 4) / 2;
    }

    private int tokenX(int t, int patchesX) {
        int rem = t % (4 * (patchesX / 2));
        return (rem / 4) * 2 + rem % 2;
    }

    /** Position table resized to the target grid (align-corners bilinear; identity at native). */
    private MemoryView<MemorySegment> resizePositions(
            MemoryArena<MemorySegment> scratch, int patchesX, int patchesY) {
        if (patchesX == positionSide && patchesY == positionSide) return positionEmbedding;
        MemoryView<MemorySegment> out =
                Views.allocateF32(scratch, (long) patchesX * patchesY, visionDim);
        Parallel.forRows(
                patchesY,
                y -> {
                    float gy = y * (positionSide - 1.0f) / Math.max(1, patchesY - 1);
                    int y0 = (int) gy, y1 = Math.min(positionSide - 1, y0 + 1);
                    float wy = gy - y0;
                    for (int x = 0; x < patchesX; x++) {
                        float gx = x * (positionSide - 1.0f) / Math.max(1, patchesX - 1);
                        int x0 = (int) gx, x1 = Math.min(positionSide - 1, x0 + 1);
                        float wx = gx - x0;
                        long dst = ((long) y * patchesX + x) * visionDim;
                        for (int c = 0; c < visionDim; c++) {
                            float a =
                                    getF(
                                            positionEmbedding,
                                            ((long) y0 * positionSide + x0) * visionDim + c);
                            float b =
                                    getF(
                                            positionEmbedding,
                                            ((long) y0 * positionSide + x1) * visionDim + c);
                            float d0 =
                                    getF(
                                            positionEmbedding,
                                            ((long) y1 * positionSide + x0) * visionDim + c);
                            float d1 =
                                    getF(
                                            positionEmbedding,
                                            ((long) y1 * positionSide + x1) * visionDim + c);
                            putF(
                                    out,
                                    dst + c,
                                    a * (1 - wx) * (1 - wy)
                                            + b * wx * (1 - wy)
                                            + d0 * (1 - wx) * wy
                                            + d1 * wx * wy);
                        }
                    }
                });
        return out;
    }

    /**
     * llama.cpp's {@code ggml_gelu_f32} tanh approximation with the same op order and constants
     * ({@code 0.5f*x*(1 + tanhf(SQRT_2_OVER_PI*x*(1 + 0.044715f*x*x)))}). The qwen3vl tower FFN and
     * merger both use {@code FFN_GELU} (clip.use_gelu=1), not {@code FFN_GELU_QUICK}.
     */
    private static void geluTanhInPlace(MemoryView<MemorySegment> t, int rows, int rowDim) {
        final float sqrt2OverPi = 0.79788456080286535587989211986876f;
        final float coefA = 0.044715f;
        Parallel.forRows(
                rows,
                r -> {
                    long base = (long) r * rowDim;
                    for (int c = 0; c < rowDim; c++) {
                        float x = getF(t, base + c);
                        float inner = sqrt2OverPi * x * (1.0f + coefA * x * x);
                        putF(t, base + c, 0.5f * x * (1.0f + (float) Math.tanh(inner)));
                    }
                });
    }

    private static float getF(MemoryView<MemorySegment> v, long i) {
        return Segments.readFloat(v.memory().base(), v.byteOffset() + i * Float.BYTES);
    }

    private static void putF(MemoryView<MemorySegment> v, long i, float f) {
        Segments.writeFloat(v.memory().base(), v.byteOffset() + i * Float.BYTES, f);
    }

    // --- loading ---

    public static Qwen35Vision loadModel(
            Path label, GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(tensors, "tensors");
        String projectorType =
                gguf.getStringOrDefault(
                        "clip.vision.projector_type",
                        gguf.getStringOrDefault("clip.projector_type", ""));
        if (!"qwen3vl_merger".equals(projectorType))
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": unsupported vision projector '"
                            + projectorType
                            + "' (expected qwen3vl_merger)");

        int patchSize = gguf.getValueOrDefault(int.class, "clip.vision.patch_size", 16);
        int merge = gguf.getValueOrDefault(int.class, "clip.vision.spatial_merge_size", 2);
        int visionDim = gguf.getValueOrDefault(int.class, "clip.vision.embedding_length", 1152);
        int modelDim = gguf.getValueOrDefault(int.class, "clip.vision.projection_dim", visionDim);
        int headCount = gguf.getValueOrDefault(int.class, "clip.vision.attention.head_count", 16);
        int layerCount = gguf.getValueOrDefault(int.class, "clip.vision.block_count", 27);
        int ffnDim = gguf.getValueOrDefault(int.class, "clip.vision.feed_forward_length", 4304);
        float eps =
                gguf.getValueOrDefault(
                        float.class, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
        if (patchSize <= 0
                || visionDim <= 0
                || modelDim <= 0
                || headCount <= 0
                || layerCount <= 0
                || ffnDim <= 0
                || merge <= 1)
            throw new IllegalArgumentException(label.getFileName() + ": invalid vision metadata");

        MemoryView<MemorySegment> position = require(tensors, "v.position_embd.weight");
        Shape positionShape = position.dataType().logicalShape(position.shape());
        if (position.dataType() != DataType.FP32
                || !positionShape.isFlat()
                || positionShape.flatRank() != 2
                || positionShape.flatAt(1) != visionDim)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": v.position_embd.weight expected [positions, "
                            + visionDim
                            + "] FP32 but was "
                            + positionShape
                            + " "
                            + position.dataType().name());
        int positionSize = Math.toIntExact(positionShape.flatAt(0));
        int positionSide = (int) Math.sqrt(positionSize);
        if (positionSide * positionSide != positionSize)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": v.position_embd.weight positions "
                            + positionSize
                            + " is not a perfect square (native grid expected)");

        Layer[] layers = new Layer[layerCount];
        for (int i = 0; i < layerCount; i++) {
            String p = "v.blk." + i + ".";
            layers[i] =
                    new Layer(
                            require(tensors, p + "ln1.weight"),
                            require(tensors, p + "ln1.bias"),
                            require(tensors, p + "attn_qkv.weight"),
                            require(tensors, p + "attn_qkv.bias"),
                            require(tensors, p + "attn_out.weight"),
                            require(tensors, p + "attn_out.bias"),
                            require(tensors, p + "ln2.weight"),
                            require(tensors, p + "ln2.bias"),
                            require(tensors, p + "ffn_up.weight"),
                            require(tensors, p + "ffn_up.bias"),
                            require(tensors, p + "ffn_down.weight"),
                            require(tensors, p + "ffn_down.bias"));
        }

        int projectorInput = Math.multiplyExact(Math.multiplyExact(merge, merge), visionDim);
        MemoryView<MemorySegment> mm0Weight = require(tensors, "mm.0.weight");
        Shape mm0Shape = mm0Weight.dataType().logicalShape(mm0Weight.shape());
        if (!mm0Shape.isFlat() || mm0Shape.flatRank() != 2 || mm0Shape.flatAt(1) != projectorInput)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": mm.0.weight expected input width "
                            + projectorInput
                            + " but was "
                            + mm0Shape);
        int projectorDim = Math.toIntExact(mm0Shape.flatAt(0));
        Linear mm0 = loadLinear(tensors, "mm.0", projectorDim, projectorInput);
        Linear mm2 = loadLinear(tensors, "mm.2", modelDim, projectorDim);

        return new Qwen35Vision(
                patchSize,
                visionDim,
                modelDim,
                headCount,
                ffnDim,
                merge,
                positionSide,
                eps,
                require(tensors, "v.patch_embd.weight"),
                require(tensors, "v.patch_embd.weight.1"),
                require(tensors, "v.patch_embd.bias"),
                position,
                require(tensors, "v.post_ln.weight"),
                require(tensors, "v.post_ln.bias"),
                mm0,
                mm2,
                layers);
    }

    private void validateLayer(Layer layer, int index) {
        Objects.requireNonNull(layer, "layer " + index);
        String prefix = "v.blk." + index + ".";
        int qkvDim = 3 * visionDim;
        requireWeight(layer.qkvW(), prefix + "attn_qkv.weight", Shape.flat(qkvDim, visionDim));
        requireF32(layer.qkvB(), prefix + "attn_qkv.bias", Shape.flat(qkvDim));
        requireWeight(
                layer.attnOutW(), prefix + "attn_out.weight", Shape.flat(visionDim, visionDim));
        requireF32(layer.attnOutB(), prefix + "attn_out.bias", Shape.flat(visionDim));
        requireF32(layer.ln1W(), prefix + "ln1.weight", Shape.flat(visionDim));
        requireF32(layer.ln1B(), prefix + "ln1.bias", Shape.flat(visionDim));
        requireF32(layer.ln2W(), prefix + "ln2.weight", Shape.flat(visionDim));
        requireF32(layer.ln2B(), prefix + "ln2.bias", Shape.flat(visionDim));
        requireWeight(layer.ffnUpW(), prefix + "ffn_up.weight", Shape.flat(ffnDim, visionDim));
        requireF32(layer.ffnUpB(), prefix + "ffn_up.bias", Shape.flat(ffnDim));
        requireWeight(layer.ffnDownW(), prefix + "ffn_down.weight", Shape.flat(visionDim, ffnDim));
        requireF32(layer.ffnDownB(), prefix + "ffn_down.bias", Shape.flat(visionDim));
    }

    private static Linear loadLinear(
            Map<String, MemoryView<MemorySegment>> tensors,
            String name,
            int outputDim,
            int inputDim) {
        return requireLinear(
                new Linear(
                        require(tensors, name + ".weight"),
                        require(tensors, name + ".bias"),
                        outputDim,
                        inputDim),
                name);
    }

    private static Linear requireLinear(Linear linear, String name) {
        Objects.requireNonNull(linear, name);
        if (linear.outputDim() <= 0 || linear.inputDim() <= 0)
            throw new IllegalArgumentException(name + ": invalid dimensions");
        requireWeight(
                linear.weight(),
                name + ".weight",
                Shape.flat(linear.outputDim(), linear.inputDim()));
        requireF32(linear.bias(), name + ".bias", Shape.flat(linear.outputDim()));
        return linear;
    }

    private static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> tensors, String name) {
        MemoryView<MemorySegment> value = tensors.get(name);
        if (value == null) throw new IllegalStateException("mmproj tensor missing: " + name);
        return value;
    }

    private static MemoryView<MemorySegment> requireWeight(
            MemoryView<MemorySegment> value, String name, Shape expected) {
        Objects.requireNonNull(value, name);
        Views.requireContiguous(value, name);
        DataType type = value.dataType();
        if (type != DataType.FP32
                && type != DataType.FP16
                && type != DataType.BF16
                && type != DataType.Q8_0)
            throw new IllegalArgumentException(name + ": unsupported weight type " + type.name());
        Shape actual = type.logicalShape(value.shape());
        if (!actual.equals(expected))
            throw new IllegalArgumentException(
                    name + ": expected shape " + expected + " but was " + actual);
        return value;
    }

    private static MemoryView<MemorySegment> requirePatchF32(
            MemoryView<MemorySegment> value, String name, int outputDim, int inputDim) {
        Objects.requireNonNull(value, name);
        Views.requireDense(value, DataType.FP32, name);
        Shape actual = value.dataType().logicalShape(value.shape());
        if (!actual.isFlat()
                || actual.flatAt(0) != outputDim
                || actual.size() != (long) outputDim * inputDim)
            throw new IllegalArgumentException(
                    name
                            + ": expected output/input "
                            + outputDim
                            + "x"
                            + inputDim
                            + " but was "
                            + actual);
        return value;
    }

    private static MemoryView<MemorySegment> requireF32(
            MemoryView<MemorySegment> value, String name, Shape expected) {
        Objects.requireNonNull(value, name);
        Views.requireDense(value, DataType.FP32, name);
        if (!value.shape().equals(expected))
            throw new IllegalArgumentException(
                    name + ": expected shape " + expected + " but was " + value.shape());
        return value;
    }
}
