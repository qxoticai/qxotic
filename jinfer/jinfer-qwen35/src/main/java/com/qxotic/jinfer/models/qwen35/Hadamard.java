package com.qxotic.jinfer.models.qwen35;

import static com.qxotic.jinfer.Segments.F_SPECIES;
import static com.qxotic.jinfer.Segments.USE_VECTOR_API;
import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.vectorBase;
import static com.qxotic.jinfer.Segments.vectorSegment;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import jdk.incubator.vector.FloatVector;

/**
 * Prism's folded Hadamard rotation (GGUF {@code prism.hadamard.*}, version 2, as read by
 * llama.cpp's {@code load_hparams_hadamard}). A folded weight is stored as {@code W·D·H}: {@code D}
 * a ±1 diagonal per input width, {@code H} the normalized Sylvester Walsh-Hadamard matrix over
 * blocks of the input. Both are their own inverses, so {@code W·x = W_f·(H·(D·x))}: the model is
 * unchanged, only its weights live in a basis that quantizes to ternary better. The token embedding
 * stores rotated rows, restored after lookup as {@code D·(H·z)}, and doubles as the folded output
 * head.
 */
final class Hadamard {

    private final int block;
    private final Map<Integer, float[]>
            factors; // by input width: D's signs times H's 1/sqrt(block)
    private final int headDim, keyHeads, repeats; // ssm_out regroup; repeats == 1: none

    /** {@code signs} by input width, each ±1; a width without an entry has identity signs. */
    Hadamard(
            int block,
            Map<Integer, int[]> signs,
            int[] widths,
            int headDim,
            int keyHeads,
            int repeats) {
        float scale = (float) (1 / Math.sqrt(block));
        this.block = block;
        this.factors = new HashMap<>();
        for (int width : widths) {
            float[] f = new float[width];
            int[] sign = signs.get(width);
            for (int i = 0; i < width; i++) f[i] = sign == null ? scale : sign[i] * scale;
            factors.put(width, f);
        }
        this.headDim = headDim;
        this.keyHeads = keyHeads;
        this.repeats = repeats;
    }

    /** The rotation a GGUF declares, or {@code null} for an ordinary checkpoint. */
    static Hadamard read(GGUF gguf, Qwen35.Configuration c, Set<String> tensorNames) {
        if (!gguf.containsKey("prism.hadamard.version")) return null;
        int version = gguf.getValue(int.class, "prism.hadamard.version");
        boolean tied = gguf.getValueOrDefault(boolean.class, "prism.hadamard.tied_output", false);
        if (version != 2 || !tied || tensorNames.contains("output.weight"))
            throw unsupported("version " + version + " (only 2, tied output, is supported)");
        require(gguf, "transform", "normalized-sylvester-walsh-hadamard");
        require(gguf, "axis", "input-last-dimension");
        if (c.isMoE() || c.hasMtp()) throw unsupported("MoE or MTP checkpoints");
        int block = gguf.getValue(int.class, "prism.hadamard.block_size");
        if (block <= 0 || (block & (block - 1)) != 0) throw unsupported("block_size " + block);

        Set<String> expected = new HashSet<>();
        for (int i = 0; i < c.numberOfLayers(); i++) {
            String[] kinds =
                    c.isFullAttention()[i]
                            ? new String[] {"attn_q", "attn_k", "attn_v", "attn_output"}
                            : new String[] {"attn_qkv", "attn_gate", "ssm_out"};
            for (String kind : kinds) expected.add("blk." + i + "." + kind + ".weight");
            for (String kind : new String[] {"ffn_gate", "ffn_up", "ffn_down"})
                expected.add("blk." + i + "." + kind + ".weight");
        }
        String[] folded = gguf.getValue(String[].class, "prism.hadamard.weight_names");
        if (!new HashSet<>(Arrays.asList(folded)).equals(expected))
            throw unsupported("a folded weight set other than every projection");
        String[] inverse = gguf.getValue(String[].class, "prism.hadamard.inverse_weight_names");
        if (!Arrays.equals(inverse, new String[] {"token_embd.weight"}))
            throw unsupported("inverse weights " + Arrays.toString(inverse));

        Map<Integer, int[]> signs = new HashMap<>();
        String mode = gguf.getString("prism.hadamard.sign_mode");
        if (mode.equals("explicit")) {
            int[] values = gguf.getValue(int[].class, "prism.hadamard.sign_values");
            int at = 0;
            for (int width : gguf.getValue(int[].class, "prism.hadamard.sign_widths")) {
                if (width <= 0 || at + width > values.length || signs.containsKey(width))
                    throw unsupported("sign width " + width);
                int[] sign = Arrays.copyOfRange(values, at, at + width);
                if (Arrays.stream(sign).anyMatch(v -> v != 1 && v != -1))
                    throw unsupported("sign values other than ±1");
                signs.put(width, sign);
                at += width;
            }
            if (at != values.length) throw unsupported("sign_values longer than sign_widths");
        } else if (!mode.equals("identity")) throw unsupported("sign_mode " + mode);
        int[] widths = {c.embeddingLength(), c.queryDim(), c.ssmInnerSize(), c.hiddenDim()};
        for (int width : widths)
            if (width % block != 0 || (mode.equals("explicit") && !signs.containsKey(width)))
                throw unsupported("input width " + width);

        boolean grouped =
                gguf.getValueOrDefault(boolean.class, "prism.hadamard.gdn_v_grouped", false);
        int repeats = grouped ? c.ssmTimeStepRank() / c.ssmGroupCount() : 1;
        return new Hadamard(block, signs, widths, c.headVDim(), c.ssmGroupCount(), repeats);
    }

    /** {@code dst = H·(D·src)} over {@code rows} rows of {@code width}: a folded matmul's input. */
    MemoryView<MemorySegment> forward(
            MemoryView<MemorySegment> src, MemoryView<MemorySegment> dst, int rows, int width) {
        transform(src, dst, rows, width, false, false);
        return dst;
    }

    /**
     * {@link #forward} for {@code ssm_out}: the delta net emits value heads tiled {@code [hd, nk,
     * rep]}, but the fold used the grouped order {@code [hd, rep, nk]}, so regroup first.
     */
    MemoryView<MemorySegment> forwardDeltaOut(
            MemoryView<MemorySegment> src, MemoryView<MemorySegment> dst, int rows, int width) {
        transform(src, dst, rows, width, repeats > 1, false);
        return dst;
    }

    /** In place, {@code h = D·(H·z)}: a looked-up token embedding back to the model's basis. */
    void inverse(MemoryView<MemorySegment> rows, int count, int width) {
        transform(rows, rows, count, width, false, true);
    }

    /**
     * Works in {@code dst}'s rows, so a call allocates nothing; {@code dst} may be {@code src}. One
     * job per (row, block): a decode step's single row still spreads over the pool.
     */
    private void transform(
            MemoryView<MemorySegment> src,
            MemoryView<MemorySegment> dst,
            int rows,
            int width,
            boolean regroup,
            boolean inverse) {
        float[] factor = factors.get(width);
        MemorySegment in = src.memory().base(), out = dst.memory().base();
        long inBase = src.byteOffset(), outBase = dst.byteOffset();
        long inStride = src.stride().flatAt(0) * Float.BYTES;
        long outStride = dst.stride().flatAt(0) * Float.BYTES;
        int blocks = width / block;
        Parallel.forLoop(
                rows * blocks,
                job -> {
                    long from = inBase + job / blocks * inStride;
                    long to = outBase + job / blocks * outStride;
                    int first = job % blocks * block, end = first + block;
                    if (!inverse)
                        for (int i = first; i < end; i++) {
                            float x = readFloat(in, from + 4L * (regroup ? tiled(i) : i));
                            writeFloat(out, to + 4L * i, x * factor[i]);
                        }
                    rotate(out, to + 4L * first);
                    if (inverse)
                        for (int i = first; i < end; i++)
                            writeFloat(out, to + 4L * i, readFloat(out, to + 4L * i) * factor[i]);
                });
    }

    /**
     * Element {@code i} of the grouped order {@code [hd, rep, nk]}, in the tiled {@code [hd, nk,
     * rep]}.
     */
    private int tiled(int i) {
        int h = i % headDim, group = i / headDim;
        return h + headDim * (group / repeats + keyHeads * (group % repeats));
    }

    /**
     * The unnormalized Walsh-Hadamard transform of one block at byte {@code at}, in place. With the
     * Vector API, H_block = H_(block/L) (x) H_L for L lanes: each L-wide chunk is multiplied by H_L
     * as L broadcast fmas (no lane shuffles), then the stages of span L and up are whole-vector
     * butterflies.
     */
    private void rotate(MemorySegment x, long at) {
        if (!USE_VECTOR_API || block % F_SPECIES.length() != 0) {
            for (int h = 1; h < block; h <<= 1)
                for (int i = 0; i < block; i += 2 * h)
                    for (int j = i; j < i + h; j++) {
                        long p = at + 4L * j, q = p + 4L * h;
                        float a = readFloat(x, p), b = readFloat(x, q);
                        writeFloat(x, p, a + b);
                        writeFloat(x, q, a - b);
                    }
            return;
        }
        int lanes = F_SPECIES.length();
        MemorySegment v = vectorSegment(x);
        long base = vectorBase(x) + at;
        for (int r = 0; r < block; r += lanes) {
            long o = base + 4L * r;
            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int c = 0; c < lanes; c++)
                acc =
                        FloatVector.fromArray(F_SPECIES, H_LANES, c * lanes)
                                .fma(
                                        FloatVector.broadcast(
                                                F_SPECIES, readFloat(x, at + 4L * (r + c))),
                                        acc);
            acc.intoMemorySegment(v, o, ByteOrder.LITTLE_ENDIAN);
        }
        for (int h = lanes; h < block; h <<= 1)
            for (int i = 0; i < block; i += 2 * h)
                for (int j = i; j < i + h; j += lanes) {
                    long p = base + 4L * j, q = p + 4L * h;
                    FloatVector a =
                            FloatVector.fromMemorySegment(F_SPECIES, v, p, ByteOrder.LITTLE_ENDIAN);
                    FloatVector b =
                            FloatVector.fromMemorySegment(F_SPECIES, v, q, ByteOrder.LITTLE_ENDIAN);
                    a.add(b).intoMemorySegment(v, p, ByteOrder.LITTLE_ENDIAN);
                    a.sub(b).intoMemorySegment(v, q, ByteOrder.LITTLE_ENDIAN);
                }
    }

    /** H_L for the vector's L lanes, row-major: row c is the vector that element c contributes. */
    private static final float[] H_LANES = hadamard(USE_VECTOR_API ? F_SPECIES.length() : 1);

    private static float[] hadamard(int n) {
        float[] h = new float[n * n];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++) h[r * n + c] = Integer.bitCount(r & c) % 2 == 0 ? 1 : -1;
        return h;
    }

    private static void require(GGUF gguf, String key, String value) {
        String actual = gguf.getString("prism.hadamard." + key);
        if (!actual.equals(value)) throw unsupported(key + " " + actual);
    }

    private static IllegalArgumentException unsupported(String what) {
        return new IllegalArgumentException("Qwen3.5: unsupported prism.hadamard " + what);
    }
}
