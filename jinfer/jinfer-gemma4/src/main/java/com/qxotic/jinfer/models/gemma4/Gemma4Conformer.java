package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jinfer.kernels.FastMath;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/** Gemma 4 Conformer audio tower ({@code projector_type=gemma4a}). */
public final class Gemma4Conformer implements MediaProjector<Media.Audio> {
    static final int CHUNK = 12;
    static final int PAST = 12;
    static final int CONTEXT = CHUNK + PAST;
    static final int RPE = PAST + 1;
    private static final int CONV0_CHANNELS = 128;
    private static final int CONV1_CHANNELS = 32;
    private static final int CONV_KERNEL = 5;

    record Block(
            MemoryView<MemorySegment> ffNorm,
            Clamped ffUp,
            Clamped ffDown,
            MemoryView<MemorySegment> ffPostNorm,
            MemoryView<MemorySegment> attnPreNorm,
            Clamped query,
            Clamped key,
            Clamped value,
            Clamped attentionOutput,
            MemoryView<MemorySegment> attnPostNorm,
            MemoryView<MemorySegment> relative,
            MemoryView<MemorySegment> perDimScale,
            MemoryView<MemorySegment> convPreNorm,
            Clamped convPointwise1,
            MemoryView<MemorySegment> convDepthwise,
            MemoryView<MemorySegment> convPostNorm,
            Clamped convPointwise2,
            MemoryView<MemorySegment> ffNorm1,
            Clamped ffUp1,
            Clamped ffDown1,
            MemoryView<MemorySegment> ffPostNorm1,
            MemoryView<MemorySegment> outputNorm) {}

    private final int dim, heads, headDim, ffDim, nMel, outputDim;
    private final float eps;
    private final AudioPreprocess preprocess;
    private final float[] conv0, conv1;
    private final MemoryView<MemorySegment> norm0, norm1, outputBias;
    private final Clamped inputProjection, outputProjection, modelProjection;
    private final Block[] blocks;

    Gemma4Conformer(
            int dim,
            int heads,
            int ffDim,
            int nMel,
            int outputDim,
            float eps,
            float[] conv0,
            MemoryView<MemorySegment> norm0,
            float[] conv1,
            MemoryView<MemorySegment> norm1,
            Clamped inputProjection,
            Block[] blocks,
            Clamped outputProjection,
            MemoryView<MemorySegment> outputBias,
            Clamped modelProjection) {
        validateArchitecture(Path.of("gemma4a"), dim, heads, nMel);
        if (ffDim <= 0 || outputDim <= 0 || !(eps > 0f) || !Float.isFinite(eps))
            throw new IllegalArgumentException("invalid gemma4a dimensions or epsilon");
        this.dim = dim;
        this.heads = heads;
        this.headDim = dim / heads;
        this.ffDim = ffDim;
        this.nMel = nMel;
        this.outputDim = outputDim;
        this.eps = eps;
        this.preprocess = new AudioPreprocess(nMel);
        this.conv0 = requireTaps(conv0, 9 * CONV0_CHANNELS, "a.conv1d.0.weight");
        this.norm0 = requireF32(norm0, "a.conv1d.0.norm.weight", CONV0_CHANNELS);
        this.conv1 = requireTaps(conv1, 9 * CONV0_CHANNELS * CONV1_CHANNELS, "a.conv1d.1.weight");
        this.norm1 = requireF32(norm1, "a.conv1d.1.norm.weight", CONV1_CHANNELS);
        this.inputProjection = requireClamped(inputProjection, "a.input_projection", dim, dim);
        this.blocks = Objects.requireNonNull(blocks, "blocks").clone();
        for (int i = 0; i < this.blocks.length; i++) validateBlock(this.blocks[i], i);
        this.outputProjection =
                requireClamped(outputProjection, "a.pre_encode.out", outputDim, dim);
        this.outputBias = requireF32(outputBias, "a.pre_encode.out.bias", outputDim);
        this.modelProjection =
                requireClamped(modelProjection, "mm.a.input_projection", outputDim, outputDim);
    }

    @Override
    public int positions(Media.Audio audio) {
        int samples = AudioPreprocess.mono16kLength(audio), total = 0;
        for (int offset = 0; offset < samples; offset += AudioPreprocess.CHUNK_SAMPLES) {
            int length = Math.min(AudioPreprocess.CHUNK_SAMPLES, samples - offset);
            total += tokensForFrames(AudioPreprocess.framesFor(length));
        }
        return Math.max(1, total);
    }

    @Override
    public String planId() {
        return "gemma4a mel=" + nMel + " dim=" + dim + " blocks=" + blocks.length;
    }

    static int tokensForFrames(int frames) {
        int once = (frames - 1) / 2 + 1;
        return (once - 1) / 2 + 1;
    }

    @Override
    public void project(Media.Audio audio, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(audio, "audio");
        Objects.requireNonNull(sink, "sink");
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        List<AudioPreprocess.MelChunk> mels = preprocess.logMel(audio);
        if (mels.isEmpty()) mels = List.of(new AudioPreprocess.MelChunk(new float[0], 0));
        for (AudioPreprocess.MelChunk mel : mels) {
            MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
            try {
                MemoryView<MemorySegment> rows = forward(mel, scratch);
                int count = Math.toIntExact(rows.shape().flatAt(0));
                for (int first = 0; first < count; first += maxChunkSize)
                    sink.accept(rows.slice(0, first, Math.min(count, first + maxChunkSize)));
            } finally {
                Arenas.close(scratch);
            }
        }
    }

    private MemoryView<MemorySegment> forward(
            AudioPreprocess.MelChunk mel, MemoryArena<MemorySegment> scratch) {
        int frames = mel.frames();
        int time2 = (frames - 1) / 2 + 1;
        int rows = tokensForFrames(frames);
        int frequency2 = nMel / 2, frequency4 = nMel / 4;
        MemoryView<MemorySegment> melView = Views.allocateF32(scratch, Math.max(0, frames * nMel));
        Views.copyFromArray(melView, 0, mel.data(), 0, mel.data().length, "mel");

        MemoryView<MemorySegment> c0 =
                Views.allocateF32(scratch, CONV0_CHANNELS * time2 * frequency2);
        Convolutions.conv2dStride2Pad1(melView, frames, nMel, 1, conv0, CONV0_CHANNELS, c0);
        Norms.layerNormChannelsReluInPlace(c0, norm0, CONV0_CHANNELS, time2 * frequency2, 1e-6f);
        MemoryView<MemorySegment> c1 =
                Views.allocateF32(scratch, CONV1_CHANNELS * rows * frequency4);
        Convolutions.conv2dStride2Pad1(
                c0, time2, frequency2, CONV0_CHANNELS, conv1, CONV1_CHANNELS, c1);
        Norms.layerNormChannelsReluInPlace(c1, norm1, CONV1_CHANNELS, rows * frequency4, 1e-6f);

        MemoryView<MemorySegment> flat = Views.allocateF32(scratch, rows, dim);
        Ops.channelLastCopy(c1, CONV1_CHANNELS, rows, frequency4, flat);

        Scratch work = Scratch.allocate(scratch, rows, dim, ffDim, outputDim);
        inputProjection.gemm(flat, dim, work.x, dim, rows, work.clamp);
        for (Block block : blocks) {
            halfFfn(work.x, block.ffNorm, block.ffUp, block.ffDown, block.ffPostNorm, rows, work);
            attention(work.x, block, rows, work);
            convolution(work.x, block, rows, work);
            halfFfn(
                    work.x,
                    block.ffNorm1,
                    block.ffUp1,
                    block.ffDown1,
                    block.ffPostNorm1,
                    rows,
                    work);
            Norms.rmsnormRows(work.x, work.x, block.outputNorm, rows, dim, eps);
        }
        outputProjection.gemm(work.x, dim, work.projected, outputDim, rows, work.clamp);
        Ops.addRowBiasInPlace(work.projected, 0, outputBias, 0, rows, outputDim);
        Parallel.forRows(
                rows,
                row ->
                        Norms.rmsnormNoWeight(
                                work.projected,
                                (long) row * outputDim,
                                work.projected,
                                (long) row * outputDim,
                                outputDim,
                                eps));
        modelProjection.gemm(work.projected, outputDim, work.output, outputDim, rows, work.clamp);
        return work.output;
    }

    private record Scratch(
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> norm,
            MemoryView<MemorySegment> ff,
            MemoryView<MemorySegment> ffOutput,
            MemoryView<MemorySegment> query,
            MemoryView<MemorySegment> key,
            MemoryView<MemorySegment> value,
            MemoryView<MemorySegment> attention,
            MemoryView<MemorySegment> pointwise,
            MemoryView<MemorySegment> glu,
            MemoryView<MemorySegment> projected,
            MemoryView<MemorySegment> output,
            MemoryView<MemorySegment> clamp) {
        static Scratch allocate(
                MemoryArena<MemorySegment> arena, int rows, int dim, int ffDim, int outputDim) {
            int max = Math.max(Math.max(dim * 2, ffDim), outputDim);
            return new Scratch(
                    Views.allocateF32(arena, rows, dim),
                    Views.allocateF32(arena, rows, dim),
                    Views.allocateF32(arena, rows, ffDim),
                    Views.allocateF32(arena, rows, dim),
                    Views.allocateF32(arena, rows, dim),
                    Views.allocateF32(arena, rows, dim),
                    Views.allocateF32(arena, rows, dim),
                    Views.allocateF32(arena, rows, dim),
                    Views.allocateF32(arena, rows, dim * 2),
                    Views.allocateF32(arena, rows, dim),
                    Views.allocateF32(arena, rows, outputDim),
                    Views.allocateF32(arena, rows, outputDim),
                    Views.allocateF32(arena, rows, max));
        }
    }

    private void halfFfn(
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> preNorm,
            Clamped up,
            Clamped down,
            MemoryView<MemorySegment> postNorm,
            int rows,
            Scratch work) {
        Norms.rmsnormRows(work.norm, x, preNorm, rows, dim, eps);
        up.gemm(work.norm, dim, work.ff, ffDim, rows, work.clamp);
        Parallel.forRows(rows, row -> Ops.siluInPlace(work.ff, (long) row * ffDim, ffDim));
        down.gemm(work.ff, ffDim, work.ffOutput, dim, rows, work.clamp);
        Norms.rmsnormRows(work.ffOutput, work.ffOutput, postNorm, rows, dim, eps);
        Parallel.forRows(
                rows,
                row ->
                        Ops.saxpyInPlace(
                                x, (long) row * dim, work.ffOutput, (long) row * dim, dim, 0.5f));
    }

    private void attention(MemoryView<MemorySegment> x, Block block, int rows, Scratch work) {
        Norms.rmsnormRows(work.norm, x, block.attnPreNorm, rows, dim, eps);
        block.query.gemm(work.norm, dim, work.query, dim, rows, work.clamp);
        block.key.gemm(work.norm, dim, work.key, dim, rows, work.clamp);
        block.value.gemm(work.norm, dim, work.value, dim, rows, work.clamp);
        float queryScale = (float) ((1.0 / Math.sqrt(headDim)) / Math.log(2.0));
        float keyScale = (float) (Math.log1p(Math.exp(1.0)) / Math.log(2.0));
        Parallel.forRows(
                rows,
                row -> {
                    long base = (long) row * dim;
                    for (int head = 0; head < heads; head++) {
                        long at = base + (long) head * headDim;
                        Norms.scaleByWeight(
                                work.query,
                                at,
                                work.query,
                                at,
                                block.perDimScale,
                                headDim,
                                queryScale);
                        Ops.multiplyInPlace(work.key, at, headDim, keyScale);
                    }
                });

        int blockCount = (rows + CHUNK - 1) / CHUNK;
        Parallel.parallelFor(
                0,
                blockCount * heads,
                unit -> {
                    int chunk = unit / heads, head = unit % heads, headBase = head * headDim;
                    float[] scores = new float[CONTEXT];
                    for (int localQuery = 0; localQuery < CHUNK; localQuery++) {
                        int queryRow = chunk * CHUNK + localQuery;
                        if (queryRow >= rows) break;
                        long queryOffset = (long) queryRow * dim + headBase;
                        for (int slot = 0; slot < CONTEXT; slot++) {
                            int keyRow = chunk * CHUNK - PAST + slot;
                            boolean valid =
                                    keyRow >= 0
                                            && keyRow < rows
                                            && keyRow <= queryRow
                                            && queryRow - keyRow < PAST;
                            if (!valid) {
                                scores[slot] = -1e9f;
                                continue;
                            }
                            float content =
                                    Ops.dot(
                                            work.query,
                                            queryOffset,
                                            work.key,
                                            (long) keyRow * dim + headBase,
                                            headDim);
                            int relativePosition = slot - localQuery;
                            float position =
                                    relativePosition >= 0 && relativePosition < RPE
                                            ? Ops.dot(
                                                    work.query,
                                                    queryOffset,
                                                    block.relative,
                                                    (long) relativePosition * dim + headBase,
                                                    headDim)
                                            : 0f;
                            scores[slot] = 50f * FastMath.tanh((content + position) / 50f);
                        }
                        float maximum = Float.NEGATIVE_INFINITY;
                        for (float score : scores) maximum = Math.max(maximum, score);
                        float sum = 0f;
                        for (int slot = 0; slot < CONTEXT; slot++) {
                            scores[slot] = FastMath.expNeg(scores[slot] - maximum);
                            sum += scores[slot];
                        }
                        float inverse = 1f / sum;
                        long outputOffset = (long) queryRow * dim + headBase;
                        Ops.fillInPlace(work.attention, outputOffset, headDim, 0f);
                        for (int slot = 0; slot < CONTEXT; slot++) {
                            int keyRow = chunk * CHUNK - PAST + slot;
                            if (keyRow >= 0 && keyRow < rows && scores[slot] != 0f)
                                Ops.saxpyInPlace(
                                        work.attention,
                                        outputOffset,
                                        work.value,
                                        (long) keyRow * dim + headBase,
                                        headDim,
                                        scores[slot] * inverse);
                        }
                    }
                });
        block.attentionOutput.gemm(work.attention, dim, work.norm, dim, rows, work.clamp);
        Norms.rmsnormRows(work.norm, work.norm, block.attnPostNorm, rows, dim, eps);
        Ops.addInPlace(x, 0, work.norm, 0, Math.multiplyExact(rows, dim));
    }

    private void convolution(MemoryView<MemorySegment> x, Block block, int rows, Scratch work) {
        Norms.rmsnormRows(work.norm, x, block.convPreNorm, rows, dim, eps);
        block.convPointwise1.gemm(work.norm, dim, work.pointwise, dim * 2, rows, work.clamp);
        Parallel.forRows(
                rows,
                row ->
                        Activations.glu(
                                work.glu,
                                (long) row * dim,
                                work.pointwise,
                                (long) row * dim * 2,
                                dim));
        Convolutions.causalDepthwise1d(
                work.glu, block.convDepthwise, work.norm, rows, dim, CONV_KERNEL);
        Norms.rmsnormRows(work.norm, work.norm, block.convPostNorm, rows, dim, eps);
        Parallel.forRows(rows, row -> Ops.siluInPlace(work.norm, (long) row * dim, dim));
        block.convPointwise2.gemm(work.norm, dim, work.glu, dim, rows, work.clamp);
        Ops.addInPlace(x, 0, work.glu, 0, Math.multiplyExact(rows, dim));
    }

    static float[] buildPositionEmbedding(int dim) {
        int half = dim / 2;
        float logIncrement = (float) (Math.log(10000.0) / Math.max(half - 1, 1));
        float[] embedding = new float[RPE * dim];
        for (int positionIndex = 0; positionIndex < RPE; positionIndex++) {
            float position = PAST - positionIndex;
            for (int i = 0; i < half; i++) {
                float angle = position * (float) Math.exp(-i * logIncrement);
                embedding[positionIndex * dim + i] = (float) Math.sin(angle);
                embedding[positionIndex * dim + i + half] = (float) Math.cos(angle);
            }
        }
        return embedding;
    }

    public static Gemma4Conformer loadModel(Path path, Arena arena) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(arena, "arena");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(path, gguf, ModelLoader.loadTensors(channel, gguf, arena), arena);
        }
    }

    public static Gemma4Conformer loadModel(
            GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, Arena arena) {
        return loadModel(Path.of("mmproj.gguf"), gguf, tensors, arena);
    }

    public static Gemma4Conformer loadModel(
            Path label, GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors, Arena arena) {
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(tensors, "tensors");
        Objects.requireNonNull(arena, "arena");
        String type = gguf.getStringOrDefault("clip.audio.projector_type", "");
        if (!"gemma4a".equals(type))
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": expected clip.audio.projector_type=gemma4a but was '"
                            + type
                            + "'");
        int dim = gguf.getValue(int.class, "clip.audio.embedding_length");
        int heads = gguf.getValue(int.class, "clip.audio.attention.head_count");
        int ffDim = gguf.getValue(int.class, "clip.audio.feed_forward_length");
        int blockCount = gguf.getValue(int.class, "clip.audio.block_count");
        int nMel = gguf.getValue(int.class, "clip.audio.num_mel_bins");
        int outputDim = gguf.getValue(int.class, "clip.audio.projection_dim");
        float eps =
                gguf.getValueOrDefault(
                        float.class, "clip.audio.attention.layer_norm_epsilon", 1e-6f);
        if (blockCount < 0) throw new IllegalArgumentException("block_count must not be negative");
        validateArchitecture(label, dim, heads, nMel);

        PanamaMemoryArena persistent = new PanamaMemoryArena(arena);
        MemoryView<MemorySegment> positions =
                Views.fromFloatArray(persistent, buildPositionEmbedding(dim))
                        .view(Shape.flat(RPE, dim));

        Block[] blocks = new Block[blockCount];
        for (int i = 0; i < blockCount; i++) {
            String prefix = "a.blk." + i + ".";
            MemoryView<MemorySegment> relative = Views.allocateF32(persistent, RPE, dim);
            MatMul.gemm(
                    Gemma4VisionUnified.requireWeight(
                            tensors, prefix + "attn_k_rel.weight", Shape.flat(dim, dim)),
                    positions,
                    relative,
                    RPE);
            blocks[i] =
                    new Block(
                            requireF32(tensors, prefix + "ffn_norm.weight", dim),
                            Clamped.load(tensors, prefix + "ffn_up", ffDim, dim),
                            Clamped.load(tensors, prefix + "ffn_down", dim, ffDim),
                            requireF32(tensors, prefix + "ffn_post_norm.weight", dim),
                            requireF32(tensors, prefix + "attn_pre_norm.weight", dim),
                            Clamped.load(tensors, prefix + "attn_q", dim, dim),
                            Clamped.load(tensors, prefix + "attn_k", dim, dim),
                            Clamped.load(tensors, prefix + "attn_v", dim, dim),
                            Clamped.load(tensors, prefix + "attn_out", dim, dim),
                            requireF32(tensors, prefix + "attn_post_norm.weight", dim),
                            relative,
                            requireF32(tensors, prefix + "per_dim_scale.weight", dim / heads),
                            requireF32(tensors, prefix + "conv_norm.weight", dim),
                            Clamped.load(tensors, prefix + "conv_pw1", dim * 2, dim),
                            requireF32(tensors, prefix + "conv_dw.weight", dim, CONV_KERNEL),
                            requireF32(tensors, prefix + "norm_conv.weight", dim),
                            Clamped.load(tensors, prefix + "conv_pw2", dim, dim),
                            requireF32(tensors, prefix + "ffn_norm_1.weight", dim),
                            Clamped.load(tensors, prefix + "ffn_up_1", ffDim, dim),
                            Clamped.load(tensors, prefix + "ffn_down_1", dim, ffDim),
                            requireF32(tensors, prefix + "ffn_post_norm_1.weight", dim),
                            requireF32(tensors, prefix + "ln2.weight", dim));
        }
        return new Gemma4Conformer(
                dim,
                heads,
                ffDim,
                nMel,
                outputDim,
                eps,
                taps(tensors, "a.conv1d.0.weight", persistent, CONV0_CHANNELS, 1, 3, 3),
                requireF32(tensors, "a.conv1d.0.norm.weight", CONV0_CHANNELS),
                taps(
                        tensors,
                        "a.conv1d.1.weight",
                        persistent,
                        CONV1_CHANNELS,
                        CONV0_CHANNELS,
                        3,
                        3),
                requireF32(tensors, "a.conv1d.1.norm.weight", CONV1_CHANNELS),
                Clamped.load(tensors, "a.input_projection", dim, dim),
                blocks,
                Clamped.load(tensors, "a.pre_encode.out", outputDim, dim),
                requireF32(tensors, "a.pre_encode.out.bias", outputDim),
                Clamped.load(tensors, "mm.a.input_projection", outputDim, outputDim));
    }

    static void validateArchitecture(Path label, int dim, int heads, int nMel) {
        int flattened = nMel > 0 ? (nMel / 4) * CONV1_CHANNELS : 0;
        if (nMel % 4 != 0 || flattened != dim || heads <= 0 || dim % heads != 0 || dim % 2 != 0)
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": unsupported gemma4a geometry (mel bins "
                            + nMel
                            + ", flattened "
                            + flattened
                            + ", encoder width "
                            + dim
                            + ", heads "
                            + heads
                            + ")");
    }

    private void validateBlock(Block block, int index) {
        Objects.requireNonNull(block, "block " + index);
        String prefix = "a.blk." + index + ".";
        requireF32(block.ffNorm, prefix + "ffn_norm.weight", dim);
        requireClamped(block.ffUp, prefix + "ffn_up", ffDim, dim);
        requireClamped(block.ffDown, prefix + "ffn_down", dim, ffDim);
        requireF32(block.ffPostNorm, prefix + "ffn_post_norm.weight", dim);
        requireF32(block.attnPreNorm, prefix + "attn_pre_norm.weight", dim);
        requireClamped(block.query, prefix + "attn_q", dim, dim);
        requireClamped(block.key, prefix + "attn_k", dim, dim);
        requireClamped(block.value, prefix + "attn_v", dim, dim);
        requireClamped(block.attentionOutput, prefix + "attn_out", dim, dim);
        requireF32(block.attnPostNorm, prefix + "attn_post_norm.weight", dim);
        Gemma4VisionUnified.requireF32(block.relative, prefix + "relative", Shape.flat(RPE, dim));
        requireF32(block.perDimScale, prefix + "per_dim_scale.weight", headDim);
        requireF32(block.convPreNorm, prefix + "conv_norm.weight", dim);
        requireClamped(block.convPointwise1, prefix + "conv_pw1", dim * 2, dim);
        requireF32(block.convDepthwise, prefix + "conv_dw.weight", dim, CONV_KERNEL);
        requireF32(block.convPostNorm, prefix + "norm_conv.weight", dim);
        requireClamped(block.convPointwise2, prefix + "conv_pw2", dim, dim);
        requireF32(block.ffNorm1, prefix + "ffn_norm_1.weight", dim);
        requireClamped(block.ffUp1, prefix + "ffn_up_1", ffDim, dim);
        requireClamped(block.ffDown1, prefix + "ffn_down_1", dim, ffDim);
        requireF32(block.ffPostNorm1, prefix + "ffn_post_norm_1.weight", dim);
        requireF32(block.outputNorm, prefix + "ln2.weight", dim);
    }

    private static Clamped requireClamped(Clamped value, String name, int outputDim, int inputDim) {
        Objects.requireNonNull(value, name);
        Gemma4VisionUnified.requireWeight(
                value.weight(), name + ".weight", Shape.flat(outputDim, inputDim));
        if (value.inputMin() > value.inputMax() || value.outputMin() > value.outputMax())
            throw new IllegalArgumentException(name + ": invalid clamp bounds");
        return value;
    }

    private static MemoryView<MemorySegment> requireF32(
            Map<String, MemoryView<MemorySegment>> tensors, String name, long... dimensions) {
        MemoryView<MemorySegment> value = tensors.get(name);
        if (value == null) throw new IllegalStateException("mmproj tensor missing: " + name);
        return Gemma4VisionUnified.requireF32(value, name, Shape.flat(dimensions));
    }

    private static MemoryView<MemorySegment> requireF32(
            MemoryView<MemorySegment> value, String name, long... dimensions) {
        return Gemma4VisionUnified.requireF32(value, name, Shape.flat(dimensions));
    }

    private static float[] taps(
            Map<String, MemoryView<MemorySegment>> tensors,
            String name,
            PanamaMemoryArena arena,
            long... dimensions) {
        Shape shape = Shape.flat(dimensions);
        int count = Math.toIntExact(shape.size());
        MemoryView<MemorySegment> weight = Gemma4VisionUnified.requireWeight(tensors, name, shape);
        MemoryView<MemorySegment> decoded = Views.allocateF32(arena, count);
        Convert.copyToF32(weight, 0, decoded, 0, count);
        return Views.toFloatArray(decoded, name);
    }

    private static float[] requireTaps(float[] values, int count, String name) {
        Objects.requireNonNull(values, name);
        if (values.length != count)
            throw new IllegalArgumentException(
                    name + ": expected " + count + " elements but was " + values.length);
        return values.clone();
    }
}
