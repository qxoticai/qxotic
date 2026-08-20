package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Norms;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Reference;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/** Gemma 4 raw-waveform audio projector ({@code projector_type=gemma4ua}). */
public final class Gemma4Audio implements MediaProjector<Media.Audio> {
    private final int modelDim, frameSize;
    private final float eps;
    private final MemoryView<MemorySegment> inputProjection;

    Gemma4Audio(int modelDim, float eps, MemoryView<MemorySegment> inputProjection) {
        if (modelDim <= 0) throw new IllegalArgumentException("modelDim must be positive");
        if (!(eps > 0) || !Float.isFinite(eps))
            throw new IllegalArgumentException("RMS epsilon must be finite and positive");
        this.modelDim = modelDim;
        this.eps = eps;
        MemoryView<MemorySegment> projection =
                Objects.requireNonNull(inputProjection, "inputProjection");
        Views.requireContiguous(projection, "mm.a.input_projection.weight");
        DataType type = projection.dataType();
        Shape shape = type.logicalShape(projection.shape());
        if ((type != DataType.FP32 && type != DataType.BF16 && type != DataType.Q8_0)
                || !shape.isFlat()
                || shape.flatAt(0) != modelDim
                || shape.size() % modelDim != 0)
            throw new IllegalArgumentException(
                    "mm.a.input_projection.weight: expected "
                            + modelDim
                            + " output rows but was "
                            + shape
                            + " "
                            + type);
        this.frameSize = Math.toIntExact(shape.size() / modelDim);
        this.inputProjection = projection;
    }

    @Override
    public int positions(Media.Audio audio) {
        int samples = AudioPreprocess.mono16kLength(audio);
        return Math.max(1, Math.toIntExact(((long) samples + frameSize - 1) / frameSize));
    }

    @Override
    public String planId() {
        return "gemma4ua frame=" + frameSize;
    }

    @Override
    public void project(Media.Audio audio, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(audio, "audio");
        Objects.requireNonNull(sink, "sink");
        if (maxChunkSize <= 0) throw new IllegalArgumentException("maxChunkSize must be positive");
        Views.checkAlive(inputProjection, "inputProjection"); // fail-fast on freed weights

        float[] pcm = AudioPreprocess.toMono16k(audio);
        int rows = Math.max(1, Math.toIntExact(((long) pcm.length + frameSize - 1) / frameSize));
        for (int firstRow = 0; firstRow < rows; ) {
            int chunkRows = Math.min(maxChunkSize, rows - firstRow);
            MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
            try {
                MemoryView<MemorySegment> frames = Views.allocateF32(scratch, chunkRows, frameSize);
                int rowBase = firstRow;
                Parallel.forRows(
                        chunkRows,
                        row -> {
                            int sourceBase = Math.multiplyExact(rowBase + row, frameSize);
                            long destinationBase = (long) row * frameSize;
                            int count = Math.min(frameSize, Math.max(0, pcm.length - sourceBase));
                            Views.copyFromArray(
                                    frames,
                                    destinationBase,
                                    pcm,
                                    sourceBase,
                                    count,
                                    "audio frames");
                            Norms.rmsnormNoWeight(
                                    frames,
                                    destinationBase,
                                    frames,
                                    destinationBase,
                                    frameSize,
                                    eps);
                        });
                MemoryView<MemorySegment> projected =
                        Views.allocateF32(scratch, chunkRows, modelDim);
                MatMul.gemm(
                        inputProjection,
                        frames,
                        frameSize,
                        projected,
                        modelDim,
                        modelDim,
                        chunkRows,
                        frameSize);
                sink.accept(projected);
            } finally {
                Arenas.close(scratch);
            }
            firstRow += chunkRows;
        }
        Reference.reachabilityFence(this); // kernels read weights via raw bases; pin `this`
    }

    public static Gemma4Audio loadModel(Path path, Arena arena) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(arena, "arena");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return loadModel(path, gguf, ModelLoader.loadTensors(channel, gguf, arena));
        }
    }

    public static Gemma4Audio loadModel(GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        return loadModel(Path.of("mmproj.gguf"), gguf, tensors);
    }

    public static Gemma4Audio loadModel(
            Path label, GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(tensors, "tensors");
        String type = gguf.getStringOrDefault("clip.audio.projector_type", "");
        if (!"gemma4ua".equals(type))
            throw new IllegalArgumentException(
                    label.getFileName()
                            + ": expected clip.audio.projector_type=gemma4ua but was '"
                            + type
                            + "'");
        int modelDim = gguf.getValueOrDefault(int.class, "clip.audio.projection_dim", 3840);
        float eps =
                gguf.getValueOrDefault(
                        float.class, "clip.audio.attention.layer_norm_epsilon", 1e-6f);
        MemoryView<MemorySegment> projection = tensors.get("mm.a.input_projection.weight");
        if (projection == null)
            throw new IllegalStateException(
                    label.getFileName() + ": mmproj tensor missing: mm.a.input_projection.weight");
        return new Gemma4Audio(modelDim, eps, projection);
    }
}
