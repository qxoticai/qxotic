// Shared GGUF loading plumbing: parse metadata, memory-map tensors, and convert GGUF tensors into
// FloatTensor views. Architecture-agnostic - used by every model port. The legacy engine's
// arch-dispatching loader lives in jinfer-models (LegacyModelLoader).
package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.IntFunction;

interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(label + ": "
                        + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
                        + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}

public final class ModelLoader {

    private ModelLoader() {
    }

    /** Parses the GGUF metadata (com.qxotic:gguf) from the channel, leaving its position past the header. */
    public static GGUF readGguf(FileChannel fileChannel, String modelLabel) throws IOException {
        try (var ignored = Timer.log("Parse " + modelLabel)) {
            fileChannel.position(0L);
            return GGUF.read(Channels.newChannel(
                    new BufferedInputStream(Channels.newInputStream(fileChannel), 1 << 20)));
        }
    }

    /** Memory-maps the tensor data (whole-file mapping outlives the process: Arena.global). */
    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, GGUF gguf) throws IOException {
        return loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensors());
    }

    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset,
                                                    Collection<TensorEntry> tensors) throws IOException {
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset,
                fileChannel.size() - tensorDataOffset, Arena.global());
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensors.size());
        for (TensorEntry tensor : tensors) {
            int[] shape = Arrays.stream(tensor.shape()).mapToInt(Math::toIntExact).toArray();
            long sizeInBytes = tensor.ggmlType().byteSizeFor(FloatTensor.numberOfElementsLong(shape));
            MemorySegment memorySegment = tensorData.asSlice(tensor.offset(), sizeInBytes);
            tensorEntries.put(tensor.name(), new GGMLTensorEntry(tensor.name(), tensor.ggmlType(), shape, memorySegment));
        }
        return tensorEntries;
    }

    /** The optional "llama3" RoPE frequency-scaling factors ({@code rope_freqs.weight}), or null if
     *  the model uses plain RoPE. These are per-frequency divisors (1.0 for high frequencies, up to
     *  the long-context factor for low frequencies); see {@link RoPE#precomputeFreqsCisFromFreqs}. */
    public static float[] ropeFreqFactors(Map<String, GGMLTensorEntry> tensorEntries) {
        GGMLTensorEntry e = tensorEntries.get("rope_freqs.weight");
        return e == null ? null : e.memorySegment().toArray(ValueLayout.JAVA_FLOAT);
    }

    // Shared GGUF tensor-loading plumbing used by every model loader (Llama/Gemma4/Qwen35/...).

    /** Quantized tensor by name, or null if absent. */
    public static FloatTensor quantOrNull(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        return entry != null ? loadQuantized(entry) : null;
    }

    /** F32 tensor by name, or null if absent. */
    public static F32FloatTensor f32OrNull(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        return entry != null ? toF32Tensor(entry) : null;
    }

    /** First present entry among alternate tensor names (GGUF converter naming drift), or null. */
    public static GGMLTensorEntry firstPresent(Map<String, GGMLTensorEntry> entries, String... names) {
        for (String name : names) {
            GGMLTensorEntry entry = entries.get(name);
            if (entry != null) return entry;
        }
        return null;
    }

    /** Per-layer array of quantized tensors; a slot is null when its tensor is absent. */
    public static FloatTensor[] quantArray(int n, IntFunction<GGMLTensorEntry> get) {
        FloatTensor[] a = new FloatTensor[n];
        for (int i = 0; i < n; i++) {
            GGMLTensorEntry e = get.apply(i);
            a[i] = e != null ? loadQuantized(e) : null;
        }
        return a;
    }

    /** Per-layer array of F32 tensors; a slot is null when its tensor is absent. */
    public static F32FloatTensor[] f32Array(int n, IntFunction<GGMLTensorEntry> get) {
        F32FloatTensor[] a = new F32FloatTensor[n];
        for (int i = 0; i < n; i++) {
            GGMLTensorEntry e = get.apply(i);
            a[i] = e != null ? toF32Tensor(e) : null;
        }
        return a;
    }

    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        long numElements = FloatTensor.numberOfElementsLong(entry.shape());
        return switch (ggmlType) {
            case Q8_0 -> new Q8_0FloatTensor(numElements, entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(numElements, entry.memorySegment());
            case Q4_1 -> new Q4_1FloatTensor(numElements, entry.memorySegment());
            case Q5_1 -> new Q5_1FloatTensor(numElements, entry.memorySegment());
            case Q4_K -> new Q4_KFloatTensor(numElements, entry.memorySegment());
            case Q5_K -> new Q5_KFloatTensor(numElements, entry.memorySegment());
            case Q6_K -> new Q6_KFloatTensor(numElements, entry.memorySegment());
            case F32 -> new F32FloatTensor(numElements, entry.memorySegment());
            case F16 -> new F16FloatTensor(numElements, entry.memorySegment());
            case BF16 -> new BF16FloatTensor(numElements, entry.memorySegment());
            case MXFP4 -> new MXFP4FloatTensor(numElements, entry.memorySegment());
            case NVFP4 -> new NVFP4FloatTensor(numElements, entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    /** Zero-copy F32 view of a GGUF tensor (native mapped segment). */
    public static F32FloatTensor toF32Tensor(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        if (ggmlType != GGMLType.F32) {
            throw new UnsupportedOperationException("Conversion to " + ggmlType);
        }
        return new F32FloatTensor(FloatTensor.numberOfElementsLong(tensorEntry.shape()), tensorEntry.memorySegment());
    }
}
