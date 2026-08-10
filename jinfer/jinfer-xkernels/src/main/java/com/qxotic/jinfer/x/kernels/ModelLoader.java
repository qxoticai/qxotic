// Shared GGUF loading plumbing: parse metadata, memory-map tensors, and expose them as
// MemoryView<MemorySegment> weights (one jota Memory over the READ_ONLY mmap, one view per
// tensor). Ported from jinfer-kernels ModelLoader: FloatTensor.create + flat shapes become
// typed views carrying real Shapes; dtypes are restricted to the cycle-1 scope {Q8_0, F32, F16}
// — anything else fails at LOAD time, not inside a kernel.
package com.qxotic.jinfer.x.kernels;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public final class ModelLoader {

    private ModelLoader() {}

    /**
     * The cycle-1 dtype scope: Q8_0 weights, F32 norms/taps, F16 KV. Anything else throws HERE
     * (load time) — the old tree discovered unsupported quants inside a kernel call.
     */
    public static DataType dataType(GGMLType ggmlType) {
        return switch (ggmlType) {
            case F32 -> DataType.FP32;
            case F16 -> DataType.FP16;
            case Q8_0 -> DataType.Q8_0;
            default ->
                    throw new UnsupportedOperationException(
                            "GGMLType " + ggmlType + " outside the cycle-1 scope {Q8_0, F32, F16}");
        };
    }

    /**
     * Parses the GGUF metadata (com.qxotic:gguf) from the channel, leaving its position past the
     * header.
     */
    public static GGUF readGguf(FileChannel fileChannel, String modelLabel) throws IOException {
        try (var ignored = Timer.log("Parse " + modelLabel)) {
            fileChannel.position(0L);
            return GGUF.read(
                    Channels.newChannel(
                            new BufferedInputStream(
                                    Channels.newInputStream(fileChannel), 1 << 20)));
        }
    }

    /**
     * Memory-maps the tensor data in one READ_ONLY mapping (shared by every tensor view) into the
     * caller's {@code arena} - who provides the arena owns the weights' lifetime ({@code ofAuto} =
     * unmapped by GC once the model graph drops; {@code global} = process lifetime; a scoped arena
     * = deterministic unmap, which must outlive every model sharing these weights). File-backed
     * READ_ONLY pages are kernel-reclaimable under memory pressure regardless of arena choice.
     * (Kernels read via raw addresses ({@code Segments.GLOBAL_SEGMENT}) that bypass liveness
     * checks; ports run {@code Views.checkAlive} once per forward on the weight/KV views.)
     */
    public static Map<String, MemoryView<MemorySegment>> loadTensors(
            FileChannel fileChannel, GGUF gguf, Arena arena) throws IOException {
        return loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensors(), arena);
    }

    /**
     * Like {@link #loadTensors(FileChannel, GGUF, Arena)} but tensor data is at {@code baseOffset
     * +} the GGUF tensor data offset — used for self-archives where the GGUF blob is embedded at a
     * non-zero position in the file.
     */
    public static Map<String, MemoryView<MemorySegment>> loadTensors(
            FileChannel fileChannel, GGUF gguf, long baseOffset, Arena arena) throws IOException {
        return loadTensors(
                fileChannel, baseOffset + gguf.getTensorDataOffset(), gguf.getTensors(), arena);
    }

    public static Map<String, MemoryView<MemorySegment>> loadTensors(
            FileChannel fileChannel,
            long tensorDataOffset,
            Collection<TensorEntry> tensors,
            Arena arena)
            throws IOException {
        MemorySegment tensorData =
                fileChannel.map(
                        FileChannel.MapMode.READ_ONLY,
                        tensorDataOffset,
                        fileChannel.size() - tensorDataOffset,
                        arena);
        // ONE jota Memory over the whole mapping; each tensor is a byte-offset view into it
        // (replaces FloatTensor.create over per-tensor asSlice segments).
        Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(tensorData);
        Map<String, MemoryView<MemorySegment>> tensorViews = HashMap.newHashMap(tensors.size());
        for (TensorEntry tensor : tensors) {
            DataType dtype = dataType(tensor.ggmlType()); // scope guard: throws on unsupported
            // GGUF dims are FASTEST-first (shape[0] is the contiguous dim); a jota row-major
            // layout wants slowest-first, so reverse, then let the block dtype fold the last
            // (contiguous) dim into blocks (physicalShape).
            long[] ggufShape = tensor.shape();
            long[] dims = new long[ggufShape.length];
            for (int i = 0; i < ggufShape.length; i++) {
                dims[i] = ggufShape[ggufShape.length - 1 - i];
            }
            Shape physical = dtype.physicalShape(Shape.flat(dims));
            long elements = Shape.flat(dims).size();
            assert dtype.byteSizeFor(physical) == tensor.ggmlType().byteSizeFor(elements)
                    : tensor.name() + ": view byte size disagrees with GGUF";
            tensorViews.put(
                    tensor.name(),
                    MemoryView.of(memory, tensor.offset(), dtype, Layout.rowMajor(physical)));
        }
        return tensorViews;
    }

    /**
     * The optional "llama3" RoPE frequency-scaling factors ({@code rope_freqs.weight}), or null if
     * the model uses plain RoPE. These are per-frequency divisors (1.0 for high frequencies, up to
     * the long-context factor for low frequencies); see {@code RoPE.withFreqFactors}.
     */
    public static float[] ropeFreqFactors(Map<String, MemoryView<MemorySegment>> tensorViews) {
        MemoryView<MemorySegment> e = tensorViews.get("rope_freqs.weight");
        if (e == null) return null;
        long n = e.logicalSize();
        return e.memory()
                .base()
                .asSlice(e.byteOffset(), n * Float.BYTES)
                .toArray(ValueLayout.JAVA_FLOAT);
    }

    /** View by name, or null if absent (dtype rides on the view; kernels check at entry). */
    public static MemoryView<MemorySegment> viewOrNull(
            Map<String, MemoryView<MemorySegment>> views, String name) {
        return views.get(name);
    }

    /** First present view among alternate tensor names (GGUF converter naming drift), or null. */
    public static MemoryView<MemorySegment> firstPresent(
            Map<String, MemoryView<MemorySegment>> views, String... names) {
        for (String name : names) {
            MemoryView<MemorySegment> view = views.get(name);
            if (view != null) return view;
        }
        return null;
    }

    /** Per-layer array of views; a slot is null when its tensor is absent. */
    public static MemoryView<MemorySegment>[] viewArray(
            int n, java.util.function.IntFunction<MemoryView<MemorySegment>> get) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment>[] a = new MemoryView[n];
        for (int i = 0; i < n; i++) {
            a[i] = get.apply(i);
        }
        return a;
    }
}
