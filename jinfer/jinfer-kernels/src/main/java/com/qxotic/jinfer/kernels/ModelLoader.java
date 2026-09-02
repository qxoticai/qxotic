package com.qxotic.jinfer.kernels;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memories;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;
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
import java.util.Optional;

/**
 * Shared GGUF loading support: parses metadata, memory-maps tensor data read-only, and exposes each
 * tensor as a {@code MemoryView<MemorySegment>}. Unsupported data types fail during loading.
 */
public final class ModelLoader {

    private ModelLoader() {}

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
     * +} the GGUF tensor data offset - used for self-archives where the GGUF blob is embedded at a
     * non-zero position in the file.
     */
    public static Map<String, MemoryView<MemorySegment>> loadTensors(
            FileChannel fileChannel, GGUF gguf, long baseOffset, Arena arena) throws IOException {
        return loadTensors(
                fileChannel, baseOffset + gguf.getTensorDataOffset(), gguf.getTensors(), arena);
    }

    /**
     * Maps the tensors described by {@code tensors}, whose data starts at {@code tensorDataOffset}
     * in the channel. The mappings remain valid for the lifetime of {@code arena}.
     */
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
        Memory<MemorySegment> memory = Memories.of(tensorData);
        Map<String, MemoryView<MemorySegment>> tensorViews = HashMap.newHashMap(tensors.size());
        for (TensorEntry tensor : tensors) {
            DataType dtype =
                    GGMLDataTypes.toDataType(
                            tensor.ggmlType()); // scope guard: throws on unsupported
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
        // Load-time weight packing (jam's in-memory layouts; -Djinfer.pack=false to disable):
        // packed tensors move into a page-aligned slab in the SAME arena and their canonical mmap
        // pages are dropped - one copy total, shared as-is with Metal via unified memory.
        return JamPack.apply(tensorViews, arena);
    }

    /**
     * The optional "llama3" RoPE frequency-scaling factors ({@code rope_freqs.weight}), empty if
     * the model uses plain RoPE. These are per-frequency divisors (1.0 for high frequencies, up to
     * the long-context factor for low frequencies); see {@code RoPE.withFreqFactors}.
     */
    public static Optional<float[]> ropeFreqFactors(
            Map<String, MemoryView<MemorySegment>> tensorViews) {
        return findF32(tensorViews, "rope_freqs.weight")
                .map(
                        e ->
                                e.memory()
                                        .base()
                                        .asSlice(e.byteOffset(), e.logicalSize() * Float.BYTES)
                                        .toArray(ValueLayout.JAVA_FLOAT));
    }

    /** View by name; throws {@link IllegalArgumentException} if absent. */
    public static MemoryView<MemorySegment> require(
            Map<String, MemoryView<MemorySegment>> views, String name) {
        MemoryView<MemorySegment> view = views.get(name);
        if (view == null) throw new IllegalArgumentException("missing tensor: " + name);
        return view;
    }

    /** FP32 view by name (dtype checked AT LOAD), or throw if absent. */
    public static MemoryView<MemorySegment> requireF32(
            Map<String, MemoryView<MemorySegment>> views, String name) {
        MemoryView<MemorySegment> view = require(views, name);
        Views.requireDatatype(view, DataType.FP32, name);
        return view;
    }

    /** View by name if present - any dtype (it rides on the view; kernels check at entry). */
    public static Optional<MemoryView<MemorySegment>> find(
            Map<String, MemoryView<MemorySegment>> views, String name) {
        return Optional.ofNullable(views.get(name));
    }

    /** FP32 view by name if present; when present the dtype is checked AT LOAD. */
    public static Optional<MemoryView<MemorySegment>> findF32(
            Map<String, MemoryView<MemorySegment>> views, String name) {
        MemoryView<MemorySegment> view = views.get(name);
        if (view != null) Views.requireDatatype(view, DataType.FP32, name);
        return Optional.ofNullable(view);
    }

    /** First present view among alternate tensor names (GGUF converter naming drift). */
    public static Optional<MemoryView<MemorySegment>> findFirst(
            Map<String, MemoryView<MemorySegment>> views, String... names) {
        for (String name : names) {
            MemoryView<MemorySegment> view = views.get(name);
            if (view != null) return Optional.of(view);
        }
        return Optional.empty();
    }
}
