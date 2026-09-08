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
    /**
     * The one-line remedy for a JVM started without the incubating Vector API, which every kernel
     * needs: a library user otherwise meets a NoClassDefFoundError deep in the packer. Every loader
     * passes through here; a native image has the module compiled in.
     */
    public static void requireVectorApi() {
        if (System.getProperty("org.graalvm.nativeimage.imagecode") == null
                && ModuleLayer.boot().findModule("jdk.incubator.vector").isEmpty()) {
            throw new IllegalStateException(
                    "jinfer needs the Vector API: start the JVM with --add-modules"
                            + " jdk.incubator.vector (and --enable-native-access=ALL-UNNAMED)");
        }
    }

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
                fileChannel,
                checkedAdd(baseOffset, gguf.getTensorDataOffset(), "tensor data offset"),
                gguf.getTensors(),
                arena);
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
        requireVectorApi();
        if ((tensorDataOffset & (Float.BYTES - 1)) != 0)
            throw new IllegalArgumentException(
                    "GGUF tensor data offset must be 4-byte aligned, got " + tensorDataOffset);
        long fileSize = fileChannel.size();
        if (tensorDataOffset < 0 || tensorDataOffset > fileSize)
            throw new IllegalArgumentException(
                    "GGUF tensor data offset "
                            + tensorDataOffset
                            + " is outside a "
                            + fileSize
                            + "-byte file");
        long available = fileSize - tensorDataOffset;
        long required = requiredTensorBytes(tensors, available, "GGUF");
        MemorySegment tensorData =
                fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, required, arena);
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

    /** Rejects a parsed GGUF whose declared tensors do not fit within {@code byteSize}. */
    public static void requireComplete(GGUF gguf, long byteSize, String source) {
        long dataOffset = gguf.getTensorDataOffset();
        if (byteSize < 0 || dataOffset < 0 || dataOffset > byteSize)
            throw new IllegalArgumentException(
                    source
                            + " is truncated: tensor data starts at "
                            + dataOffset
                            + " of "
                            + byteSize);
        requiredTensorBytes(gguf.getTensors(), byteSize - dataOffset, source);
    }

    private static long requiredTensorBytes(
            Collection<TensorEntry> tensors, long available, String source) {
        long required = 0;
        for (TensorEntry tensor : tensors) {
            if (tensor.offset() < 0)
                throw new IllegalArgumentException(
                        source + " tensor " + tensor.name() + " has a negative offset");
            long end =
                    checkedAdd(
                            tensor.offset(),
                            tensor.ggmlType().byteSizeFor(Shape.flat(tensor.shape()).size()),
                            "tensor " + tensor.name() + " end");
            if (end > available) {
                throw new IllegalArgumentException(
                        source
                                + " is truncated, or holds metadata only: tensor "
                                + tensor.name()
                                + " needs bytes up to "
                                + end
                                + " of the tensor data, the file has "
                                + Math.max(available, 0)
                                + ". Re-download it, or convert the model again.");
            }
            required = Math.max(required, end);
        }
        return required;
    }

    private static long checkedAdd(long left, long right, String name) {
        try {
            return Math.addExact(left, right);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("GGUF " + name + " overflows", overflow);
        }
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
