package com.qxotic.jota.memory;

import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class StridedCopy {

    private static final MemoryAccess<MemorySegment> HOST_ACCESS =
            NativeMemoryFactory.memoryAccess();
    private static final MemoryOperations<MemorySegment> HOST_OPS =
            NativeMemoryFactory.memoryOperations();

    private StridedCopy() {}

    static <B> void copy(MemoryDomain<B> domain, MemoryView<B> src, MemoryView<B> dst) {
        if (src.dataType() != dst.dataType()) {
            throw new IllegalArgumentException(
                    "Data type mismatch: " + src.dataType() + " vs " + dst.dataType());
        }
        if (!src.shape().equals(dst.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + src.shape() + " vs " + dst.shape());
        }
        if (!src.memory().device().equals(dst.memory().device())) {
            throw new IllegalArgumentException("Source and destination devices must match");
        }
        long bytes = src.dataType().byteSizeFor(src.shape());
        if (bytes == 0) {
            return;
        }
        MemoryOperations<B> ops = domain.memoryOperations();
        if (src.isRowMajorContiguous() && dst.isRowMajorContiguous()) {
            ops.copy(src.memory(), src.byteOffset(), dst.memory(), dst.byteOffset(), bytes);
            return;
        }
        MemoryAccess<B> access = domain.directAccess();
        if (access == null) {
            copyViaHost(domain, src, dst);
            return;
        }
        if (src.memory() == dst.memory()) {
            // ponytail: a self-copy goes through a contiguous temp (two passes). Only views over
            // the same Memory are treated as aliased; two Memory objects over the same bytes are
            // the caller's problem.
            MemoryView<B> tmp =
                    MemoryView.rowMajor(
                            domain.memoryAllocator().allocateMemory(bytes),
                            src.dataType(),
                            src.shape());
            copyElements(access, ops, src, tmp);
            copyElements(access, ops, tmp, dst);
            return;
        }
        copyElements(access, ops, src, dst);
    }

    /**
     * Odometer over the flat dims with running byte offsets. Scalar widths go through typed access;
     * any other width (the block-quantized types, 17..210 bytes) is one bulk copy per element, so
     * no dtype is ever named here.
     */
    private static <B> void copyElements(
            MemoryAccess<B> a, MemoryOperations<B> ops, MemoryView<B> src, MemoryView<B> dst) {
        long[] dims = src.shape().toArray();
        long[] ss = src.byteStride().toArray();
        long[] ds = dst.byteStride().toArray();
        int elem = Math.toIntExact(src.dataType().byteSize());
        int rank = dims.length;
        long[] idx = new long[rank];
        long so = src.byteOffset();
        long doff = dst.byteOffset();
        Memory<B> s = src.memory();
        Memory<B> d = dst.memory();
        while (true) {
            switch (elem) {
                case 1 -> a.writeByte(d, doff, a.readByte(s, so));
                case 2 -> a.writeShort(d, doff, a.readShort(s, so));
                case 4 -> a.writeInt(d, doff, a.readInt(s, so));
                case 8 -> a.writeLong(d, doff, a.readLong(s, so));
                default -> ops.copy(s, so, d, doff, elem);
            }
            int ax = rank - 1;
            for (; ax >= 0; ax--) {
                so += ss[ax];
                doff += ds[ax];
                if (++idx[ax] < dims[ax]) {
                    break;
                }
                so -= ss[ax] * dims[ax];
                doff -= ds[ax] * dims[ax];
                idx[ax] = 0;
            }
            if (ax < 0) {
                return;
            }
        }
    }

    /** Opaque domain (no direct access): mirror only each view's byte span on the host. */
    private static <B> void copyViaHost(
            MemoryDomain<B> domain, MemoryView<B> src, MemoryView<B> dst) {
        MemoryOperations<B> ops = domain.memoryOperations();
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> hostSrc = stage(arena, ops, src, true);
            // A non-contiguous dst has gaps that must survive: bring its span down, write, push
            // back.
            boolean readBack = !dst.isRowMajorContiguous();
            MemoryView<MemorySegment> hostDst = stage(arena, ops, dst, readBack);
            copyElements(HOST_ACCESS, HOST_OPS, hostSrc, hostDst);
            long[] span = byteSpan(dst);
            ops.copyFromNative(hostDst.memory(), 0, dst.memory(), span[0], span[1] - span[0]);
        }
    }

    private static <B> MemoryView<MemorySegment> stage(
            Arena arena, MemoryOperations<B> ops, MemoryView<B> view, boolean download) {
        long[] span = byteSpan(view);
        long bytes = span[1] - span[0];
        Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(arena.allocate(bytes));
        if (download) {
            ops.copyToNative(view.memory(), span[0], mem, 0, bytes);
        }
        return MemoryView.of(mem, view.byteOffset() - span[0], view.dataType(), view.layout());
    }

    /**
     * The [min, max) byte range a view touches: the same walk as {@link MemoryView#isWithinBounds}.
     */
    static long[] byteSpan(MemoryView<?> view) {
        long min = view.byteOffset();
        long max = view.byteOffset();
        long[] dims = view.shape().toArray();
        long[] strides = view.byteStride().toArray();
        for (int i = 0; i < dims.length; i++) {
            long span = Math.multiplyExact(dims[i] - 1, strides[i]);
            if (span >= 0) {
                max = Math.addExact(max, span);
            } else {
                min = Math.addExact(min, span);
            }
        }
        return new long[] {min, Math.addExact(max, view.dataType().byteSize())};
    }
}
