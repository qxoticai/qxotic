package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;

/**
 * A memory backend: its allocator, element access ({@code null} when the host cannot address the
 * memory, e.g. a GPU) and bulk operations. Closing a domain closes its allocator; whether that
 * releases memory is the allocator's ownership (see {@link MemoryAllocators}). The shared array
 * domains own nothing and their {@code close()} is a no-op.
 */
public interface MemoryDomain<B> extends AutoCloseable {
    Device device();

    MemoryAllocator<B> memoryAllocator();

    /** Optional capability, can be null for opaque memory implementations e.g. GPUs. */
    MemoryAccess<B> directAccess();

    MemoryOperations<B> memoryOperations();

    /**
     * Returns the memory allocation granularity in bytes. Delegates to the underlying memory
     * allocator.
     *
     * @return the allocation granularity in bytes
     * @see MemoryAllocator#memoryGranularity()
     */
    default long memoryGranularity() {
        return memoryAllocator().memoryGranularity();
    }

    /**
     * Checks if this domain can allocate memory for the given DataType. Delegates to the underlying
     * memory allocator.
     *
     * @param dataType the data type to check
     * @return true if this domain can allocate the given DataType
     * @see MemoryAllocator#supportsDataType(DataType)
     */
    default boolean supportsDataType(DataType dataType) {
        return memoryAllocator().supportsDataType(dataType);
    }

    default void copy(MemoryView<B> src, MemoryView<B> dst) {
        StridedCopy.copy(this, src, dst);
    }

    static <S, D> void copy(
            MemoryDomain<S> srcDomain,
            MemoryView<S> src,
            MemoryDomain<D> dstDomain,
            MemoryView<D> dst) {
        if (src.dataType() != dst.dataType()) {
            throw new IllegalArgumentException(
                    "Data type mismatch: " + src.dataType() + " vs " + dst.dataType());
        }
        if (!src.shape().equals(dst.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + src.shape() + " vs " + dst.shape());
        }

        if (srcDomain.device().equals(dstDomain.device())
                && sharesMemoryOperations(srcDomain, dstDomain)) {
            copySameDevice(srcDomain, src, dst);
            return;
        }

        if (src.isRowMajorContiguous() && dst.isRowMajorContiguous()) {
            copyContiguous(srcDomain, src, dstDomain, dst);
            return;
        }

        MemoryView<S> srcContig = contiguousCopy(srcDomain, src);
        MemoryView<D> dstContig = allocateContiguous(dstDomain, dst.dataType(), dst.shape());
        copyContiguous(srcDomain, srcContig, dstDomain, dstContig);
        copySameDevice(dstDomain, dstContig, dst);
    }

    private static boolean sharesMemoryOperations(MemoryDomain<?> left, MemoryDomain<?> right) {
        return left.memoryOperations() == right.memoryOperations();
    }

    private static <S, D> void copySameDevice(
            MemoryDomain<S> domain, MemoryView<S> src, MemoryView<D> dst) {
        if (domain.device().equals(dst.memory().device())) {
            @SuppressWarnings("unchecked")
            MemoryView<S> castDst = (MemoryView<S>) dst;
            domain.copy(src, castDst);
            return;
        }
        throw new IllegalArgumentException("Source and destination devices must match");
    }

    private static <B> MemoryView<B> contiguousCopy(MemoryDomain<B> domain, MemoryView<B> src) {
        MemoryView<B> dst = allocateContiguous(domain, src.dataType(), src.shape());
        StridedCopy.copy(domain, src, dst);
        return dst;
    }

    private static <B> MemoryView<B> allocateContiguous(
            MemoryDomain<B> domain, DataType dataType, Shape shape) {
        return MemoryView.of(
                domain.memoryAllocator().allocateMemory(dataType, shape),
                dataType,
                Layout.rowMajor(shape));
    }

    private static <S, D> void copyContiguous(
            MemoryDomain<S> srcDomain,
            MemoryView<S> src,
            MemoryDomain<D> dstDomain,
            MemoryView<D> dst) {
        long bytes = src.dataType().byteSizeFor(src.shape());
        if (bytes == 0) {
            return;
        }
        MemoryOperations.copy(
                srcDomain.memoryOperations(),
                src.memory(),
                src.byteOffset(),
                dstDomain.memoryOperations(),
                dst.memory(),
                dst.byteOffset(),
                bytes);
    }

    /** Redeclared without {@code throws Exception}: closing a domain never throws a checked one. */
    @Override
    void close();
}
