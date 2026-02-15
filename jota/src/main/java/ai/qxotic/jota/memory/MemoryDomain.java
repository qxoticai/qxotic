package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import java.lang.foreign.MemorySegment;

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
            MemoryDomain<S> srcContext,
            MemoryView<S> src,
            MemoryDomain<D> dstContext,
            MemoryView<D> dst) {
        if (src.dataType() != dst.dataType()) {
            throw new IllegalArgumentException(
                    "Data type mismatch: " + src.dataType() + " vs " + dst.dataType());
        }
        if (!src.shape().equals(dst.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + src.shape() + " vs " + dst.shape());
        }

        if (srcContext.device().equals(dstContext.device())) {
            copySameDevice(srcContext, src, dst);
            return;
        }

        if (src.isRowMajorContiguous() && dst.isRowMajorContiguous()) {
            copyContiguous(srcContext, src, dstContext, dst);
            return;
        }

        MemoryDomain<MemorySegment> nativeContext = nativeContext();
        MemoryView<S> srcContig = contiguousCopy(srcContext, src);
        MemoryView<MemorySegment> nativeContig =
                allocateContiguous(nativeContext, src.dataType(), src.shape());
        copyContiguous(srcContext, srcContig, nativeContext, nativeContig);

        MemoryView<D> dstContig = allocateContiguous(dstContext, dst.dataType(), dst.shape());
        copyContiguous(nativeContext, nativeContig, dstContext, dstContig);
        copySameDevice(dstContext, dstContig, dst);
    }

    private static <S, D> void copySameDevice(
            MemoryDomain<S> domain, MemoryView<S> src, MemoryView<D> dst) {
        if (srcContextDevice(domain).equals(dst.memory().device())) {
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
            MemoryDomain<S> srcContext,
            MemoryView<S> src,
            MemoryDomain<D> dstContext,
            MemoryView<D> dst) {
        long bytes = src.shape().size() * src.dataType().byteSize();
        if (bytes == 0) {
            return;
        }
        MemoryOperations.copy(
                srcContext.memoryOperations(),
                src.memory(),
                src.byteOffset(),
                dstContext.memoryOperations(),
                dst.memory(),
                dst.byteOffset(),
                bytes,
                nativeContext().memoryAllocator().allocateMemory(bytes));
    }

    @SuppressWarnings("unchecked")
    private static MemoryDomain<MemorySegment> nativeContext() {
        return (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();
    }

    private static <B> Device srcContextDevice(MemoryDomain<B> domain) {
        return domain.device();
    }

    @Override
    void close();

    @Override
    String toString();
}
