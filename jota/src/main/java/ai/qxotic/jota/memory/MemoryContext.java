package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import java.lang.foreign.MemorySegment;

public interface MemoryContext<B> extends AutoCloseable {
    Device device();

    MemoryAllocator<B> memoryAllocator();

    /** Optional capability, can be null for opaque memory implementations e.g. GPUs. */
    MemoryAccess<B> memoryAccess();

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
     * Checks if this context can allocate memory for the given DataType. Delegates to the
     * underlying memory allocator.
     *
     * @param dataType the data type to check
     * @return true if this context can allocate the given DataType
     * @see MemoryAllocator#supportsDataType(DataType)
     */
    default boolean supportsDataType(DataType dataType) {
        return memoryAllocator().supportsDataType(dataType);
    }

    default void copy(MemoryView<B> src, MemoryView<B> dst) {
        StridedCopy.copy(this, src, dst);
    }

    static <S, D> void copy(
            MemoryContext<S> srcContext,
            MemoryView<S> src,
            MemoryContext<D> dstContext,
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

        if (src.isContiguous() && dst.isContiguous()) {
            copyContiguous(srcContext, src, dstContext, dst);
            return;
        }

        MemoryContext<MemorySegment> nativeContext = nativeContext();
        MemoryView<S> srcContig = contiguousCopy(srcContext, src);
        MemoryView<MemorySegment> nativeContig =
                allocateContiguous(nativeContext, src.dataType(), src.shape().size());
        copyContiguous(srcContext, srcContig, nativeContext, nativeContig);

        MemoryView<D> dstContig =
                allocateContiguous(dstContext, dst.dataType(), dst.shape().size());
        copyContiguous(nativeContext, nativeContig, dstContext, dstContig);
        copySameDevice(dstContext, dstContig, dst);
    }

    private static <S, D> void copySameDevice(
            MemoryContext<S> context, MemoryView<S> src, MemoryView<D> dst) {
        if (srcContextDevice(context).equals(dst.memory().device())) {
            @SuppressWarnings("unchecked")
            MemoryView<S> castDst = (MemoryView<S>) dst;
            StridedCopy.copy(context, src, castDst);
            return;
        }
        throw new IllegalArgumentException("Source and destination devices must match");
    }

    private static <B> MemoryView<B> contiguousCopy(MemoryContext<B> context, MemoryView<B> src) {
        MemoryView<B> dst = allocateContiguous(context, src.dataType(), src.shape().size());
        StridedCopy.copy(context, src, dst);
        return dst;
    }

    private static <B> MemoryView<B> allocateContiguous(
            MemoryContext<B> context, DataType dataType, long elementCount) {
        return MemoryView.of(
                context.memoryAllocator().allocateMemory(dataType, elementCount),
                dataType,
                Layout.rowMajor(elementCount));
    }

    private static <S, D> void copyContiguous(
            MemoryContext<S> srcContext,
            MemoryView<S> src,
            MemoryContext<D> dstContext,
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
    private static MemoryContext<MemorySegment> nativeContext() {
        return (MemoryContext<MemorySegment>)
                Environment.current().registry().context(Device.NATIVE);
    }

    private static <B> Device srcContextDevice(MemoryContext<B> context) {
        return context.device();
    }

    @Override
    void close();

    String toString();
}
