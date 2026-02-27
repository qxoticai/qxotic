package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

public final class ExecutionContexts {

    private ExecutionContexts() {}

    public static ExecutionContext defaultContext() {
        return forDevice(Environment.current().defaultDevice());
    }

    public static ExecutionContext forDevice(Device device) {
        Objects.requireNonNull(device, "device");
        MemoryDomain<?> memoryDomain = Environment.current().memoryDomainFor(device);
        return forContext(memoryDomain);
    }

    public static ExecutionContext forContext(MemoryDomain<?> memoryDomain) {
        Objects.requireNonNull(memoryDomain, "memoryDomain");
        return new DefaultExecutionContext(memoryDomain);
    }

    private static final class DefaultExecutionContext implements ExecutionContext {
        private final MemoryDomain<?> memoryDomain;
        private final ScratchAllocator scratch;

        private DefaultExecutionContext(MemoryDomain<?> memoryDomain) {
            this.memoryDomain = memoryDomain;
            this.scratch = new DefaultScratchAllocator(memoryDomain);
        }

        @Override
        public Device device() {
            return memoryDomain.device();
        }

        @Override
        public Tensor allocateOutput(Shape shape, DataType dtype) {
            Objects.requireNonNull(shape, "shape");
            Objects.requireNonNull(dtype, "dtype");
            Memory<?> memory = memoryDomain.memoryAllocator().allocateMemory(dtype, shape);
            MemoryView<?> view = MemoryView.of(memory, dtype, Layout.rowMajor(shape));
            return Tensor.of(view);
        }

        @Override
        public Tensor allocateOutput(Shape shape, DataType dtype, Layout layout) {
            Objects.requireNonNull(shape, "shape");
            Objects.requireNonNull(dtype, "dtype");
            Objects.requireNonNull(layout, "layout");
            if (!layout.shape().equals(shape)) {
                throw new IllegalArgumentException(
                        "Layout shape mismatch: expected " + shape + " but got " + layout.shape());
            }
            OutputBufferSpec spec = computeOutputSpec(layout, dtype);
            Memory<?> memory = memoryDomain.memoryAllocator().allocateMemory(spec.byteSize);
            MemoryView<?> view = MemoryView.of(memory, spec.byteOffset, dtype, layout);
            return Tensor.of(view);
        }

        @Override
        public ScratchAllocator scratch() {
            return scratch;
        }

        @Override
        public void barrier() {
            // Synchronous execution - nothing to barrier
        }

        private OutputBufferSpec computeOutputSpec(Layout layout, DataType dataType) {
            long[] shape = layout.shape().toArray();
            long[] strideBytes = layout.stride().scale(dataType.byteSize()).toArray();
            long minOffset = 0;
            long maxOffset = 0;
            for (int i = 0; i < shape.length; i++) {
                long dim = shape[i];
                if (dim <= 1) {
                    continue;
                }
                long span = (dim - 1) * strideBytes[i];
                if (strideBytes[i] >= 0) {
                    maxOffset += span;
                } else {
                    minOffset += span;
                }
            }
            long byteOffset = -minOffset;
            long byteSize = maxOffset - minOffset + dataType.byteSize();
            return new OutputBufferSpec(byteOffset, byteSize);
        }

        private record OutputBufferSpec(long byteOffset, long byteSize) {}
    }

    private static final class DefaultScratchAllocator implements ScratchAllocator {
        private final MemoryDomain<?> memoryDomain;

        private DefaultScratchAllocator(MemoryDomain<?> memoryDomain) {
            this.memoryDomain = memoryDomain;
        }

        @Override
        public Tensor allocate(Shape shape, DataType dtype) {
            Objects.requireNonNull(shape, "shape");
            Objects.requireNonNull(dtype, "dtype");
            Memory<?> memory = memoryDomain.memoryAllocator().allocateMemory(dtype, shape);
            MemoryView<?> view = MemoryView.of(memory, dtype, Layout.rowMajor(shape));
            return Tensor.of(view);
        }

        @Override
        public MemorySegment allocateBytes(long size) {
            Memory<?> memory = memoryDomain.memoryAllocator().allocateMemory(size);
            Object base = memory.base();
            if (base instanceof MemorySegment segment) {
                return segment;
            }
            throw new UnsupportedOperationException(
                    "Scratch byte allocation not supported for device " + memoryDomain.device());
        }
    }
}
