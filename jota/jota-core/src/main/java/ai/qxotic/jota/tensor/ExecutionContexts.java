package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

public final class ExecutionContexts {

    private static final KernelRegistry GLOBAL_REGISTRY = new DefaultKernelRegistry();

    private ExecutionContexts() {}

    public static ExecutionContext defaultContext() {
        return forDevice(Environment.current().defaultDevice());
    }

    public static KernelRegistry globalRegistry() {
        return GLOBAL_REGISTRY;
    }

    public static ExecutionContext forDevice(Device device) {
        return forDevice(device, GLOBAL_REGISTRY);
    }

    public static ExecutionContext forDevice(Device device, KernelRegistry registry) {
        Objects.requireNonNull(device, "device");
        MemoryDomain<?> memoryDomain = Environment.current().runtimeFor(device).memoryDomain();
        return forContext(memoryDomain, registry);
    }

    public static ExecutionContext forContext(
            MemoryDomain<?> memoryDomain, KernelRegistry registry) {
        Objects.requireNonNull(memoryDomain, "memoryDomain");
        Objects.requireNonNull(registry, "registry");
        return new DefaultExecutionContext(memoryDomain, registry);
    }

    private static final class DefaultExecutionContext implements ExecutionContext {
        private final MemoryDomain<?> memoryDomain;
        private final KernelRegistry registry;
        private final ScratchAllocator scratch;
        private final ComputeStream stream;

        private DefaultExecutionContext(MemoryDomain<?> memoryDomain, KernelRegistry registry) {
            this.memoryDomain = memoryDomain;
            this.registry = registry;
            this.scratch = new DefaultScratchAllocator(memoryDomain);
            this.stream = new ImmediateComputeStream();
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
        public ComputeStream stream() {
            return stream;
        }

        @Override
        public void barrier() {
            ComputeStream.Event event = stream.record();
            stream.waitFor(event);
        }

        @Override
        public KernelRegistry kernels() {
            return registry;
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

    private static final class ImmediateComputeStream implements ComputeStream {
        @Override
        public void enqueue(Runnable task) {
            Objects.requireNonNull(task, "task");
            task.run();
        }

        @Override
        public void waitFor(Event event) {}

        @Override
        public Event record() {
            return ImmediateEvent.INSTANCE;
        }

        @Override
        public boolean isComplete() {
            return true;
        }

        private enum ImmediateEvent implements Event {
            INSTANCE
        }
    }
}
