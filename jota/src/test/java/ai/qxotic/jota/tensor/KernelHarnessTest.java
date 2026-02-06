package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Jota;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class KernelHarnessTest {

    @SuppressWarnings("unchecked")
    private static final MemoryDomain<MemorySegment> CONTEXT =
            (MemoryDomain<MemorySegment>) Environment.current().nativeBackend().memoryDomain();

    @Test
    void executesRegisteredKernelThroughDefaultDomain() {
        ExecutionContext ctx = Jota.defaultExecutionContext();
        KernelRegistry registry = Jota.kernelRegistry();
        registry.register("test_add_kernel", new AddKernel());

        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {10f, 20f, 30f, 40f}, Shape.of(2, 2));
        KernelInput input = Inputs.of(a, b);

        KernelExecutor executor = KernelExecutor.auto(registry);
        KernelOutput output = executor.execute("test_add_kernel", input, ctx);
        MemoryView<?> view = output.get(0).materialize();

        assertEquals(DataType.FP32, view.dataType());
        assertEquals(Shape.of(2, 2), view.shape());
        assertEquals(11f, readFloat(view, 0), 0.0001f);
        assertEquals(22f, readFloat(view, 1), 0.0001f);
        assertEquals(33f, readFloat(view, 2), 0.0001f);
        assertEquals(44f, readFloat(view, 3), 0.0001f);
    }

    @Test
    void executesKernelWithScalarParam() {
        ExecutionContext ctx = Jota.defaultExecutionContext();
        KernelRegistry registry = Jota.kernelRegistry();
        registry.register("test_scale_kernel", new ScaleKernel());

        Tensor a = Tensor.of(new float[] {1f, 2f, 3f, 4f}, Shape.of(2, 2));
        KernelInput input = Inputs.builder().tensor("a", a).param("scale", 3.0f).build();

        KernelExecutor executor = KernelExecutor.auto(registry);
        KernelOutput output = executor.execute("test_scale_kernel", input, ctx);
        MemoryView<?> view = output.get(0).materialize();

        assertEquals(DataType.FP32, view.dataType());
        assertEquals(Shape.of(2, 2), view.shape());
        assertEquals(3f, readFloat(view, 0), 0.0001f);
        assertEquals(6f, readFloat(view, 1), 0.0001f);
        assertEquals(9f, readFloat(view, 2), 0.0001f);
        assertEquals(12f, readFloat(view, 3), 0.0001f);
    }

    @Test
    void executesKernelWithNamedInputs() {
        ExecutionContext ctx = Jota.defaultExecutionContext();
        KernelRegistry registry = Jota.kernelRegistry();
        registry.register("test_named_add_kernel", new NamedAddKernel());

        Tensor a = Tensor.of(new float[] {2f, 4f, 6f, 8f}, Shape.of(2, 2));
        Tensor b = Tensor.of(new float[] {1f, 3f, 5f, 7f}, Shape.of(2, 2));
        KernelInput input = Inputs.builder().tensor("lhs", a).tensor("rhs", b).build();

        KernelExecutor executor = KernelExecutor.auto(registry);
        KernelOutput output = executor.execute("test_named_add_kernel", input, ctx);
        MemoryView<?> view = output.get(0).materialize();

        assertEquals(3f, readFloat(view, 0), 0.0001f);
        assertEquals(7f, readFloat(view, 1), 0.0001f);
        assertEquals(11f, readFloat(view, 2), 0.0001f);
        assertEquals(15f, readFloat(view, 3), 0.0001f);
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = CONTEXT.directAccess();
        return access.readFloat(typedView.memory(), offset);
    }

    private static final class AddKernel implements Kernel<KernelInput, KernelOutput> {
        @Override
        public KernelSignature signature() {
            return KernelSignature.builder("test_add_kernel")
                    .input("a", -1, null, null)
                    .input("b", -1, null, null)
                    .output("out", -1, null, null)
                    .supportedDevices(Device.PANAMA)
                    .build();
        }

        @Override
        public boolean supports(Device device) {
            return device.equals(Device.PANAMA);
        }

        @Override
        public KernelOutput execute(KernelInput input, ExecutionContext ctx) {
            Tensor a = input.get(0);
            Tensor b = input.get(1);
            return Outputs.of(a.add(b));
        }
    }

    private static final class ScaleKernel implements Kernel<KernelInput, KernelOutput> {
        @Override
        public KernelSignature signature() {
            return KernelSignature.builder("test_scale_kernel")
                    .input("a", -1, null, null)
                    .inputScalar("scale", DataType.FP32)
                    .output("out", -1, null, null)
                    .supportedDevices(Device.PANAMA)
                    .build();
        }

        @Override
        public boolean supports(Device device) {
            return device.equals(Device.PANAMA);
        }

        @Override
        public KernelOutput execute(KernelInput input, ExecutionContext ctx) {
            Tensor a = input.get(0);
            float scale = input.param("scale", Float.class);
            return Outputs.of(a.multiply(scale));
        }
    }

    private static final class NamedAddKernel implements Kernel<KernelInput, KernelOutput> {
        @Override
        public KernelSignature signature() {
            return KernelSignature.builder("test_named_add_kernel")
                    .input("lhs", -1, null, null)
                    .input("rhs", -1, null, null)
                    .output("out", -1, null, null)
                    .supportedDevices(Device.PANAMA)
                    .build();
        }

        @Override
        public boolean supports(Device device) {
            return device.equals(Device.PANAMA);
        }

        @Override
        public KernelOutput execute(KernelInput input, ExecutionContext ctx) {
            Tensor lhs = input.get("lhs");
            Tensor rhs = input.get("rhs");
            return Outputs.of(lhs.add(rhs));
        }
    }
}
