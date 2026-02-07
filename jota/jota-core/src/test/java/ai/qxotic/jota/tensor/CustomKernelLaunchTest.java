package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.panama.PanamaDeviceRuntime;
import ai.qxotic.jota.runtime.DeviceRuntime;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class CustomKernelLaunchTest {

    static PanamaDeviceRuntime runtime;

    @SuppressWarnings("unchecked")
    static MemoryDomain<MemorySegment> domain() {
        return runtime.memoryDomain();
    }

    @BeforeAll
    static void setUp() {
        runtime = new PanamaDeviceRuntime();
    }

    @Test
    void launchScaleKernel() {
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class ScaleKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> in = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(1);
                        float scale = args.getFloat(2);
                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        long n = in.shape().size();
                        for (long i = 0; i < n; i++) {
                            long inOff = in.byteOffset() + i * 4;
                            long outOff = out.byteOffset() + i * 4;
                            float v = access.readFloat(in.memory(), inOff);
                            access.writeFloat(out.memory(), outOff, v * scale);
                        }
                    }
                }
                """;

        runtime.registerKernel(
                "scale", KernelProgram.source(KernelProgram.Language.JAVA, source, "ScaleKernel"));

        MemoryView<?> in = allocate(DataType.FP32, Shape.flat(4));
        MemoryView<?> out = allocate(DataType.FP32, Shape.flat(4));
        writeFloats(in, 1f, 2f, 3f, 4f);

        runtime.launchKernel("scale", in, out, 2.0f);

        assertFloats(out, 2f, 4f, 6f, 8f);
    }

    @Test
    void launchAddKernel() {
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class AddKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> a = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> b = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(2);
                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        long n = a.shape().size();
                        for (long i = 0; i < n; i++) {
                            long aOff = a.byteOffset() + i * 4;
                            long bOff = b.byteOffset() + i * 4;
                            long oOff = out.byteOffset() + i * 4;
                            float va = access.readFloat(a.memory(), aOff);
                            float vb = access.readFloat(b.memory(), bOff);
                            access.writeFloat(out.memory(), oOff, va + vb);
                        }
                    }
                }
                """;

        runtime.registerKernel(
                "add", KernelProgram.source(KernelProgram.Language.JAVA, source, "AddKernel"));

        MemoryView<?> a = allocate(DataType.FP32, Shape.flat(4));
        MemoryView<?> b = allocate(DataType.FP32, Shape.flat(4));
        MemoryView<?> out = allocate(DataType.FP32, Shape.flat(4));
        writeFloats(a, 1f, 2f, 3f, 4f);
        writeFloats(b, 10f, 20f, 30f, 40f);

        runtime.launchKernel("add", a, b, out);

        assertFloats(out, 11f, 22f, 33f, 44f);
    }

    @Test
    void launchWithTensorArgs() {
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class NegateKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> in = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        long n = in.shape().size();
                        for (long i = 0; i < n; i++) {
                            long inOff = in.byteOffset() + i * 4;
                            long outOff = out.byteOffset() + i * 4;
                            float v = access.readFloat(in.memory(), inOff);
                            access.writeFloat(out.memory(), outOff, -v);
                        }
                    }
                }
                """;

        runtime.registerKernel(
                "negate",
                KernelProgram.source(KernelProgram.Language.JAVA, source, "NegateKernel"));

        Tensor input = Tensor.of(new float[] {1f, -2f, 3f, -4f});
        MemoryView<?> out = allocate(DataType.FP32, Shape.flat(4));

        // Pass a Tensor (not MemoryView) — should be materialized automatically
        runtime.launchKernel("negate", input, out);

        assertFloats(out, -1f, 2f, -3f, 4f);
    }

    @Test
    void launchWithExplicitLaunchConfig() {
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class FillKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(0);
                        float value = args.getFloat(1);
                        int count = args.getInt(2);
                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        for (int i = 0; i < count; i++) {
                            access.writeFloat(out.memory(), out.byteOffset() + (long) i * 4, value);
                        }
                    }
                }
                """;

        runtime.registerKernel(
                "fill", KernelProgram.source(KernelProgram.Language.JAVA, source, "FillKernel"));

        MemoryView<?> out = allocate(DataType.FP32, Shape.flat(4));

        // LaunchConfig is ignored by the Java backend, but the API should accept it
        runtime.launchKernel("fill", LaunchConfig.grid(1).block(256), out, 42.0f, 4);

        assertFloats(out, 42f, 42f, 42f, 42f);
    }

    @Test
    void launchWithHandleBasedApi() {
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class DoubleKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> in = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        long n = in.shape().size();
                        for (long i = 0; i < n; i++) {
                            long inOff = in.byteOffset() + i * 4;
                            long outOff = out.byteOffset() + i * 4;
                            float v = access.readFloat(in.memory(), inOff);
                            access.writeFloat(out.memory(), outOff, v * 2f);
                        }
                    }
                }
                """;

        // Use the handle returned by registerKernel directly
        KernelExecutable doubleKernel =
                runtime.registerKernel(
                        "double_it",
                        KernelProgram.source(KernelProgram.Language.JAVA, source, "DoubleKernel"));

        MemoryView<?> in = allocate(DataType.FP32, Shape.flat(3));
        MemoryView<?> out = allocate(DataType.FP32, Shape.flat(3));
        writeFloats(in, 5f, 10f, 15f);

        // Launch via handle + KernelArgs.fromVarargs (hot path)
        doubleKernel.launch(
                LaunchConfig.auto(),
                KernelArgs.fromVarargs(in, out),
                new ExecutionStream(runtime.device(), null, true));

        assertFloats(out, 10f, 20f, 30f);
    }

    @Test
    void launchGemvKernel() {
        // y[M] = A[M,N] * x[N]  — row-contiguous A, contiguous x
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class GemvKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> A = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> x = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> y = (MemoryView<MemorySegment>) args.getBuffer(2);
                        int M = args.getInt(3);
                        int N = args.getInt(4);

                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        MemorySegment aBuf = A.memory().base();
                        MemorySegment xBuf = x.memory().base();
                        MemorySegment yBuf = y.memory().base();
                        long aBase = A.byteOffset();
                        long xBase = x.byteOffset();
                        long yBase = y.byteOffset();

                        for (int row = 0; row < M; row++) {
                            float dot = 0f;
                            long rowOff = aBase + (long) row * N * Float.BYTES;
                            for (int col = 0; col < N; col++) {
                                float aVal = access.readFloat(A.memory(), rowOff + (long) col * Float.BYTES);
                                float xVal = access.readFloat(x.memory(), xBase + (long) col * Float.BYTES);
                                dot += aVal * xVal;
                            }
                            access.writeFloat(y.memory(), yBase + (long) row * Float.BYTES, dot);
                        }
                    }
                }
                """;

        runtime.registerKernel(
                "gemv", KernelProgram.source(KernelProgram.Language.JAVA, source, "GemvKernel"));

        int M = 3, N = 4;

        // A = [[1, 2, 3, 4],
        //      [5, 6, 7, 8],
        //      [9, 10, 11, 12]]
        MemoryView<?> A = allocate(DataType.FP32, Shape.of(M, N));
        writeFloats(A, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

        // x = [1, 0, 2, -1]
        MemoryView<?> x = allocate(DataType.FP32, Shape.flat(N));
        writeFloats(x, 1, 0, 2, -1);

        MemoryView<?> y = allocate(DataType.FP32, Shape.flat(M));

        // y = A * x = [1*1+2*0+3*2+4*(-1), 5*1+6*0+7*2+8*(-1), 9*1+10*0+11*2+12*(-1)]
        //           = [3, 11, 19]
        runtime.launchKernel("gemv", A, x, y, M, N);

        assertFloats(y, 3f, 11f, 19f);
    }

    @Test
    void launchWithAllTensorArgs() {
        // Both inputs are Tensors, output is a fresh Tensor (allocated via Tensor.zeros)
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class AddTensorsKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> a = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> b = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(2);
                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        long n = a.shape().size();
                        for (long i = 0; i < n; i++) {
                            long off = i * Float.BYTES;
                            float va = access.readFloat(a.memory(), a.byteOffset() + off);
                            float vb = access.readFloat(b.memory(), b.byteOffset() + off);
                            access.writeFloat(out.memory(), out.byteOffset() + off, va + vb);
                        }
                    }
                }
                """;

        runtime.registerKernel(
                "add_tensors",
                KernelProgram.source(KernelProgram.Language.JAVA, source, "AddTensorsKernel"));

        Tensor a = Tensor.of(new float[] {1f, 2f, 3f});
        Tensor b = Tensor.of(new float[] {10f, 20f, 30f});
        Tensor out = Tensor.of(new float[3]);

        // All three arguments are Tensors — materialized automatically
        runtime.launchKernel("add_tensors", a, b, out);

        assertFloats(out.materialize(), 11f, 22f, 33f);
    }

    @Test
    void launchWithTensorFactories() {
        // Use Tensor.ones and Tensor.full as inputs
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class MulTensorsKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> a = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> b = (MemoryView<MemorySegment>) args.getBuffer(1);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(2);
                        MemoryAccess<MemorySegment> access = domain.directAccess();
                        long n = a.shape().size();
                        for (long i = 0; i < n; i++) {
                            long off = i * Float.BYTES;
                            float va = access.readFloat(a.memory(), a.byteOffset() + off);
                            float vb = access.readFloat(b.memory(), b.byteOffset() + off);
                            access.writeFloat(out.memory(), out.byteOffset() + off, va * vb);
                        }
                    }
                }
                """;

        runtime.registerKernel(
                "mul_tensors",
                KernelProgram.source(KernelProgram.Language.JAVA, source, "MulTensorsKernel"));

        Environment eagerEnv =
                new Environment(
                        Device.PANAMA,
                        Environment.current().defaultFloat(),
                        Environment.current().runtimes(),
                        ExecutionMode.EAGER);
        Environment.with(
                eagerEnv,
                () -> {
                    // Broadcast tensors must be made contiguous before passing to kernels
                    // that index linearly — this is the caller's responsibility (like PyTorch)
                    Tensor ones = Tensor.ones(DataType.FP32, Shape.flat(4)).contiguous();
                    Tensor fives = Tensor.full(5.0f, Shape.flat(4)).contiguous();
                    Tensor out = Tensor.of(new float[4]);

                    runtime.launchKernel("mul_tensors", ones, fives, out);

                    assertFloats(out.materialize(), 5f, 5f, 5f, 5f);
                    return null;
                });
    }

    @Test
    void tensorVarargMustBeContiguous() {
        Tensor broadcast = Tensor.full(5.0f, Shape.flat(4));
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> KernelArgs.fromVarargs(broadcast));
        assertTrue(ex.getMessage().contains("row-major contiguous"));
    }

    @Test
    void launchGemvWithTensorInputs() {
        // GEMV using Tensor inputs and MemoryView output
        runtime.registerKernel(
                "gemv_t",
                KernelProgram.source(
                        KernelProgram.Language.JAVA, GEMV_KERNEL_SOURCE, "GemvTensorKernel"));

        int M = 2, N = 3;
        Tensor A = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, Shape.of(M, N));
        Tensor x = Tensor.of(new float[] {1, -1, 2});
        MemoryView<?> y = allocate(DataType.FP32, Shape.flat(M));

        // y = A * x = [1-2+6, 4-5+12] = [5, 11]
        runtime.launchKernel("gemv_t", A, x, y, M, N);

        assertFloats(y, 5f, 11f);
    }

    // Shared GEMV kernel source used by tensor-input tests
    // language=java
    private static final String GEMV_KERNEL_SOURCE =
            """
            package ai.qxotic.jota.tensor.jit;

            import ai.qxotic.jota.memory.MemoryAccess;
            import ai.qxotic.jota.memory.MemoryDomain;
            import ai.qxotic.jota.memory.MemoryView;
            import ai.qxotic.jota.tensor.JavaKernel;
            import ai.qxotic.jota.tensor.KernelArgs;
            import java.lang.foreign.MemorySegment;

            public final class GemvTensorKernel implements JavaKernel {
                @Override
                @SuppressWarnings("unchecked")
                public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                    MemoryView<MemorySegment> A = (MemoryView<MemorySegment>) args.getBuffer(0);
                    MemoryView<MemorySegment> x = (MemoryView<MemorySegment>) args.getBuffer(1);
                    MemoryView<MemorySegment> y = (MemoryView<MemorySegment>) args.getBuffer(2);
                    int M = args.getInt(3);
                    int N = args.getInt(4);
                    MemoryAccess<MemorySegment> access = domain.directAccess();
                    for (int row = 0; row < M; row++) {
                        float dot = 0f;
                        long rowOff = A.byteOffset() + (long) row * N * Float.BYTES;
                        for (int col = 0; col < N; col++) {
                            float aVal = access.readFloat(A.memory(), rowOff + (long) col * Float.BYTES);
                            float xVal = access.readFloat(x.memory(), x.byteOffset() + (long) col * Float.BYTES);
                            dot += aVal * xVal;
                        }
                        access.writeFloat(y.memory(), y.byteOffset() + (long) row * Float.BYTES, dot);
                    }
                }
            }
            """;

    @Test
    void tooFewArgsThrowsAtKernelLevel() {
        // The scale kernel expects: buffer, buffer, scalar
        // Pass only one buffer — the kernel will fail when accessing missing args
        // language=java
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryAccess;
                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class MismatchKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        MemoryView<MemorySegment> in = (MemoryView<MemorySegment>) args.getBuffer(0);
                        MemoryView<MemorySegment> out = (MemoryView<MemorySegment>) args.getBuffer(1);
                        float scale = args.getFloat(2);
                    }
                }
                """;

        runtime.registerKernel(
                "mismatch",
                KernelProgram.source(KernelProgram.Language.JAVA, source, "MismatchKernel"));

        MemoryView<?> buf = allocate(DataType.FP32, Shape.flat(4));

        // Only one buffer, no scalar — kernel tries to read index 1 and 2
        assertThrows(Exception.class, () -> runtime.launchKernel("mismatch", buf));
    }

    @Test
    void wrongArgTypeThrows() {
        // Passing an unsupported arg type (e.g. a plain String)
        assertThrows(
                IllegalArgumentException.class,
                () -> KernelArgs.fromVarargs("not a valid kernel arg"));
    }

    @Test
    void scalarReadOnBufferEntryThrows() {
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class BadScalarReadKernel implements JavaKernel {
                    @Override
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        // index 0 is a buffer, but we try to read it as a scalar
                        float v = args.getFloat(0);
                    }
                }
                """;

        runtime.registerKernel(
                "bad_scalar_read",
                KernelProgram.source(
                        KernelProgram.Language.JAVA, source, "BadScalarReadKernel"));

        MemoryView<?> buf = allocate(DataType.FP32, Shape.flat(4));
        assertThrows(Exception.class, () -> runtime.launchKernel("bad_scalar_read", buf));
    }

    @Test
    void bufferReadOnScalarEntryThrows() {
        // language=java
        String source =
                """
                package ai.qxotic.jota.tensor.jit;

                import ai.qxotic.jota.memory.MemoryDomain;
                import ai.qxotic.jota.memory.MemoryView;
                import ai.qxotic.jota.tensor.JavaKernel;
                import ai.qxotic.jota.tensor.KernelArgs;
                import java.lang.foreign.MemorySegment;

                public final class BadBufferReadKernel implements JavaKernel {
                    @Override
                    @SuppressWarnings("unchecked")
                    public void execute(MemoryDomain<MemorySegment> domain, KernelArgs args) {
                        // index 0 is a scalar, but we try to read it as a buffer
                        MemoryView<MemorySegment> v =
                                (MemoryView<MemorySegment>) args.getBuffer(0);
                    }
                }
                """;

        runtime.registerKernel(
                "bad_buffer_read",
                KernelProgram.source(
                        KernelProgram.Language.JAVA, source, "BadBufferReadKernel"));

        assertThrows(Exception.class, () -> runtime.launchKernel("bad_buffer_read", 42));
    }

    @Test
    void nullArgThrows() {
        assertThrows(Exception.class, () -> KernelArgs.fromVarargs((Object) null));
    }

    @Test
    void unregisteredKernelThrows() {
        assertThrows(
                IllegalArgumentException.class, () -> runtime.launchKernel("no_such_kernel", 1));
    }

    // ── helpers ──────────────────────────────────────────────────

    private MemoryView<?> allocate(DataType dtype, Shape shape) {
        var mem = domain().memoryAllocator().allocateMemory(dtype, shape.size());
        return MemoryView.of(mem, dtype, ai.qxotic.jota.Layout.rowMajor(shape));
    }

    @SuppressWarnings("unchecked")
    private void writeFloats(MemoryView<?> view, float... values) {
        MemoryView<MemorySegment> v = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = domain().directAccess();
        for (int i = 0; i < values.length; i++) {
            access.writeFloat(v.memory(), v.byteOffset() + (long) i * Float.BYTES, values[i]);
        }
    }

    @SuppressWarnings("unchecked")
    private void assertFloats(MemoryView<?> view, float... expected) {
        MemoryView<MemorySegment> v = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = domain().directAccess();
        for (int i = 0; i < expected.length; i++) {
            float actual =
                    access.readFloat(v.memory(), v.byteOffset() + (long) i * Float.BYTES);
            assertEquals(expected[i], actual, 0.0001f, "mismatch at index " + i);
        }
    }
}
