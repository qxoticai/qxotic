package ai.qxotic.jota.runtime.javaaot;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class JavaAotReductionKernelTest {

    private Environment javaAotEnv() {
        return new Environment(
                Device.JAVA_AOT,
                DataType.FP32,
                Environment.current().runtimes(),
                ExecutionMode.LAZY);
    }

    @Test
    void reductionSum() {
        assertTrue(Environment.current().runtimes().hasRuntime(Device.JAVA_AOT));
        Environment env = javaAotEnv();

        MemoryView<?> output =
                Environment.with(
                        env,
                        () -> {
                            Tensor input = Tensor.iota(24, DataType.FP32).view(Shape.of(2, 3, 4));
                            return Tracer.trace(input, t -> t.sum(DataType.FP32, 2)).materialize();
                        });

        assertEquals(Shape.of(2, 3), output.shape());
        assertClose(readFp32(output, 0), 6.0f);
        assertClose(readFp32(output, 1), 22.0f);
        assertClose(readFp32(output, 2), 38.0f);
        assertClose(readFp32(output, 3), 54.0f);
        assertClose(readFp32(output, 4), 70.0f);
        assertClose(readFp32(output, 5), 86.0f);
    }

    @Test
    void reductionMax() {
        Environment env = javaAotEnv();

        MemoryView<?> output =
                Environment.with(
                        env,
                        () -> {
                            Tensor input = Tensor.iota(24, DataType.FP32).view(Shape.of(2, 3, 4));
                            return Tracer.trace(input, t -> t.max(true, 2)).materialize();
                        });

        assertEquals(Shape.of(2, 3, 1), output.shape());
        assertClose(readFp32(output, 0), 3.0f);
        assertClose(readFp32(output, 1), 7.0f);
        assertClose(readFp32(output, 2), 11.0f);
        assertClose(readFp32(output, 3), 15.0f);
        assertClose(readFp32(output, 4), 19.0f);
        assertClose(readFp32(output, 5), 23.0f);
    }

    @Test
    void reductionProd() {
        Environment env = javaAotEnv();

        MemoryView<?> output =
                Environment.with(
                        env,
                        () -> {
                            Tensor input =
                                    Tensor.iota(24, DataType.FP32).add(1f).view(Shape.of(2, 3, 4));
                            return Tracer.trace(input, t -> t.product(DataType.FP32, 2))
                                    .materialize();
                        });

        assertEquals(Shape.of(2, 3), output.shape());
        assertClose(readFp32(output, 0), 24.0f);
        assertClose(readFp32(output, 1), 1680.0f);
        assertClose(readFp32(output, 2), 11880.0f);
        assertClose(readFp32(output, 3), 43680.0f);
        assertClose(readFp32(output, 4), 116280.0f);
        assertClose(readFp32(output, 5), 255024.0f);
    }

    @Test
    void reductionMin() {
        Environment env = javaAotEnv();

        MemoryView<?> output =
                Environment.with(
                        env,
                        () -> {
                            Tensor input = Tensor.iota(24, DataType.FP32).view(Shape.of(2, 3, 4));
                            return Tracer.trace(input, t -> t.min(0)).materialize();
                        });

        assertEquals(Shape.of(3, 4), output.shape());
        for (int i = 0; i < 12; i++) {
            assertClose(readFp32(output, i), i);
        }
    }

    @Test
    void keepsDimsOnNonSuffixReduction() {
        Environment env = javaAotEnv();

        MemoryView<?> output =
                Environment.with(
                        env,
                        () -> {
                            Tensor input = Tensor.iota(24, DataType.FP32).view(Shape.of(2, 3, 4));
                            return Tracer.trace(input, t -> t.max(true, 0)).materialize();
                        });

        assertEquals(Shape.of(1, 3, 4), output.shape());
        for (int i = 0; i < 12; i++) {
            assertClose(readFp32(output, i), i + 12);
        }
    }

    @SuppressWarnings("unchecked")
    private float readFp32(MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access =
                (MemoryAccess<MemorySegment>)
                        Environment.current()
                                .runtimeFor(Device.JAVA_AOT)
                                .memoryDomain()
                                .directAccess();
        long offset = Indexing.linearToOffset(typed, linearIndex);
        return access.readFloat(typed.memory(), offset);
    }

    private void assertClose(float actual, float expected) {
        assertEquals(expected, actual, 1e-5f);
    }
}
