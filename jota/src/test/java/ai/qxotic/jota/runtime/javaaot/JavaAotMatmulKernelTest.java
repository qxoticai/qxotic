package ai.qxotic.jota.runtime.javaaot;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class JavaAotMatmulKernelTest {

    private Environment javaAotEnv() {
        return new Environment(
                Device.JAVA_AOT,
                DataType.FP32,
                Environment.current().runtimes(),
                ExecutionMode.LAZY);
    }

    @Test
    void matmulWorksOnJavaAot() {
        Environment env = javaAotEnv();
        MemoryView<?> output =
                Environment.with(
                        env,
                        () -> {
                            Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                            Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                            return a.matmul(b).materialize();
                        });

        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFp32(output, 0), 22f);
        assertClose(readFp32(output, 1), 28f);
        assertClose(readFp32(output, 2), 49f);
        assertClose(readFp32(output, 3), 64f);
    }

    @Test
    void batchedMatmulWorksOnJavaAot() {
        Environment env = javaAotEnv();
        MemoryView<?> output =
                Environment.with(
                        env,
                        () -> {
                            Tensor a =
                                    Tensor.iota(12, DataType.FP32).add(1f).view(Shape.of(2, 2, 3));
                            Tensor b =
                                    Tensor.iota(12, DataType.FP32).add(1f).view(Shape.of(2, 3, 2));
                            return a.batchedMatmul(b).materialize();
                        });

        assertEquals(Shape.of(2, 2, 2), output.shape());
        assertClose(readFp32(output, 0), 22f);
        assertClose(readFp32(output, 1), 28f);
        assertClose(readFp32(output, 2), 49f);
        assertClose(readFp32(output, 3), 64f);
        assertClose(readFp32(output, 4), 220f);
        assertClose(readFp32(output, 5), 244f);
        assertClose(readFp32(output, 6), 301f);
        assertClose(readFp32(output, 7), 334f);
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
