package ai.qxotic.jota.runtime.javaaot;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class JavaAotBitwiseComparisonKernelTest {

    private Environment javaAotEnv() {
        return new Environment(
                Device.JAVA_AOT,
                DataType.FP32,
                Environment.current().runtimes(),
                ExecutionMode.LAZY);
    }

    @Test
    void bitwiseAndWithSignedValues() {
        MemoryView<?> out =
                evalI32Binary(
                        Tensor.of(Tensor.iota(4, DataType.I32).subtract(1).materialize()),
                        Tensor.of(Tensor.iota(4, DataType.I32).multiply(2).add(1).materialize()),
                        Tensor::bitwiseAnd);
        assertI32(out, 0, -1 & 1);
        assertI32(out, 1, 0 & 3);
        assertI32(out, 2, 1 & 5);
        assertI32(out, 3, 2 & 7);
    }

    @Test
    void bitwiseOrWithSignedValues() {
        MemoryView<?> out =
                evalI32Binary(
                        Tensor.of(Tensor.iota(4, DataType.I32).subtract(1).materialize()),
                        Tensor.of(Tensor.iota(4, DataType.I32).multiply(2).add(1).materialize()),
                        Tensor::bitwiseOr);
        assertI32(out, 0, -1 | 1);
        assertI32(out, 1, 0 | 3);
        assertI32(out, 2, 1 | 5);
        assertI32(out, 3, 2 | 7);
    }

    @Test
    void bitwiseXorWithSignedValues() {
        MemoryView<?> out =
                evalI32Binary(
                        Tensor.of(Tensor.iota(4, DataType.I32).subtract(1).materialize()),
                        Tensor.of(Tensor.iota(4, DataType.I32).multiply(2).add(1).materialize()),
                        Tensor::bitwiseXor);
        assertI32(out, 0, -1 ^ 1);
        assertI32(out, 1, 0 ^ 3);
        assertI32(out, 2, 1 ^ 5);
        assertI32(out, 3, 2 ^ 7);
    }

    @Test
    void comparisonEqualMixedResults() {
        MemoryView<?> out =
                evalFp32Binary(
                        Tensor.of(Tensor.iota(4, DataType.FP32).add(1f).materialize()),
                        Tensor.of(Tensor.iota(4, DataType.FP32).multiply(2f).materialize()),
                        Tensor::equal);
        assertBool(out, 0, false);
        assertBool(out, 1, true);
        assertBool(out, 2, false);
        assertBool(out, 3, false);
    }

    @Test
    void comparisonLessThanMixedResults() {
        MemoryView<?> out =
                evalFp32Binary(
                        Tensor.of(Tensor.iota(4, DataType.FP32).add(1f).materialize()),
                        Tensor.of(Tensor.iota(4, DataType.FP32).multiply(2f).materialize()),
                        Tensor::lessThan);
        assertBool(out, 0, false);
        assertBool(out, 1, false);
        assertBool(out, 2, true);
        assertBool(out, 3, true);
    }

    private MemoryView<?> evalFp32Binary(
            Tensor left, Tensor right, java.util.function.BiFunction<Tensor, Tensor, Tensor> fn) {
        Environment env = javaAotEnv();
        return Environment.with(env, () -> Tracer.trace(left, right, fn).materialize());
    }

    private MemoryView<?> evalI32Binary(
            Tensor left, Tensor right, java.util.function.BiFunction<Tensor, Tensor, Tensor> fn) {
        Environment env = javaAotEnv();
        return Environment.with(env, () -> Tracer.trace(left, right, fn).materialize());
    }

    @SuppressWarnings("unchecked")
    private int readI32(MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access =
                (MemoryAccess<MemorySegment>)
                        Environment.current()
                                .runtimeFor(Device.JAVA_AOT)
                                .memoryDomain()
                                .directAccess();
        long offset = Indexing.linearToOffset(typed, linearIndex);
        return access.readInt(typed.memory(), offset);
    }

    @SuppressWarnings("unchecked")
    private boolean readBool(MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access =
                (MemoryAccess<MemorySegment>)
                        Environment.current()
                                .runtimeFor(Device.JAVA_AOT)
                                .memoryDomain()
                                .directAccess();
        long offset = Indexing.linearToOffset(typed, linearIndex);
        return access.readByte(typed.memory(), offset) != 0;
    }

    private void assertI32(MemoryView<?> view, int index, int expected) {
        assertEquals(expected, readI32(view, index));
    }

    private void assertBool(MemoryView<?> view, int index, boolean expected) {
        assertEquals(expected, readBool(view, index));
    }
}
