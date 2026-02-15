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

class JavaAotUnaryBinaryKernelTest {

    private Environment javaAotEnv() {
        return new Environment(
                Device.JAVA_AOT,
                DataType.FP32,
                Environment.current().runtimes(),
                ExecutionMode.LAZY);
    }

    @Test
    void unaryNegate() {
        MemoryView<?> out = unaryFp32(t -> t.negate());
        assertFp32(out, 0, -1f);
        assertFp32(out, 1, -2f);
        assertFp32(out, 2, -3f);
        assertFp32(out, 3, -4f);
    }

    @Test
    void unaryAbs() {
        MemoryView<?> out = unaryFp32(t -> t.negate().abs());
        assertFp32(out, 0, 1f);
        assertFp32(out, 1, 2f);
        assertFp32(out, 2, 3f);
        assertFp32(out, 3, 4f);
    }

    @Test
    void unaryExp() {
        MemoryView<?> out = unaryFp32(t -> t.exp());
        assertClose(readFp32(out, 0), (float) Math.exp(1.0));
        assertClose(readFp32(out, 1), (float) Math.exp(2.0));
        assertClose(readFp32(out, 2), (float) Math.exp(3.0));
        assertClose(readFp32(out, 3), (float) Math.exp(4.0));
    }

    @Test
    void unaryLog() {
        MemoryView<?> out = unaryFp32(t -> t.log());
        assertClose(readFp32(out, 0), (float) Math.log(1.0));
        assertClose(readFp32(out, 1), (float) Math.log(2.0));
        assertClose(readFp32(out, 2), (float) Math.log(3.0));
        assertClose(readFp32(out, 3), (float) Math.log(4.0));
    }

    @Test
    void unarySqrt() {
        MemoryView<?> out = unaryFp32(t -> t.sqrt());
        assertClose(readFp32(out, 0), 1f);
        assertClose(readFp32(out, 1), (float) Math.sqrt(2.0));
        assertClose(readFp32(out, 2), (float) Math.sqrt(3.0));
        assertClose(readFp32(out, 3), 2f);
    }

    @Test
    void unarySquare() {
        MemoryView<?> out = unaryFp32(t -> t.square());
        assertFp32(out, 0, 1f);
        assertFp32(out, 1, 4f);
        assertFp32(out, 2, 9f);
        assertFp32(out, 3, 16f);
    }

    @Test
    void unarySin() {
        MemoryView<?> out = unaryFp32(t -> t.sin());
        assertClose(readFp32(out, 0), (float) Math.sin(1.0));
        assertClose(readFp32(out, 1), (float) Math.sin(2.0));
        assertClose(readFp32(out, 2), (float) Math.sin(3.0));
        assertClose(readFp32(out, 3), (float) Math.sin(4.0));
    }

    @Test
    void unaryCos() {
        MemoryView<?> out = unaryFp32(t -> t.cos());
        assertClose(readFp32(out, 0), (float) Math.cos(1.0));
        assertClose(readFp32(out, 1), (float) Math.cos(2.0));
        assertClose(readFp32(out, 2), (float) Math.cos(3.0));
        assertClose(readFp32(out, 3), (float) Math.cos(4.0));
    }

    @Test
    void unaryTanh() {
        MemoryView<?> out = unaryFp32(t -> t.tanh());
        assertClose(readFp32(out, 0), (float) Math.tanh(1.0));
        assertClose(readFp32(out, 1), (float) Math.tanh(2.0));
        assertClose(readFp32(out, 2), (float) Math.tanh(3.0));
        assertClose(readFp32(out, 3), (float) Math.tanh(4.0));
    }

    @Test
    void unaryReciprocal() {
        MemoryView<?> out = unaryFp32(t -> t.reciprocal());
        assertClose(readFp32(out, 0), 1f);
        assertClose(readFp32(out, 1), 0.5f);
        assertClose(readFp32(out, 2), 1f / 3f);
        assertClose(readFp32(out, 3), 0.25f);
    }

    @Test
    void unaryLogicalNot() {
        Environment env = javaAotEnv();
        MemoryView<?> out =
                Environment.with(
                        env,
                        () -> {
                            Tensor input =
                                    Tensor.of(
                                            Tensor.iota(4, DataType.I32)
                                                    .cast(DataType.BOOL)
                                                    .materialize());
                            return Tracer.trace(input, Tensor::logicalNot).materialize();
                        });
        assertBool(out, 0, true);
        assertBool(out, 1, false);
        assertBool(out, 2, false);
        assertBool(out, 3, false);
    }

    @Test
    void unaryBitwiseNot() {
        Environment env = javaAotEnv();
        MemoryView<?> out =
                Environment.with(
                        env,
                        () -> {
                            Tensor input =
                                    Tensor.of(Tensor.iota(4, DataType.I32).add(1).materialize());
                            return Tracer.trace(input, Tensor::bitwiseNot).materialize();
                        });
        assertI32(out, 0, ~1);
        assertI32(out, 1, ~2);
        assertI32(out, 2, ~3);
        assertI32(out, 3, ~4);
    }

    @Test
    void binaryAdd() {
        MemoryView<?> out = binaryFp32((a, b) -> a.add(b));
        assertFp32(out, 0, 6f);
        assertFp32(out, 1, 8f);
        assertFp32(out, 2, 10f);
        assertFp32(out, 3, 12f);
    }

    @Test
    void binarySubtract() {
        MemoryView<?> out = binaryFp32((a, b) -> b.subtract(a));
        assertFp32(out, 0, 4f);
        assertFp32(out, 1, 4f);
        assertFp32(out, 2, 4f);
        assertFp32(out, 3, 4f);
    }

    @Test
    void binaryMultiply() {
        MemoryView<?> out = binaryFp32((a, b) -> a.multiply(b));
        assertFp32(out, 0, 5f);
        assertFp32(out, 1, 12f);
        assertFp32(out, 2, 21f);
        assertFp32(out, 3, 32f);
    }

    @Test
    void binaryDivide() {
        MemoryView<?> out = binaryFp32((a, b) -> b.divide(a));
        assertFp32(out, 0, 5f);
        assertFp32(out, 1, 3f);
        assertFp32(out, 2, 7f / 3f);
        assertFp32(out, 3, 2f);
    }

    @Test
    void binaryMin() {
        MemoryView<?> out = binaryFp32((a, b) -> a.min(b));
        assertFp32(out, 0, 1f);
        assertFp32(out, 1, 2f);
        assertFp32(out, 2, 3f);
        assertFp32(out, 3, 4f);
    }

    @Test
    void binaryMax() {
        MemoryView<?> out = binaryFp32((a, b) -> a.max(b));
        assertFp32(out, 0, 5f);
        assertFp32(out, 1, 6f);
        assertFp32(out, 2, 7f);
        assertFp32(out, 3, 8f);
    }

    @Test
    void binaryLogicalAnd() {
        MemoryView<?> out = binaryBool((a, b) -> a.logicalAnd(b));
        assertBool(out, 0, true);
        assertBool(out, 1, false);
        assertBool(out, 2, false);
        assertBool(out, 3, false);
    }

    @Test
    void binaryLogicalOr() {
        MemoryView<?> out = binaryBool((a, b) -> a.logicalOr(b));
        assertBool(out, 0, true);
        assertBool(out, 1, true);
        assertBool(out, 2, false);
        assertBool(out, 3, false);
    }

    @Test
    void binaryLogicalXor() {
        MemoryView<?> out = binaryBool((a, b) -> a.logicalXor(b));
        assertBool(out, 0, false);
        assertBool(out, 1, true);
        assertBool(out, 2, false);
        assertBool(out, 3, false);
    }

    @Test
    void binaryBitwiseAnd() {
        MemoryView<?> out = binaryI32((a, b) -> a.bitwiseAnd(b));
        assertI32(out, 0, 1 & 5);
        assertI32(out, 1, 2 & 6);
        assertI32(out, 2, 3 & 7);
        assertI32(out, 3, 4 & 8);
    }

    @Test
    void binaryBitwiseOr() {
        MemoryView<?> out = binaryI32((a, b) -> a.bitwiseOr(b));
        assertI32(out, 0, 1 | 5);
        assertI32(out, 1, 2 | 6);
        assertI32(out, 2, 3 | 7);
        assertI32(out, 3, 4 | 8);
    }

    @Test
    void binaryBitwiseXor() {
        MemoryView<?> out = binaryI32((a, b) -> a.bitwiseXor(b));
        assertI32(out, 0, 1 ^ 5);
        assertI32(out, 1, 2 ^ 6);
        assertI32(out, 2, 3 ^ 7);
        assertI32(out, 3, 4 ^ 8);
    }

    @Test
    void binaryEqual() {
        Environment env = javaAotEnv();
        MemoryView<?> out =
                Environment.with(
                        env,
                        () -> {
                            Tensor left = Tensor.of(Tensor.iota(4, DataType.FP32).materialize());
                            Tensor right =
                                    Tensor.of(Tensor.iota(4, DataType.FP32).add(1f).materialize());
                            return Tracer.trace(left, right, Tensor::equal).materialize();
                        });
        assertBool(out, 0, false);
        assertBool(out, 1, false);
        assertBool(out, 2, false);
        assertBool(out, 3, false);
    }

    @Test
    void binaryLessThan() {
        Environment env = javaAotEnv();
        MemoryView<?> out =
                Environment.with(
                        env,
                        () -> {
                            Tensor left = Tensor.of(Tensor.iota(4, DataType.FP32).materialize());
                            Tensor right =
                                    Tensor.of(Tensor.iota(4, DataType.FP32).add(1f).materialize());
                            return Tracer.trace(left, right, Tensor::lessThan).materialize();
                        });
        assertBool(out, 0, true);
        assertBool(out, 1, true);
        assertBool(out, 2, true);
        assertBool(out, 3, true);
    }

    private MemoryView<?> unaryFp32(java.util.function.Function<Tensor, Tensor> fn) {
        Environment env = javaAotEnv();
        return Environment.with(
                env,
                () -> {
                    Tensor input = Tensor.of(Tensor.iota(4, DataType.FP32).add(1f).materialize());
                    return Tracer.trace(input, fn).materialize();
                });
    }

    private MemoryView<?> binaryFp32(java.util.function.BiFunction<Tensor, Tensor, Tensor> fn) {
        Environment env = javaAotEnv();
        return Environment.with(
                env,
                () -> {
                    Tensor left = Tensor.of(Tensor.iota(4, DataType.FP32).add(1f).materialize());
                    Tensor right = Tensor.of(Tensor.iota(4, DataType.FP32).add(5f).materialize());
                    return Tracer.trace(left, right, fn).materialize();
                });
    }

    private MemoryView<?> binaryI32(java.util.function.BiFunction<Tensor, Tensor, Tensor> fn) {
        Environment env = javaAotEnv();
        return Environment.with(
                env,
                () -> {
                    Tensor left = Tensor.of(Tensor.iota(4, DataType.I32).add(1).materialize());
                    Tensor right = Tensor.of(Tensor.iota(4, DataType.I32).add(5).materialize());
                    return Tracer.trace(left, right, fn).materialize();
                });
    }

    private MemoryView<?> binaryBool(java.util.function.BiFunction<Tensor, Tensor, Tensor> fn) {
        Environment env = javaAotEnv();
        return Environment.with(
                env,
                () -> {
                    Tensor base = Tensor.of(Tensor.iota(4, DataType.FP32).materialize());
                    Tensor a = base.lessThan(Tensor.scalar(2f));
                    Tensor b = base.lessThan(Tensor.scalar(1f));
                    return Tracer.trace(a, b, fn).materialize();
                });
    }

    private void assertFp32(MemoryView<?> view, int index, float expected) {
        assertClose(readFp32(view, index), expected);
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

    private void assertClose(float actual, float expected) {
        assertEquals(expected, actual, 1e-5f);
    }
}
