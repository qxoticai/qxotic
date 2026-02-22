package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.AbstractMemoryTest;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

@Disabled
class UnaryActivationOpsTest extends AbstractMemoryTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
    }

    @Test
    void reluWithFloatInput() {
        Tensor input = Tensor.of(new float[] {-2.0f, -1.0f, 0.0f, 1.0f});
        Tensor result = input.relu();
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(4), output.shape());
        assertEquals(0.0f, readFloat(output, 0), 0.0001f);
        assertEquals(0.0f, readFloat(output, 1), 0.0001f);
        assertEquals(0.0f, readFloat(output, 2), 0.0001f);
        assertEquals(1.0f, readFloat(output, 3), 0.0001f);
    }

    @Test
    void reluWithNegativeValues() {
        Tensor input = Tensor.of(new float[] {-10.0f, -5.0f, -2.0f, -1.0f});
        Tensor result = input.relu();
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(4), output.shape());
        assertEquals(0.0f, readFloat(output, 0), 0.0001f);
        assertEquals(0.0f, readFloat(output, 1), 0.0001f);
        assertEquals(0.0f, readFloat(output, 2), 0.0001f);
        assertEquals(0.0f, readFloat(output, 3), 0.0001f);
    }

    @Test
    void reluWithScalar() {
        Tensor input = Tensor.of(new float[] {-5.0f});
        Tensor result = input.relu();
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(1), output.shape());
        assertEquals(0.0f, readFloat(output, 0), 0.0001f);
    }

    @Test
    void reluWithLargeValues() {
        Tensor input = Tensor.of(new float[] {100.0f, 200.0f, 300.0f});
        Tensor result = input.relu();
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(3), output.shape());
        assertEquals(100.0f, readFloat(output, 0), 0.0001f);
        assertEquals(200.0f, readFloat(output, 1), 0.0001f);
        assertEquals(300.0f, readFloat(output, 2), 0.0001f);
    }

    @Test
    void sigmoidWithFloatInput() {
        Tensor input = Tensor.of(new float[] {-2.0f, -1.0f, 0.0f, 1.0f});
        Tensor result = input.sigmoid();
        MemoryView<?> output = result.materialize();

        float delta = 0.0001f;
        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(5), output.shape());
        assertEquals(0.1192f, readFloat(output, 0), delta);
        assertEquals(0.7311f, readFloat(output, 1), delta);
        assertEquals(0.8808f, readFloat(output, 2), delta);
        assertEquals(0.9526f, readFloat(output, 3), delta);
        assertEquals(0.9933f, readFloat(output, 4), delta);
    }

    @Test
    void sigmoidWithNegativeValues() {
        Tensor input = Tensor.of(new float[] {-10.0f, -5.0f, -2.0f, -1.0f});
        Tensor result = input.sigmoid();
        MemoryView<?> output = result.materialize();

        float delta = 0.0001f;
        assertEquals(0.000045f, readFloat(output, 0), delta);
        assertEquals(0.0067f, readFloat(output, 1), delta);
        assertEquals(0.1192f, readFloat(output, 2), delta);
        assertEquals(0.8808f, readFloat(output, 3), delta);
        assertEquals(0.2689f, readFloat(output, 4), delta);
    }

    @Test
    void sigmoidWithScalar() {
        Tensor input = Tensor.of(new float[] {0.0f});
        Tensor result = input.sigmoid();
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(1), output.shape());
        assertEquals(0.5f, readFloat(output, 0), 0.0001f);
    }

    @Test
    void sigmoidWithLargeValues() {
        Tensor input = Tensor.of(new float[] {10.0f, 50.0f, 100.0f});
        Tensor result = input.sigmoid();
        MemoryView<?> output = result.materialize();

        float delta = 0.0001f;
        assertEquals(0.9999546f, readFloat(output, 0), delta);
        assertEquals(1.0f, readFloat(output, 1), 0.001f);
        assertEquals(1.0f, readFloat(output, 2), 0.001f);
    }

    @Test
    void siluWithFloatInput() {
        Tensor input = Tensor.of(new float[] {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
        Tensor result = input.silu();
        MemoryView<?> output = result.materialize();

        float delta = 0.0001f;
        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(5), output.shape());
        assertEquals(-0.2689f, readFloat(output, 0), delta);
        assertEquals(-0.2689f, readFloat(output, 1), delta);
        assertEquals(-0.1192f, readFloat(output, 2), delta);
        assertEquals(0.0000f, readFloat(output, 3), delta);
        assertEquals(0.4621f, readFloat(output, 4), delta);
    }

    @Test
    void siluWithNegativeValues() {
        Tensor input = Tensor.of(new float[] {-10.0f, -5.0f, -2.0f, -1.0f});
        Tensor result = input.silu();
        MemoryView<?> output = result.materialize();

        float delta = 0.0001f;
        assertEquals(-0.0000500f, readFloat(output, 0), delta);
        assertEquals(-0.0067f, readFloat(output, 1), delta);
        assertEquals(-0.1192f, readFloat(output, 2), delta);
        assertEquals(-0.2689f, readFloat(output, 3), delta);
        assertEquals(-0.4621f, readFloat(output, 4), 0.0001f);
    }

    @Test
    void siluWithScalar() {
        Tensor input = Tensor.of(new float[] {-5.0f});
        Tensor result = input.silu();
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(1), output.shape());
        assertEquals(-0.00669f, readFloat(output, 0), 0.0001f);
    }

    @Test
    void siluWithLargeValues() {
        Tensor input = Tensor.of(new float[] {10.0f, 50.0f, 100.0f});
        Tensor result = input.silu();
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(3), output.shape());
        assertEquals(0.0f, readFloat(output, 0), 0.001f);
        assertEquals(1.0f, readFloat(output, 1), 0.001f);
        assertEquals(0.9999085f, readFloat(output, 2), 0.001f);
    }

    @Test
    void geluWithFloatInput() {
        Tensor input = Tensor.of(new float[] {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
        Tensor result = input.gelu();
        MemoryView<?> output = result.materialize();

        float delta = 0.001f;
        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(5), output.shape());
        assertEquals(-0.0506f, readFloat(output, 0), delta);
        assertEquals(-0.1559f, readFloat(output, 1), delta);
        assertEquals(0.5f, readFloat(output, 2), delta);
        assertEquals(0.1570f, readFloat(output, 3), delta);
        assertEquals(0.4211f, readFloat(output, 4), delta);
    }

    @Test
    void geluWithNegativeValues() {
        Tensor input = Tensor.of(new float[] {-10.0f, -5.0f, -2.0f, -1.0f});
        Tensor result = input.gelu();
        MemoryView<?> output = result.materialize();

        float delta = 0.001f;
        assertEquals(-0.0506f, readFloat(output, 0), delta);
        assertEquals(-0.0001559f, readFloat(output, 1), delta);
        assertEquals(0.0f, readFloat(output, 2), delta);
        assertEquals(0.000f, readFloat(output, 3), delta);
        assertEquals(0.0f, readFloat(output, 4), delta);
    }

    @Test
    void geluWithScalar() {
        Tensor input = Tensor.of(new float[] {-5.0f});
        Tensor result = input.gelu();
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(1), output.shape());
        assertEquals(-0.0000506f, readFloat(output, 0), 0.001f);
    }

    @Test
    void geluWithLargeValues() {
        Tensor input = Tensor.of(new float[] {10.0f, 50.0f, 100.0f});
        Tensor result = input.gelu();
        MemoryView<?> output = result.materialize();

        float delta = 0.001f;
        assertEquals(10.0f, readFloat(output, 0), 0.001f);
        assertEquals(50.0f, readFloat(output, 1), 0.001f);
        assertEquals(100.0f, readFloat(output, 2), 0.001f);
    }

    private float readFloat(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
    }
}
