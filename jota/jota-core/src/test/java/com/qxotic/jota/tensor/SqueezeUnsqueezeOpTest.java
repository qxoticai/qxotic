package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class SqueezeUnsqueezeOpTest {

    @Test
    void unsqueezeInsertsAxisAtEndForMinusOne() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor result = input.unsqueeze(-1);

        assertEquals(Shape.of(2, 3, 1), result.shape());
        assertEquals(0, result.layout().stride().flatAt(2));
    }

    @Test
    void unsqueezeSupportsPostOpWrapAroundAtStart() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor result = input.unsqueeze(-3);

        assertEquals(Shape.of(1, 2, 3), result.shape());
        assertEquals(0, result.layout().stride().flatAt(0));
    }

    @Test
    void unsqueezeOnScalarProducesVectorOfOne() {
        Tensor scalar = Tensor.scalar(42.0f);

        Tensor result = scalar.unsqueeze(-1);

        assertEquals(Shape.of(1), result.shape());
    }

    @Test
    void unsqueezeRejectsAxisOutOfRange() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        assertThrows(IllegalArgumentException.class, () -> input.unsqueeze(3));
        assertThrows(IllegalArgumentException.class, () -> input.unsqueeze(-4));
    }

    @Test
    void squeezeRemovesSingletonAxis() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 1, 3));

        Tensor result = input.squeeze(1);

        assertEquals(Shape.of(2, 3), result.shape());
    }

    @Test
    void squeezeSupportsNegativeAxis() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 1, 3));

        Tensor result = input.squeeze(-2);

        assertEquals(Shape.of(2, 3), result.shape());
    }

    @Test
    void squeezeRejectsNonSingletonAxis() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        assertThrows(IllegalArgumentException.class, () -> input.squeeze(1));
    }

    @Test
    void squeezeRejectsScalarTensor() {
        Tensor scalar = Tensor.scalar(1.0f);

        assertThrows(IllegalArgumentException.class, () -> scalar.squeeze(0));
    }

    @Test
    void squeezeAllRemovesAllSingletonModes() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(1, 2, 1, 3, 1));

        Tensor result = input.squeezeAll();

        assertEquals(Shape.of(2, 3), result.shape());
    }

    @Test
    void squeezeAllPreservesNestedLayout() {
        Tensor input = Tensor.iota(30, DataType.FP32).view(Shape.of(1, 2, Shape.of(3, 5), 1));

        Tensor result = input.squeezeAll();

        assertEquals(Shape.of(2, Shape.of(3, 5)), result.shape());
    }

    @Test
    void squeezeAllOnScalarReturnsSameTensor() {
        Tensor scalar = Tensor.scalar(1.0f);

        Tensor result = scalar.squeezeAll();

        assertSame(scalar, result);
    }

    @Test
    void squeezeAllWhenNoSingletonModesReturnsSameTensor() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor result = input.squeezeAll();

        assertSame(input, result);
    }

    @Test
    void squeezeAllWorksThroughTracing() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(1, 2, 1, 3));

        Tensor traced = Tracer.trace(input, Tensor::squeezeAll);

        assertEquals(Shape.of(2, 3), traced.shape());
        assertEquals(Shape.of(2, 3), traced.materialize().shape());
    }

    @Test
    void unsqueezeThenSqueezeRoundTripShape() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor result = input.unsqueeze(-1).squeeze(-1);

        assertEquals(input.shape(), result.shape());
    }

    @Test
    void unsqueezePreservesNestedLayoutAtEnd() {
        Tensor input = Tensor.iota(30, DataType.FP32).view(Shape.of(2, Shape.of(3, 5)));

        Tensor result = input.unsqueeze(-1);

        assertEquals(Shape.of(2, Shape.of(3, 5), 1), result.shape());
        assertEquals(0, result.layout().stride().flatAt(3));
    }

    @Test
    void unsqueezePreservesNestedLayoutAtFront() {
        Tensor input = Tensor.iota(30, DataType.FP32).view(Shape.of(2, Shape.of(3, 5)));
        Tensor result = input.unsqueeze(0);
        assertEquals(Shape.of(1, 2, Shape.of(3, 5)), result.shape());
        assertEquals(Stride.of(0, 15, Stride.of(5, 1)), result.stride());
    }

    @Test
    void squeezePreservesNestedLayout() {
        Tensor input = Tensor.iota(30, DataType.FP32).view(Shape.of(1, 2, Shape.of(3, 5)));

        Tensor result = input.squeeze(0);

        assertEquals(Shape.of(2, Shape.of(3, 5)), result.shape());
        assertEquals(Stride.of(15, Stride.of(5, 1)), result.stride());
    }

    @Test
    void squeezeRejectsNestedModeWithSizeNotOne() {
        Tensor input = Tensor.iota(30, DataType.FP32).view(Shape.of(2, Shape.of(3, 5)));

        assertThrows(IllegalArgumentException.class, () -> input.squeeze(1));
    }

    @Test
    void nestedUnsqueezeSqueezeRoundTripShape() {
        Tensor input = Tensor.iota(30, DataType.FP32).view(Shape.of(2, Shape.of(3, 5)));
        var original = input.materialize();

        Tensor result = input.unsqueeze(-1).squeeze(-1);
        var materialized = result.materialize();

        assertEquals(input.shape(), result.shape());
        assertArrayEquals(
                original.layout().stride().toArray(), materialized.layout().stride().toArray());
    }

    @Test
    void unsqueezeAndSqueezeWorkThroughTracing() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor traced = Tracer.trace(input, t -> t.unsqueeze(-1).squeeze(-1));

        assertEquals(Shape.of(2, 3), traced.shape());
        assertEquals(Shape.of(2, 3), traced.materialize().shape());
    }
}
