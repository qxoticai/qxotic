package com.qxotic.jota;

import com.llm4j.jota.FloatBinaryOperator;
import com.qxotic.jota.memory.*;
import com.qxotic.jota.memory.impl.ContextFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class JotaTest {

    @AutoClose
    MemoryContext<float[]> context = ContextFactory.ofFloats();

    public static MemoryView<float[]> ofFloatsVector(float... floats) {
        return MemoryViewFactory.of(DataType.FP32, MemoryFactory.ofFloats(floats), Layout.rowMajor(floats.length));
    }

    public static MemoryView<float[]> full(float value, Shape shape) {
        var floats = new float[Math.toIntExact(shape.size())];
        Arrays.fill(floats, value);
        return ofFloatsVector(floats).view(shape);
    }

    public static MemoryView<float[]> ones(Shape shape) {
        return full(1f, shape);
    }

    public static MemoryView<float[]> zeros(Shape shape) {
        return full(0f, shape);
    }

    public static MemoryView<float[]> range(int fromInclusive, int toExclusive) {
        int n = toExclusive - fromInclusive;
        var floats = new float[n];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = (float) (fromInclusive + i);
        }
        return ofFloatsVector(floats);
    }

    public static MemoryView<float[]> range(int toExclusive) {
        return range(0, toExclusive);
    }

    @Test
    void testOps() {
        var view2x3 = range(6).view(Shape.of(2, 3));

        var view2 = zeros(Shape.of(2));
        var view3 = zeros(Shape.of(3));

        FloatOperations<float[]> ops = context.floatOperations();
        // ops.fold(view2x3, FloatBinaryOperator.sum(), 0f, view3, 0);
        // ops.fold(view2x3, FloatBinaryOperator.sum(), 0f, view2, 1);

        ops.reduce(view2x3, FloatBinaryOperator.sum(), view3, 0);
        ops.reduce(view2x3, FloatBinaryOperator.sum(), view2, 1);

        MemoryAccess<float[]> memoryAccess2 = context.memoryAccess();
        System.out.println(MemoryViewPrinter.toString(view2x3, memoryAccess2));
        MemoryAccess<float[]> memoryAccess1 = context.memoryAccess();
        System.out.println(MemoryViewPrinter.toString(view2, memoryAccess1));
        MemoryAccess<float[]> memoryAccess = context.memoryAccess();
        System.out.println(MemoryViewPrinter.toString(view3, memoryAccess));
    }

    @Test
    void testEmpty() {
        MemoryView<float[]> view = ofFloatsVector();
        assertTrue(view.shape().hasZeroElements());
    }

    @Test
    void testScalar() {
        MemoryView<float[]> view = ofFloatsVector(1.23f).view(Shape.scalar());
        assertTrue(view.shape().isScalar());
    }

    @Test
    void testCreate() {
        MemoryView<float[]> view = range(2 * 3);
        MemoryAccess<float[]> memoryAccess1 = context.memoryAccess();
        System.out.println(MemoryViewPrinter.toString(view, memoryAccess1));
        view = view.view(Shape.of(2, 3)).permute(1, 0);
        MemoryAccess<float[]> memoryAccess = context.memoryAccess();
        System.out.println(MemoryViewPrinter.toString(view, memoryAccess));
    }
}
