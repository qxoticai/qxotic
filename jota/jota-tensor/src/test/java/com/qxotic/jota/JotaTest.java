package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.MemoryViewPrinter;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import java.util.Arrays;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Test;

public class JotaTest {

    @AutoClose MemoryDomain<float[]> domain = DomainFactory.ofFloats();

    public static MemoryView<float[]> ofFloatsVector(float... floats) {
        return MemoryViewFactory.of(
                DataType.FP32, MemoryFactory.ofFloats(floats), Layout.rowMajor(floats.length));
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
        MemoryAccess<float[]> memoryAccess1 = domain.directAccess();
        System.out.println(MemoryViewPrinter.toString(view, memoryAccess1));
        view = view.view(Shape.of(2, 3)).permute(1, 0);
        MemoryAccess<float[]> memoryAccess = domain.directAccess();
        System.out.println(MemoryViewPrinter.toString(view, memoryAccess));
    }
}
