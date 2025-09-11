package com.llm4j.jota.cuda;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.*;
import com.llm4j.jota.memory.impl.MemoryFactory;
import com.llm4j.jota.memory.impl.MemoryViewFactory;
import jcuda.driver.CUdeviceptr;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static com.llm4j.jota.FloatBinaryOperator.sum;
import static com.llm4j.jota.FloatUnaryOperator.identity;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class CUDATest {

    private CUDAContext context;
    private ScopedMemoryAllocator<CUdeviceptr> allocator;
    private MemoryOperations<CUdeviceptr> memoryOperations;
    private MemoryAccess<CUdeviceptr> memoryAccess;
    private FloatOperations<CUdeviceptr> floatOperations;

    @BeforeEach
    void setUp() {
        context = new CUDAContext();
        allocator = CUDAScopedMemoryAllocator.instance();
        memoryOperations = CUDAMemoryOperations.instance();
        memoryAccess = CUDAMemoryAccess.instance();
        floatOperations = CUDAFloatOperations.instance();
    }

    @AfterEach
    void tearDown() {
        context.close();
    }

    @Test
    void testMemoryAllocationAndDeallocation() {
        try (ScopedMemory<CUdeviceptr> memory = allocator.allocateMemory(1024, 4)) {
            assertNotNull(memory.base());
            assertEquals(1024, memory.byteSize());
        }
    }

    @Test
    void testMemoryAccessFloat() {
        try (ScopedMemory<CUdeviceptr> memory = allocator.allocateMemory(4, 4)) {
            float value = 123.45f;
            memoryAccess.writeFloat(memory, 0, value);
            float readValue = memoryAccess.readFloat(memory, 0);
            assertEquals(value, readValue, 0.001f);
        }
    }

    @Test
    void testMemoryCopy() {
        try (ScopedMemory<CUdeviceptr> src = allocator.allocateMemory(16, 4);
             ScopedMemory<CUdeviceptr> dst = allocator.allocateMemory(16, 4);
             Arena arena = Arena.ofConfined()) {

            float[] hostData = {1.0f, 2.0f, 3.0f, 4.0f};
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, hostData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(hostSegment), 0, src, 0, 16);

            memoryOperations.copy(src, 0, dst, 0, 16);

            MemorySegment resultSegment = arena.allocate(16);
            memoryOperations.copyToNative(dst, 0, MemoryFactory.ofMemorySegment(resultSegment), 0, 16);

            float[] result = resultSegment.toArray(ValueLayout.JAVA_FLOAT);

            for (int i = 0; i < hostData.length; i++) {
                assertEquals(hostData[i], result[i], 0.001f);
            }
        }
    }

    @Test
    void testElementWiseAddContiguous() {
        Shape shape = Shape.of(2, 2);
        try (ScopedMemory<CUdeviceptr> leftMem = allocator.allocateMemory(DataType.F32, shape.totalNumberOfElements());
             ScopedMemory<CUdeviceptr> rightMem = allocator.allocateMemory(DataType.F32, shape.totalNumberOfElements());
             ScopedMemory<CUdeviceptr> outMem = allocator.allocateMemory(DataType.F32, shape.totalNumberOfElements());
             Arena arena = Arena.ofConfined()) {

            float[] leftData = {1.0f, 2.0f, 3.0f, 4.0f};
            float[] rightData = {5.0f, 6.0f, 7.0f, 8.0f};
            float[] expected = {6.0f, 8.0f, 10.0f, 12.0f};

            MemorySegment leftHostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, leftData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(leftHostSegment), 0, leftMem, 0, DataType.F32.byteSizeFor(shape));

            MemorySegment rightHostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, rightData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(rightHostSegment), 0, rightMem, 0, DataType.F32.byteSizeFor(shape));

            MemoryView<CUdeviceptr> leftView = MemoryViewFactory.of(shape, DataType.F32, leftMem);
            MemoryView<CUdeviceptr> rightView = MemoryViewFactory.of(shape, DataType.F32, rightMem);
            MemoryView<CUdeviceptr> outView = MemoryViewFactory.of(shape, DataType.F32, outMem);

            floatOperations.elementWise2(leftView, sum(), rightView, outView);

            MemorySegment resultSegment = arena.allocate(DataType.F32.byteSizeFor(shape));
            memoryOperations.copyToNative(outMem, 0, MemoryFactory.ofMemorySegment(resultSegment), 0, DataType.F32.byteSizeFor(shape));
            float[] result = resultSegment.toArray(ValueLayout.JAVA_FLOAT);

            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], result[i], 0.001f);
            }
        }
    }

    @Test
    void testElementWiseScalarAddContiguous() {
        Shape shape = Shape.of(2, 2);
        try (ScopedMemory<CUdeviceptr> inMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shape), 4);
             ScopedMemory<CUdeviceptr> outMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shape), 4);
             Arena arena = Arena.ofConfined()) {

            float[] inData = {1.0f, 2.0f, 3.0f, 4.0f};
            float scalar = 5.0f;
            float[] expected = {6.0f, 7.0f, 8.0f, 9.0f};

            MemorySegment inHostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, inData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(inHostSegment), 0, inMem, 0, DataType.F32.byteSizeFor(shape));

            MemoryView<CUdeviceptr> inView = MemoryViewFactory.of(shape, DataType.F32, inMem);
            MemoryView<CUdeviceptr> outView = MemoryViewFactory.of(shape, DataType.F32, outMem);

            floatOperations.elementWise2(inView, sum(), scalar, outView);

            MemorySegment resultSegment = arena.allocate(DataType.F32.byteSizeFor(shape));
            memoryOperations.copyToNative(outMem, 0, MemoryFactory.ofMemorySegment(resultSegment), 0, DataType.F32.byteSizeFor(shape));
            float[] result = resultSegment.toArray(ValueLayout.JAVA_FLOAT);

            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], result[i], 0.001f);
            }
        }
    }

    @Test
    void testElementWiseUnaryIdentityContiguous() {
        Shape shape = Shape.of(2, 2);
        try (ScopedMemory<CUdeviceptr> inMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shape), 4);
             ScopedMemory<CUdeviceptr> outMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shape), 4);
             Arena arena = Arena.ofConfined()) {

            float[] inData = {1.0f, 2.0f, 3.0f, 4.0f};
            float[] expected = {1.0f, 2.0f, 3.0f, 4.0f};

            MemorySegment inHostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, inData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(inHostSegment), 0, inMem, 0, DataType.F32.byteSizeFor(shape));

            MemoryView<CUdeviceptr> inView = MemoryViewFactory.of(shape, DataType.F32, inMem);
            MemoryView<CUdeviceptr> outView = MemoryViewFactory.of(shape, DataType.F32, outMem);

            floatOperations.elementWise(inView, identity(), outView);

            MemorySegment resultSegment = arena.allocate(DataType.F32.byteSizeFor(shape));
            memoryOperations.copyToNative(outMem, 0, MemoryFactory.ofMemorySegment(resultSegment), 0, DataType.F32.byteSizeFor(shape));
            float[] result = resultSegment.toArray(ValueLayout.JAVA_FLOAT);

            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], result[i], 0.001f);
            }
        }
    }

    @Test
    void testFillWithUnaryScalarContiguous() {
        Shape shape = Shape.of(2, 2);
        try (ScopedMemory<CUdeviceptr> outMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shape), 4);
             Arena arena = Arena.ofConfined()) {

            float scalar = 10.0f;
            float[] expected = {10.0f, 10.0f, 10.0f, 10.0f};

            MemoryView<CUdeviceptr> outView = MemoryViewFactory.of(shape, DataType.F32, outMem);

            floatOperations.elementWise(scalar, identity(), outView);

            MemorySegment resultSegment = arena.allocate(DataType.F32.byteSizeFor(shape));
            memoryOperations.copyToNative(outMem, 0, MemoryFactory.ofMemorySegment(resultSegment), 0, DataType.F32.byteSizeFor(shape));
            float[] result = resultSegment.toArray(ValueLayout.JAVA_FLOAT);

            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], result[i], 0.001f);
            }
        }
    }

    @Test
    void testFoldAllSumContiguous() {
        Shape shape = Shape.of(2, 2);
        try (ScopedMemory<CUdeviceptr> inMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shape), 4);
             Arena arena = Arena.ofConfined()) {
            float[] inData = {1.0f, 2.0f, 3.0f, 4.0f};
            MemorySegment inHostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, inData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(inHostSegment), 0, inMem, 0, DataType.F32.byteSizeFor(shape));
            MemoryView<CUdeviceptr> inView = MemoryViewFactory.of(shape, DataType.F32, inMem);

            float result = floatOperations.foldAll(inView, 0.0f, sum());
            assertEquals(10.0f, result, 0.001f);
        }
    }

    @Test
    void testReduceAllSumContiguous() {
        Shape shape = Shape.of(2, 2);
        try (ScopedMemory<CUdeviceptr> inMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shape), 4);
             Arena arena = Arena.ofConfined()) {
            float[] inData = {1.0f, 2.0f, 3.0f, 4.0f};
            MemorySegment inHostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, inData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(inHostSegment), 0, inMem, 0, DataType.F32.byteSizeFor(shape));
            MemoryView<CUdeviceptr> inView = MemoryViewFactory.of(shape, DataType.F32, inMem);

            float result = floatOperations.reduceAll(inView, sum());
            assertEquals(10.0f, result, 0.001f);
        }
    }

    @Test
    void testMatrixMultiplyContiguous() {
        Shape shapeA = Shape.of(2, 3);
        Shape shapeB = Shape.of(3, 2);
        Shape shapeC = Shape.of(2, 2);

        try (ScopedMemory<CUdeviceptr> aMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shapeA), 4);
             ScopedMemory<CUdeviceptr> bMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shapeB), 4);
             ScopedMemory<CUdeviceptr> cMem = allocator.allocateMemory(DataType.F32.byteSizeFor(shapeC), 4);
             Arena arena = Arena.ofConfined()) {

            float[] aData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float[] bData = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
            float[] expectedC = {58.0f, 64.0f, 139.0f, 154.0f};

            MemorySegment aHostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, aData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(aHostSegment), 0, aMem, 0, DataType.F32.byteSizeFor(shapeA));

            MemorySegment bHostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, bData);
            memoryOperations.copyFromNative(MemoryFactory.ofMemorySegment(bHostSegment), 0, bMem, 0, DataType.F32.byteSizeFor(shapeB));

            MemoryView<CUdeviceptr> aView = MemoryViewFactory.of(shapeA, DataType.F32, aMem);
            MemoryView<CUdeviceptr> bView = MemoryViewFactory.of(shapeB, DataType.F32, bMem);
            MemoryView<CUdeviceptr> cView = MemoryViewFactory.of(shapeC, DataType.F32, cMem);

            floatOperations.matrixMultiply(aView, bView, cView);

            MemorySegment cHostSegment = arena.allocate(DataType.F32.byteSizeFor(shapeC));
            memoryOperations.copyToNative(cMem, 0, MemoryFactory.ofMemorySegment(cHostSegment), 0, DataType.F32.byteSizeFor(shapeC));
            float[] resultC = cHostSegment.toArray(ValueLayout.JAVA_FLOAT);

            for (int i = 0; i < expectedC.length; i++) {
                assertEquals(expectedC[i], resultC[i], 0.001f);
            }
        }
    }
}
