package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Util;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;

import java.util.Arrays;

interface Operations<B> {
    MemoryContext<B> of(DataType dataType);
    void map(MemoryView<B> in, UnaryOp op, MemoryView<B> out);

    default MemoryView<B> cast(MemoryView<B> in, DataType dataType) {
        Shape shape = in.shape();
        MemoryContext<B> context = of(dataType);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        MemoryView<B> view = MemoryViewFactory.allocate(allocator, dataType, shape);
        map(in, UnaryOp.CAST, view);
        return view;
    }

    default MemoryView<B> map(MemoryView<B> in, UnaryOp op) {
        DataType dataType = in.dataType();
        Shape shape = in.shape();
        MemoryContext<B> context = of(dataType);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        MemoryView<B> view = MemoryViewFactory.allocate(allocator, dataType, shape);
        map(in, op, view);
        return view;
    }

    void zip2(MemoryView<B> left, BinaryOp op, MemoryView<B> right, MemoryView<B> out);

    default MemoryView<B> zip2(MemoryView<B> left, BinaryOp op, MemoryView<B> right) {
        if (left.dataType() != right.dataType()) {
            throw new UnsupportedOperationException("mixed dataTypes");
        }
        if (left.shape().flattenModes().equals(right.shape().flattenModes())) {
            throw new UnsupportedOperationException("incompatible shapes");
        }
        DataType dataType = left.dataType();
        Shape shape = left.shape();
        MemoryContext<B> context = of(dataType);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        MemoryView<B> view = MemoryViewFactory.allocate(allocator, dataType, shape);
        zip2(left, op, right, view);
        return view;
    }

    void reduce(MemoryView<B> in, BinaryOp op, MemoryView<B> out, boolean keepDims, int... _axes);

    default MemoryView<B> reduce(MemoryView<B> in, BinaryOp op, boolean keepDims, int... _axes) {
        Shape shape = reduceShape(in.shape(), keepDims, _axes);
        DataType dataType = in.dataType();
        MemoryContext<B> context = of(dataType);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        MemoryView<B> view = MemoryViewFactory.allocate(allocator, dataType, shape);
        reduce(in, op, view, keepDims, _axes);
        return view;
    }

    private static Shape reduceShape(Shape shape, boolean keepDims, int... _axes) {
        Shape outShape = shape;
        int[] axes = Arrays.stream(_axes)
                .map(axis -> Util.wrapAround(axis, shape.rank()))
                .sorted()
                .toArray();
        for (int i = axes.length - 1; i >= 0; i--) {
            int axis = axes[i];
            if (keepDims) {
                outShape = outShape.replace(axis, Shape.of(1));
            } else {
                outShape = outShape.remove(axes[i]);
            }
        }
        return outShape;
    }
}

interface UnaryOp {
    String name();

    UnaryOp IDENTITY = new UnaryOpImpl("identity");
    UnaryOp CAST = new UnaryOpImpl("cast");
    UnaryOp NEGATE = new UnaryOpImpl("negate");
    UnaryOp EXP = new UnaryOpImpl("exp");
    UnaryOp SQRT = new UnaryOpImpl("sqrt");
    UnaryOp ABS = new UnaryOpImpl("abs");

    UnaryOp SIN = new UnaryOpImpl("sin");
    UnaryOp COS = new UnaryOpImpl("cos");
}

record UnaryOpImpl(String name) implements UnaryOp {}

interface BinaryOp {
    String name();
    BinaryOp ADD = new BinaryOpImpl("add");;
    BinaryOp MULTIPLY = new BinaryOpImpl("multiply");
    BinaryOp DIVIDE = new BinaryOpImpl("divide");
    BinaryOp SUBTRACT = new BinaryOpImpl("subtract");
    BinaryOp MIN = new BinaryOpImpl("min");
    BinaryOp MAX = new BinaryOpImpl("max");
}

record BinaryOpImpl(String name) implements BinaryOp {}
