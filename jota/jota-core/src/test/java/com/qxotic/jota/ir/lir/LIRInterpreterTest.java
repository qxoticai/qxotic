package com.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ScalarArg;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.Test;

class LIRInterpreterTest {

    @Test
    void executesElementwiseAdd() {
        MemoryDomain<MemorySegment> domain = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> left =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.flat(4));
        MemoryView<MemorySegment> right =
                MemoryHelpers.full(domain, DataType.FP32, Shape.flat(4), 2.0f);
        MemoryView<MemorySegment> out = MemoryHelpers.zeros(domain, DataType.FP32, Shape.flat(4));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef leftBuf = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef rightBuf = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP32, 4);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        long stride = leftBuf.byteStrides()[0];
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(stride));
        LIRExprNode leftVal = graph.scalarLoad(leftBuf, offset, DataType.FP32);
        LIRExprNode rightVal = graph.scalarLoad(rightBuf, offset, DataType.FP32);
        LIRExprNode sum = graph.scalarBinary(BinaryOperator.ADD, leftVal, rightVal);
        LIRExprNode store = graph.store(outBuf, offset, sum);
        Block body = graph.block(List.of(store, graph.yield(List.of())));
        LIRExprNode loop =
                graph.structuredFor(
                        "i",
                        graph.indexConst(0),
                        graph.indexConst(4),
                        graph.indexConst(1),
                        List.of(),
                        body);

        LIRGraph lir = builder.build(loop);
        new LIRInterpreter().execute(lir, List.of(left, right), List.of(), List.of(out), domain);

        assertEquals(2.0f, readFloat(domain, out, 0), 0.0001f);
        assertEquals(3.0f, readFloat(domain, out, 1), 0.0001f);
        assertEquals(4.0f, readFloat(domain, out, 2), 0.0001f);
        assertEquals(5.0f, readFloat(domain, out, 3), 0.0001f);
    }

    @Test
    void executesReductionWithIterArgs() {
        MemoryDomain<MemorySegment> domain = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> input =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.flat(4));
        MemoryView<MemorySegment> out = MemoryHelpers.zeros(domain, DataType.FP32, Shape.flat(1));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef inputBuf = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP32, 1);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        long stride = inputBuf.byteStrides()[0];
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(stride));
        LIRExprNode value = graph.scalarLoad(inputBuf, offset, DataType.FP32);

        LIRExprNode accRef = graph.scalarRef("acc", DataType.FP32);
        LIRExprNode sum = graph.scalarBinary(BinaryOperator.ADD, accRef, value);
        Block body = graph.block(List.of(graph.yield(List.of(sum))));

        LIRExprNode loop =
                graph.structuredFor(
                        "i",
                        graph.indexConst(0),
                        graph.indexConst(4),
                        graph.indexConst(1),
                        List.of(
                                new LoopIterArg(
                                        "acc",
                                        DataType.FP32,
                                        graph.scalarConst(0L, DataType.FP32))),
                        body);

        LIRExprNode store = graph.store(outBuf, graph.indexConst(0), accRef);
        LIRExprNode root = graph.block(List.of(loop, store));

        LIRGraph lir = builder.build(root);
        new LIRInterpreter().execute(lir, List.of(input), List.of(), List.of(out), domain);

        assertEquals(6.0f, readFloat(domain, out, 0), 0.0001f);
    }

    @Test
    void supportsScalarInputs() {
        MemoryDomain<MemorySegment> domain = Environment.current().nativeMemoryDomain();
        MemoryView<MemorySegment> input =
                MemoryHelpers.arange(domain, DataType.FP32, 3).view(Shape.flat(3));
        MemoryView<MemorySegment> out = MemoryHelpers.zeros(domain, DataType.FP32, Shape.flat(3));

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef inputBuf = builder.addContiguousInput(DataType.FP32, 3);
        ScalarInput scalar = builder.addScalarInput(DataType.FP32);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP32, 3);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        long stride = inputBuf.byteStrides()[0];
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(stride));
        LIRExprNode value = graph.scalarLoad(inputBuf, offset, DataType.FP32);
        LIRExprNode scalarVal = graph.scalarInput(scalar.id(), scalar.dataType());
        LIRExprNode sum = graph.scalarBinary(BinaryOperator.ADD, value, scalarVal);
        LIRExprNode store = graph.store(outBuf, offset, sum);
        Block body = graph.block(List.of(store, graph.yield(List.of())));
        LIRExprNode loop =
                graph.structuredFor(
                        "i",
                        graph.indexConst(0),
                        graph.indexConst(3),
                        graph.indexConst(1),
                        List.of(),
                        body);

        LIRGraph lir = builder.build(loop);
        new LIRInterpreter()
                .execute(
                        lir,
                        List.of(input),
                        List.of(ScalarArg.ofFloat(10.0f)),
                        List.of(out),
                        domain);

        assertEquals(10.0f, readFloat(domain, out, 0), 0.0001f);
        assertEquals(11.0f, readFloat(domain, out, 1), 0.0001f);
        assertEquals(12.0f, readFloat(domain, out, 2), 0.0001f);
    }

    private float readFloat(
            MemoryDomain<MemorySegment> domain, MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = domain.directAccess();
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return access.readFloat(typedView.memory(), offset);
    }
}
