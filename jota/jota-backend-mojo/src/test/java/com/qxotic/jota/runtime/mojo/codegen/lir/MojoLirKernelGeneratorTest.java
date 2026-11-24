package com.qxotic.jota.runtime.mojo.codegen.lir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.Block;
import com.qxotic.jota.ir.lir.BufferRef;
import com.qxotic.jota.ir.lir.IndexBinaryOp;
import com.qxotic.jota.ir.lir.LIRExprGraph;
import com.qxotic.jota.ir.lir.LIRExprNode;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LoopIterArg;
import com.qxotic.jota.ir.lir.ScalarInput;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.ir.tir.UnaryOperator;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.mojo.MojoCachePaths;
import com.qxotic.jota.runtime.mojo.codegen.MojoDialect;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MojoLirKernelGeneratorTest {

    @TempDir Path tempDir;

    @AfterEach
    void clearCacheRootProperty() {
        System.clearProperty(KernelCachePaths.CACHE_ROOT_PROPERTY);
        System.clearProperty(KernelCachePaths.VERSION_PROPERTY);
    }

    @Test
    void generatesKernelProgramFromLirGraph() {
        useTempCache();

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

        MojoLirKernelGenerator generator = new MojoLirKernelGenerator();
        KernelProgram program =
                generator.generate(lir, ScratchLayout.EMPTY, KernelCacheKey.of("lir_to_mojo_add"));

        assertEquals(KernelProgram.Kind.SOURCE, program.kind());
        assertEquals("mojo", program.language());
        assertTrue(program.entryPoint().startsWith("hip_lir_"));

        String source = (String) program.payload();
        assertTrue(source.contains("from max.gpu import block_idx, block_dim, thread_idx"));
        assertTrue(source.contains("from std.math import"));
        assertTrue(source.contains("def hip_lir_"));
        assertTrue(source.contains("var gid ="));
        assertTrue(source.contains("input0: UnsafePointer[Float32, MutAnyOrigin]"));
        assertTrue(source.contains("input1: UnsafePointer[Float32, MutAnyOrigin]"));
        assertTrue(source.contains("+"));
        assertTrue(source.contains("output0: UnsafePointer[Float32, MutAnyOrigin]"));

        Path cachedMojo = MojoCachePaths.lirSourcePath("lir_to_mojo_add");
        assertTrue(Files.exists(cachedMojo));
    }

    @Test
    void generatesNativeMojoForReductionWithScalarInput() {
        useTempCache();

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef inputBuf = builder.addContiguousInput(DataType.FP32, 4);
        ScalarInput scale = builder.addScalarInput(DataType.FP32);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP32, 1);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(4));
        LIRExprNode value = graph.scalarLoad(inputBuf, offset, DataType.FP32);
        LIRExprNode scaled =
                graph.scalarBinary(
                        BinaryOperator.MULTIPLY,
                        value,
                        graph.scalarInput(scale.id(), scale.dataType()));

        LIRExprNode accRef = graph.scalarRef("acc", DataType.FP32);
        LIRExprNode sum = graph.scalarBinary(BinaryOperator.ADD, accRef, scaled);
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
        LIRGraph lir = builder.build(graph.block(List.of(loop, store)));

        MojoLirKernelGenerator generator = new MojoLirKernelGenerator();
        KernelProgram program =
                generator.generate(
                        lir, ScratchLayout.EMPTY, KernelCacheKey.of("lir_to_mojo_reduce"));

        String source = (String) program.payload();
        assertTrue(source.contains("scalar1: Float32"));
        assertTrue(source.contains("acc ="));
        assertTrue(source.contains("acc_next"));
        assertTrue(source.contains("acc = acc_next"));
        assertTrue(source.contains("for i in range(0, 4, 1):"));
    }

    @Test
    void emitsUnaryTernaryAndCastsInNativeMojo() {
        useTempCache();

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef inputBuf = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP32, 4);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(4));
        LIRExprNode x = graph.scalarLoad(inputBuf, offset, DataType.FP32);
        LIRExprNode neg = graph.scalarUnary(UnaryOperator.NEGATE, x);
        LIRExprNode zero = graph.scalarConst(0L, DataType.FP32);
        LIRExprNode cond = graph.scalarBinary(BinaryOperator.LESS_THAN, x, zero);
        LIRExprNode selected = graph.scalarTernary(cond, neg, x);
        LIRExprNode asFp64 = graph.scalarCast(selected, DataType.FP64);
        LIRExprNode backToFp32 = graph.scalarCast(asFp64, DataType.FP32);
        LIRExprNode store = graph.store(outBuf, offset, backToFp32);
        Block body = graph.block(List.of(store, graph.yield(List.of())));
        LIRExprNode loop =
                graph.structuredFor(
                        "i",
                        graph.indexConst(0),
                        graph.indexConst(4),
                        graph.indexConst(1),
                        List.of(),
                        body);

        KernelProgram program =
                new MojoLirKernelGenerator()
                        .generate(
                                builder.build(loop),
                                ScratchLayout.EMPTY,
                                KernelCacheKey.of("lir_to_mojo_unary_ternary_cast"));

        String source = (String) program.payload();
        assertTrue(source.contains("-"));
        assertTrue(source.contains(" if "));
        assertTrue(source.contains("Float64("));
        assertTrue(source.contains("Float32("));
    }

    @Test
    void emitsBitwiseLogicalAndShiftOpsInNativeMojo() {
        useTempCache();

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef inputBuf = builder.addContiguousInput(DataType.I32, 4);
        ScalarInput shift = builder.addScalarInput(DataType.I32);
        BufferRef outBuf = builder.addContiguousOutput(DataType.I32, 4);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(4));
        LIRExprNode x = graph.scalarLoad(inputBuf, offset, DataType.I32);
        LIRExprNode inv = graph.scalarUnary(UnaryOperator.BITWISE_NOT, x);
        LIRExprNode sh =
                graph.scalarBinary(
                        BinaryOperator.SHIFT_RIGHT_UNSIGNED,
                        inv,
                        graph.scalarInput(shift.id(), shift.dataType()));
        LIRExprNode masked =
                graph.scalarBinary(
                        BinaryOperator.BITWISE_AND, sh, graph.scalarConst(255, DataType.I32));
        LIRExprNode lt =
                graph.scalarBinary(
                        BinaryOperator.LESS_THAN, masked, graph.scalarConst(128, DataType.I32));
        LIRExprNode eq =
                graph.scalarBinary(
                        BinaryOperator.EQUAL, masked, graph.scalarConst(7, DataType.I32));
        LIRExprNode xor = graph.scalarBinary(BinaryOperator.LOGICAL_XOR, lt, eq);
        LIRExprNode result =
                graph.scalarTernary(
                        xor,
                        graph.scalarConst(1, DataType.I32),
                        graph.scalarConst(0, DataType.I32));
        LIRExprNode store = graph.store(outBuf, offset, result);
        Block body = graph.block(List.of(store, graph.yield(List.of())));
        LIRExprNode loop =
                graph.structuredFor(
                        "i",
                        graph.indexConst(0),
                        graph.indexConst(4),
                        graph.indexConst(1),
                        List.of(),
                        body);

        KernelProgram program =
                new MojoLirKernelGenerator()
                        .generate(
                                builder.build(loop),
                                ScratchLayout.EMPTY,
                                KernelCacheKey.of("lir_to_mojo_bitwise_logical"));

        String source = (String) program.payload();
        assertTrue(source.contains("~("));
        assertTrue(source.contains(">>"));
        assertTrue(source.contains("&"));
        assertTrue(source.contains(" != "));
    }

    @Test
    void emitsTypedIntegerAccumulatorForReduction() {
        useTempCache();

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef inputBuf = builder.addContiguousInput(DataType.I32, 6);
        BufferRef outBuf = builder.addContiguousOutput(DataType.I32, 2);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i0 = graph.indexVar("i0");
        LIRExprNode i1 = graph.indexVar("i1");
        LIRExprNode rowBase = graph.indexBinary(IndexBinaryOp.MULTIPLY, i0, graph.indexConst(12));
        LIRExprNode colOff = graph.indexBinary(IndexBinaryOp.MULTIPLY, i1, graph.indexConst(4));
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.ADD, rowBase, colOff);
        LIRExprNode x = graph.scalarLoad(inputBuf, offset, DataType.I32);
        LIRExprNode accRef = graph.scalarRef("acc0", DataType.I32);
        LIRExprNode max = graph.scalarBinary(BinaryOperator.MAX, accRef, x);

        Block innerBody = graph.block(List.of(graph.yield(List.of(max))));
        LIRExprNode innerLoop =
                graph.structuredFor(
                        "i1",
                        graph.indexConst(0),
                        graph.indexConst(3),
                        graph.indexConst(1),
                        List.of(
                                new LoopIterArg(
                                        "acc0",
                                        DataType.I32,
                                        graph.scalarConst(Integer.MIN_VALUE, DataType.I32))),
                        innerBody);

        LIRExprNode outOffset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i0, graph.indexConst(4));
        LIRExprNode store = graph.store(outBuf, outOffset, accRef);
        Block outerBody = graph.block(List.of(innerLoop, store, graph.yield(List.of())));
        LIRExprNode outerLoop =
                graph.structuredFor(
                        "i0",
                        graph.indexConst(0),
                        graph.indexConst(2),
                        graph.indexConst(1),
                        List.of(),
                        outerBody);

        KernelProgram program =
                new MojoLirKernelGenerator()
                        .generate(
                                builder.build(outerLoop),
                                ScratchLayout.EMPTY,
                                KernelCacheKey.of("lir_to_mojo_i32_reduction_typed_acc"));

        String source = (String) program.payload();
        assertTrue(source.contains("acc0 = Int32("));
        assertTrue(source.contains("acc0_next"));
        assertTrue(source.contains("acc0 = acc0_next"));
    }

    @Test
    void emitsFp64TrigViaFp32Workaround() {
        useTempCache();

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef inputBuf = builder.addContiguousInput(DataType.FP64, 2);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP64, 2);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(8));
        LIRExprNode x = graph.scalarLoad(inputBuf, offset, DataType.FP64);
        LIRExprNode c = graph.scalarUnary(UnaryOperator.COS, x);
        LIRExprNode store = graph.store(outBuf, offset, c);
        Block body = graph.block(List.of(store, graph.yield(List.of())));
        LIRExprNode loop =
                graph.structuredFor(
                        "i",
                        graph.indexConst(0),
                        graph.indexConst(2),
                        graph.indexConst(1),
                        List.of(),
                        body);

        KernelProgram program =
                new MojoLirKernelGenerator()
                        .generate(
                                builder.build(loop),
                                ScratchLayout.EMPTY,
                                KernelCacheKey.of("lir_to_mojo_fp64_trig_workaround"));

        String source = (String) program.payload();
        assertTrue(source.contains("Float64(cos(Float32("));
    }

    @Test
    void generatesLegacySyntaxForMojo0261() {
        useTempCache();

        KernelProgram program =
                new MojoLirKernelGenerator(MojoDialect.V0_26_1)
                        .generate(
                                elementwiseAddGraph(),
                                ScratchLayout.EMPTY,
                                KernelCacheKey.of("lir_to_mojo_legacy"));

        String source = (String) program.payload();
        assertTrue(source.contains("from std.gpu import block_idx, block_dim, thread_idx"));
        assertTrue(source.contains("fn hip_lir_"));
        assertTrue(source.contains("var gid ="));
    }

    @Test
    void declaresEveryLocalOnceWithVar() {
        useTempCache();

        KernelProgram program =
                new MojoLirKernelGenerator()
                        .generate(
                                elementwiseAddGraph(),
                                ScratchLayout.EMPTY,
                                KernelCacheKey.of("lir_to_mojo_var_decls"));

        String source = (String) program.payload();
        assertEquals(1, count(source, "var total_0 = 1"));
        assertEquals(1, count(source, "total_0 *= 4"));
        assertEquals(1, count(source, "var temp_2 = linear_1"));
        assertEquals(1, count(source, "temp_2 = temp_2 // 4"));
        assertEquals(1, count(source, "var i = temp_2 % 4"));
    }

    @Test
    void cachedTemporaryIsReemittedOutsideTheLoopThatDeclaredIt() {
        useTempCache();

        LIRGraph.Builder builder = LIRGraph.builder();
        ScalarInput scale = builder.addScalarInput(DataType.FP32);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP32, 1);

        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode in = graph.scalarInput(scale.id(), scale.dataType());
        // Used twice: inside the loop (in the yield) and after it (in the store).
        LIRExprNode squared = graph.scalarBinary(BinaryOperator.MULTIPLY, in, in);
        LIRExprNode accRef = graph.scalarRef("acc", DataType.FP32);
        LIRExprNode sum = graph.scalarBinary(BinaryOperator.ADD, accRef, squared);
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
                        graph.block(List.of(graph.yield(List.of(sum)))));
        LIRExprNode result = graph.scalarBinary(BinaryOperator.ADD, accRef, squared);
        LIRExprNode store = graph.store(outBuf, graph.indexConst(0), result);
        LIRGraph lir = builder.build(graph.block(List.of(loop, store)));

        String source =
                (String)
                        new MojoLirKernelGenerator()
                                .generate(
                                        lir,
                                        ScratchLayout.EMPTY,
                                        KernelCacheKey.of("lir_to_mojo_scoped_temp"))
                                .payload();

        // The graph shares one node for "acc + scalar0 * scalar0" between the yield and the
        // store. The temporary holding it inside the loop is out of scope after the loop (and
        // would hold the value from before the last update), so it must be computed again.
        assertEquals(2, count(source, "= ((scalar0 * scalar0) + acc)"), source);
        assertEquals(1, count(source, "var acc = "), source);
        assertTrue(source.contains("var t2 = ((scalar0 * scalar0) + acc)"), source);
        assertTrue(source.contains("output0[Int((0) // 4)] = t2"), source);
    }

    @Test
    void scratchBuffersUseUnsafeFromAddressConstructor() {
        useTempCache();

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef inputBuf = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef tmp = BufferRef.contiguous(99, DataType.FP32, 4);
        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(4));
        LIRExprNode first =
                graph.structuredFor(
                        "i",
                        graph.indexConst(0),
                        graph.indexConst(4),
                        graph.indexConst(1),
                        List.of(),
                        graph.block(
                                List.of(
                                        graph.store(
                                                tmp,
                                                offset,
                                                graph.scalarLoad(inputBuf, offset, DataType.FP32)),
                                        graph.yield(List.of()))));
        LIRExprNode j = graph.indexVar("j");
        LIRExprNode offset2 = graph.indexBinary(IndexBinaryOp.MULTIPLY, j, graph.indexConst(4));
        LIRExprNode second =
                graph.structuredFor(
                        "j",
                        graph.indexConst(0),
                        graph.indexConst(4),
                        graph.indexConst(1),
                        List.of(),
                        graph.block(
                                List.of(
                                        graph.store(
                                                outBuf,
                                                offset2,
                                                graph.scalarLoad(tmp, offset2, DataType.FP32)),
                                        graph.yield(List.of()))));
        LIRGraph lir = builder.build(graph.block(List.of(first, second)));

        String source =
                (String)
                        new MojoLirKernelGenerator()
                                .generate(
                                        lir,
                                        new ScratchLayout(java.util.Map.of(tmp, 64L), 80L),
                                        KernelCacheKey.of("lir_to_mojo_scratch"))
                                .payload();

        assertTrue(
                source.contains(
                        "var scratch0 = UnsafePointer[Float32, MutAnyOrigin]"
                                + "(unsafe_from_address=Int(scratch_ptr + UInt64(64)))"),
                source);
        assertTrue(!source.contains("from_address("), source);
    }

    @Test
    void literalsAndShiftAmountsCarryTheOperandType() {
        useTempCache();

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef intBuf = builder.addContiguousInput(DataType.I64, 4);
        BufferRef halfBuf = builder.addContiguousInput(DataType.FP16, 4);
        BufferRef outInt = builder.addContiguousOutput(DataType.I64, 4);
        BufferRef outHalf = builder.addContiguousOutput(DataType.FP16, 4);
        BufferRef outFlag = builder.addContiguousOutput(DataType.FP32, 4);
        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        LIRExprNode off8 = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(8));
        LIRExprNode off2 = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(2));
        LIRExprNode off4 = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(4));
        LIRExprNode v = graph.scalarLoad(intBuf, off8, DataType.I64);
        LIRExprNode shifted =
                graph.scalarBinary(
                        BinaryOperator.SHIFT_RIGHT_UNSIGNED,
                        graph.scalarBinary(BinaryOperator.SHIFT_LEFT, v, v),
                        v);
        LIRExprNode h = graph.scalarLoad(halfBuf, off2, DataType.FP16);
        LIRExprNode math =
                graph.scalarUnary(
                        UnaryOperator.LOG,
                        graph.scalarUnary(
                                UnaryOperator.EXP, graph.scalarUnary(UnaryOperator.RECIPROCAL, h)));
        LIRExprNode flag =
                graph.scalarCast(graph.scalarBinary(BinaryOperator.LESS_THAN, h, h), DataType.FP32);
        Block body =
                graph.block(
                        List.of(
                                graph.store(outInt, off8, shifted),
                                graph.store(outHalf, off2, math),
                                graph.store(outFlag, off4, flag),
                                graph.yield(List.of())));
        LIRGraph lir =
                builder.build(
                        graph.structuredFor(
                                "i",
                                graph.indexConst(0),
                                graph.indexConst(4),
                                graph.indexConst(1),
                                List.of(),
                                body));

        String source =
                (String)
                        new MojoLirKernelGenerator()
                                .generate(
                                        lir,
                                        ScratchLayout.EMPTY,
                                        KernelCacheKey.of("lir_to_mojo_typed_literals"))
                                .payload();

        assertTrue(source.contains("<< Int64(Int("), source);
        assertTrue(source.contains(">> UInt64(Int("), source);
        assertTrue(source.contains("(Float16(1.0) / "), source);
        assertTrue(source.contains("(Float16(2.718281828459045) ** "), source);
        assertTrue(source.contains(" * Float16(0.6931471805599453))"), source);
        assertTrue(source.contains("Float32(Int(("), source);
    }

    private static LIRGraph elementwiseAddGraph() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef leftBuf = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef outBuf = builder.addContiguousOutput(DataType.FP32, 4);
        LIRExprGraph graph = builder.exprGraph();
        LIRExprNode i = graph.indexVar("i");
        LIRExprNode offset = graph.indexBinary(IndexBinaryOp.MULTIPLY, i, graph.indexConst(4));
        LIRExprNode value = graph.scalarLoad(leftBuf, offset, DataType.FP32);
        LIRExprNode sum = graph.scalarBinary(BinaryOperator.ADD, value, value);
        Block body = graph.block(List.of(graph.store(outBuf, offset, sum), graph.yield(List.of())));
        return builder.build(
                graph.structuredFor(
                        "i",
                        graph.indexConst(0),
                        graph.indexConst(4),
                        graph.indexConst(1),
                        List.of(),
                        body));
    }

    private static int count(String text, String needle) {
        int n = 0;
        for (int at = text.indexOf(needle); at >= 0; at = text.indexOf(needle, at + 1)) {
            n++;
        }
        return n;
    }

    private void useTempCache() {
        System.setProperty(KernelCachePaths.CACHE_ROOT_PROPERTY, tempDir.toString());
        System.setProperty(KernelCachePaths.VERSION_PROPERTY, "test");
    }
}
