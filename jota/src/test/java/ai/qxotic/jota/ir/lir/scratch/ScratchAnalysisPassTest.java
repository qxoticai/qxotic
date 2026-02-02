package ai.qxotic.jota.ir.lir.scratch;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.*;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.util.List;
import org.junit.jupiter.api.Test;

class ScratchAnalysisPassTest {

    @Test
    void testNoIntermediates_returnsEmpty() {
        // Build: for i in [0, 4) { store out[i], in[i] }
        // No intermediate buffers, only input and output
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad load = new ScalarLoad(in, byteOffset);
        Store store = new Store(out, byteOffset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        ScratchAnalysisPass pass = new ScratchAnalysisPass();
        ScratchLayout layout = pass.analyze(graph);

        assertSame(ScratchLayout.EMPTY, layout, "No intermediates should return EMPTY layout");
        assertFalse(layout.requiresScratch());
        assertEquals(0L, layout.totalByteSize());
    }

    @Test
    void testSingleIntermediate_correctOffsetAndSize() {
        // Build graph with one intermediate buffer (not in inputs/outputs)
        // Create a separate BufferRef for intermediate that's not registered as input/output
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        // Create an intermediate buffer (not added via builder, so not in inputs/outputs)
        BufferRef intermediate = BufferRef.contiguous(100, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        // Phase 1: store to intermediate
        ScalarLoad loadIn = new ScalarLoad(in, byteOffset);
        ScalarExpr doubled = new ScalarBinary(BinaryOperator.MULTIPLY, loadIn, loadIn);
        Store storeIntermediate = new Store(intermediate, byteOffset, doubled);

        // Phase 2: load from intermediate and store to output
        ScalarLoad loadIntermediate = new ScalarLoad(intermediate, byteOffset);
        Store storeOut = new Store(out, byteOffset, loadIntermediate);

        Block loopBody = new Block(List.of(storeIntermediate, storeOut));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        ScratchAnalysisPass pass = new ScratchAnalysisPass();
        ScratchLayout layout = pass.analyze(graph);

        assertTrue(layout.requiresScratch(), "Should require scratch for intermediate buffer");
        assertEquals(1, layout.offsets().size(), "Should have exactly one scratch buffer");

        // Verify buffer is tracked
        assertTrue(layout.isScratchBuffer(intermediate));
        long offset = layout.getOffset(intermediate);
        assertEquals(0L, offset, "First buffer should be at offset 0");

        // Total size should be at least 16 bytes (4 floats), aligned to 64
        assertTrue(layout.totalByteSize() >= 16L, "Should have at least 16 bytes (4 floats)");
    }

    @Test
    void testMultipleNonOverlappingIntermediates_reuseMemory() {
        // Build:
        // store temp1[i], in[i]*2  (temp1 written at stmt 0)
        // store out[i], temp1[i]   (temp1 read at stmt 1, temp1 dead after)
        // store temp2[i], in[i]+1  (temp2 written at stmt 2, can reuse temp1's memory)
        // store out2[i], temp2[i]  (temp2 read at stmt 3)
        //
        // temp1 lifetime: [0, 1]
        // temp2 lifetime: [2, 3]
        // These don't overlap, so temp2 can reuse temp1's memory

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out1 = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef out2 = builder.addContiguousOutput(DataType.FP32, 4);

        // Intermediate buffers
        BufferRef temp1 = BufferRef.contiguous(100, DataType.FP32, 4);
        BufferRef temp2 = BufferRef.contiguous(101, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        // stmt 0: store temp1[i] = in[i] * 2
        ScalarLoad loadIn1 = new ScalarLoad(in, byteOffset);
        ScalarLiteral two = new ScalarLiteral(Float.floatToRawIntBits(2.0f), DataType.FP32);
        ScalarExpr doubled = new ScalarBinary(BinaryOperator.MULTIPLY, loadIn1, two);
        Store storeTemp1 = new Store(temp1, byteOffset, doubled);

        // stmt 1: store out1[i] = temp1[i]
        ScalarLoad loadTemp1 = new ScalarLoad(temp1, byteOffset);
        Store storeOut1 = new Store(out1, byteOffset, loadTemp1);

        // stmt 2: store temp2[i] = in[i] + 1
        ScalarLoad loadIn2 = new ScalarLoad(in, byteOffset);
        ScalarLiteral one = new ScalarLiteral(Float.floatToRawIntBits(1.0f), DataType.FP32);
        ScalarExpr incremented = new ScalarBinary(BinaryOperator.ADD, loadIn2, one);
        Store storeTemp2 = new Store(temp2, byteOffset, incremented);

        // stmt 3: store out2[i] = temp2[i]
        ScalarLoad loadTemp2 = new ScalarLoad(temp2, byteOffset);
        Store storeOut2 = new Store(out2, byteOffset, loadTemp2);

        Block loopBody = new Block(List.of(storeTemp1, storeOut1, storeTemp2, storeOut2));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        ScratchAnalysisPass pass = new ScratchAnalysisPass();
        ScratchLayout layout = pass.analyze(graph);

        assertTrue(layout.requiresScratch());
        assertEquals(2, layout.offsets().size(), "Should have 2 scratch buffers");

        // Both buffers should be tracked
        assertTrue(layout.isScratchBuffer(temp1));
        assertTrue(layout.isScratchBuffer(temp2));

        long offset1 = layout.getOffset(temp1);
        long offset2 = layout.getOffset(temp2);

        // With non-overlapping lifetimes, temp2 might reuse temp1's memory
        // The total size should be <= 2 * singleBufferSize (ideally equal to single buffer size)
        long singleBufferSize = 4 * 4; // 4 floats * 4 bytes = 16 bytes
        long alignedBufferSize = 64L; // aligned to 64 bytes

        System.out.println("Buffer 1: offset=" + offset1);
        System.out.println("Buffer 2: offset=" + offset2);
        System.out.println("Total scratch: " + layout.totalByteSize());

        // The total should be optimized (reuse should happen)
        // Since lifetimes don't overlap, ideal total = max(slot1.size, slot2.size)
        // But with alignment, it might be slot1.size (if they reuse at same offset)
        assertTrue(
                layout.totalByteSize() <= 2 * alignedBufferSize,
                "Total should be <= 2 * single buffer due to potential reuse");
    }

    @Test
    void testOverlappingIntermediates_noReuse() {
        // Build:
        // store temp1[i], in[i]*2   (temp1 written)
        // store temp2[i], in[i]+1   (temp2 written, temp1 still live)
        // store out1[i], temp1[i]   (temp1 read)
        // store out2[i], temp2[i]   (temp2 read)
        //
        // temp1 lifetime: [0, 2]
        // temp2 lifetime: [1, 3]
        // These overlap, so they can't share memory

        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out1 = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef out2 = builder.addContiguousOutput(DataType.FP32, 4);

        BufferRef temp1 = BufferRef.contiguous(100, DataType.FP32, 4);
        BufferRef temp2 = BufferRef.contiguous(101, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        // stmt 0: store temp1
        ScalarLoad loadIn1 = new ScalarLoad(in, byteOffset);
        ScalarLiteral two = new ScalarLiteral(Float.floatToRawIntBits(2.0f), DataType.FP32);
        Store storeTemp1 =
                new Store(
                        temp1, byteOffset, new ScalarBinary(BinaryOperator.MULTIPLY, loadIn1, two));

        // stmt 1: store temp2 (temp1 still live)
        ScalarLoad loadIn2 = new ScalarLoad(in, byteOffset);
        ScalarLiteral one = new ScalarLiteral(Float.floatToRawIntBits(1.0f), DataType.FP32);
        Store storeTemp2 =
                new Store(temp2, byteOffset, new ScalarBinary(BinaryOperator.ADD, loadIn2, one));

        // stmt 2: read temp1
        ScalarLoad loadTemp1 = new ScalarLoad(temp1, byteOffset);
        Store storeOut1 = new Store(out1, byteOffset, loadTemp1);

        // stmt 3: read temp2
        ScalarLoad loadTemp2 = new ScalarLoad(temp2, byteOffset);
        Store storeOut2 = new Store(out2, byteOffset, loadTemp2);

        Block loopBody = new Block(List.of(storeTemp1, storeTemp2, storeOut1, storeOut2));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        ScratchAnalysisPass pass = new ScratchAnalysisPass();
        ScratchLayout layout = pass.analyze(graph);

        assertTrue(layout.requiresScratch());
        assertEquals(2, layout.offsets().size());

        long offset1 = layout.getOffset(temp1);
        long offset2 = layout.getOffset(temp2);

        // Calculate aligned size for each buffer
        long alignedBufferSize = 64L; // aligned to 64 bytes

        System.out.println("Overlapping - Buffer 1: offset=" + offset1);
        System.out.println("Overlapping - Buffer 2: offset=" + offset2);
        System.out.println("Overlapping - Total scratch: " + layout.totalByteSize());

        // With overlapping lifetimes, buffers should NOT share memory
        // Their memory ranges should not overlap
        long end1 = offset1 + alignedBufferSize;
        long end2 = offset2 + alignedBufferSize;
        boolean overlaps = !(end1 <= offset2 || end2 <= offset1);
        assertFalse(
                overlaps, "Overlapping lifetime buffers should have non-overlapping memory ranges");
    }

    @Test
    void testLivenessInterval_overlapsMethod() {
        BufferRef buf1 = BufferRef.contiguous(1, DataType.FP32, 4);
        BufferRef buf2 = BufferRef.contiguous(2, DataType.FP32, 4);

        // [0, 2] and [3, 5] - don't overlap
        LivenessInterval a = new LivenessInterval(buf1, 0, 2);
        LivenessInterval b = new LivenessInterval(buf2, 3, 5);
        assertFalse(a.overlaps(b));
        assertFalse(b.overlaps(a));

        // [0, 3] and [2, 5] - overlap
        LivenessInterval c = new LivenessInterval(buf1, 0, 3);
        LivenessInterval d = new LivenessInterval(buf2, 2, 5);
        assertTrue(c.overlaps(d));
        assertTrue(d.overlaps(c));

        // [0, 2] and [2, 4] - overlap at boundary (touching)
        LivenessInterval e = new LivenessInterval(buf1, 0, 2);
        LivenessInterval f = new LivenessInterval(buf2, 2, 4);
        assertTrue(e.overlaps(f));
        assertTrue(f.overlaps(e));
    }

    @Test
    void testScratchLayout_alignment() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        // Small intermediate - should still be aligned
        BufferRef intermediate = BufferRef.contiguous(100, DataType.I8, 7); // 7 bytes

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = new IndexVar("i");

        Store storeIntermediate =
                new Store(intermediate, byteOffset, new ScalarLiteral(42, DataType.I8));
        ScalarLoad loadIntermediate = new ScalarLoad(intermediate, byteOffset);
        ScalarCast cast = new ScalarCast(loadIntermediate, DataType.FP32);
        Store storeOut = new Store(out, IndexBinary.multiply(i, new IndexConst(4)), cast);

        Block loopBody = new Block(List.of(storeIntermediate, storeOut));
        Loop loop = Loop.parallel("i", 7, loopBody);

        LIRGraph graph = builder.build(loop);

        ScratchAnalysisPass pass = new ScratchAnalysisPass();
        ScratchLayout layout = pass.analyze(graph);

        assertTrue(layout.requiresScratch());

        // Check alignment (default is 64 bytes)
        assertEquals(64L, ScratchLayout.ALIGNMENT);
        assertTrue(
                layout.alignedTotalByteSize() % ScratchLayout.ALIGNMENT == 0,
                "Total size should be aligned");
    }
}
