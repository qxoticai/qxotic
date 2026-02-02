package ai.qxotic.jota.ir.lir.scratch;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.*;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import java.util.List;
import org.junit.jupiter.api.Test;

class ScratchVerificationPassTest {

    private final ScratchAnalysisPass analysisPass = new ScratchAnalysisPass();
    private final ScratchVerificationPass verificationPass = new ScratchVerificationPass();

    @Test
    void testValidScratchLayout_passesVerification() {
        // Build graph with one intermediate
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef intermediate = BufferRef.contiguous(100, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad loadIn = new ScalarLoad(in, byteOffset);
        ScalarExpr doubled = new ScalarBinary(BinaryOperator.MULTIPLY, loadIn, loadIn);
        Store storeIntermediate = new Store(intermediate, byteOffset, doubled);

        ScalarLoad loadIntermediate = new ScalarLoad(intermediate, byteOffset);
        Store storeOut = new Store(out, byteOffset, loadIntermediate);

        Block loopBody = new Block(List.of(storeIntermediate, storeOut));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        ScratchLayout layout = analysisPass.analyze(graph);
        ScratchVerificationPass.VerificationResult result = verificationPass.verify(graph, layout);

        assertTrue(result.isValid(), "Valid layout should pass verification");
        assertTrue(result.errors().isEmpty(), "Should have no errors");
    }

    @Test
    void testEmptyLayout_noScratchNeeded_passesVerification() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad load = new ScalarLoad(in, byteOffset);
        Store store = new Store(out, byteOffset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        ScratchLayout layout = analysisPass.analyze(graph);
        ScratchVerificationPass.VerificationResult result = verificationPass.verify(graph, layout);

        assertTrue(result.isValid(), "Empty layout should pass verification");
        assertSame(ScratchLayout.EMPTY, layout);
    }

    @Test
    void testMissingIntermediateInLayout_failsVerification() {
        // Create a layout manually that doesn't include an intermediate
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef intermediate = BufferRef.contiguous(100, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad loadIn = new ScalarLoad(in, byteOffset);
        Store storeIntermediate = new Store(intermediate, byteOffset, loadIn);
        Store storeOut = new Store(out, byteOffset, new ScalarLoad(intermediate, byteOffset));

        Block loopBody = new Block(List.of(storeIntermediate, storeOut));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        // Create empty layout (simulating a bug where intermediate is not tracked)
        ScratchLayout emptyLayout = ScratchLayout.EMPTY;

        ScratchVerificationPass.VerificationResult result =
                verificationPass.verify(graph, emptyLayout);

        assertFalse(result.isValid(), "Should fail when intermediate not in layout");
        assertEquals(1, result.errors().size());
        assertTrue(
                result.errors().get(0).contains("not tracked in scratch layout"),
                "Error should mention missing intermediate");
    }

    @Test
    void testInputMarkedAsScratch_failsVerification() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad load = new ScalarLoad(in, byteOffset);
        Store store = new Store(out, byteOffset, load);
        Loop loop = Loop.parallel("i", 4, store);

        LIRGraph graph = builder.build(loop);

        // Create layout that incorrectly marks input as scratch
        java.util.Map<BufferRef, Long> badOffsets = new java.util.HashMap<>();
        badOffsets.put(in, 0L);
        ScratchLayout badLayout = new ScratchLayout(badOffsets, 64L);

        ScratchVerificationPass.VerificationResult result =
                verificationPass.verify(graph, badLayout);

        assertFalse(result.isValid(), "Should fail when input marked as scratch");
        assertTrue(
                result.errors().get(0).contains("Input buffer incorrectly marked as scratch"),
                "Error should mention input buffer");
    }

    @Test
    void testBufferOffsetOutOfBounds_failsVerification() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef intermediate = BufferRef.contiguous(100, DataType.FP32, 4); // 16 bytes

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad loadIn = new ScalarLoad(in, byteOffset);
        Store storeIntermediate = new Store(intermediate, byteOffset, loadIn);
        Store storeOut = new Store(out, byteOffset, new ScalarLoad(intermediate, byteOffset));

        Block loopBody = new Block(List.of(storeIntermediate, storeOut));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        // Create layout with offset that puts buffer out of bounds
        java.util.Map<BufferRef, Long> badOffsets = new java.util.HashMap<>();
        badOffsets.put(intermediate, 100L); // offset 100, size 16, total 16 -> out of bounds
        ScratchLayout badLayout = new ScratchLayout(badOffsets, 16L);

        ScratchVerificationPass.VerificationResult result =
                verificationPass.verify(graph, badLayout);

        assertFalse(result.isValid(), "Should fail when buffer extends beyond total size");
        assertTrue(
                result.errors().get(0).contains("extends beyond total size"),
                "Error should mention out of bounds");
    }

    @Test
    void testNegativeOffset_failsVerification() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef intermediate = BufferRef.contiguous(100, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad loadIn = new ScalarLoad(in, byteOffset);
        Store storeIntermediate = new Store(intermediate, byteOffset, loadIn);
        Store storeOut = new Store(out, byteOffset, new ScalarLoad(intermediate, byteOffset));

        Block loopBody = new Block(List.of(storeIntermediate, storeOut));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        // Create layout with negative offset
        java.util.Map<BufferRef, Long> badOffsets = new java.util.HashMap<>();
        badOffsets.put(intermediate, -64L);
        ScratchLayout badLayout = new ScratchLayout(badOffsets, 64L);

        ScratchVerificationPass.VerificationResult result =
                verificationPass.verify(graph, badLayout);

        assertFalse(result.isValid(), "Should fail with negative offset");
        assertTrue(
                result.errors().get(0).contains("Negative offset"),
                "Error should mention negative offset");
    }

    @Test
    void testUnalignedOffset_failsVerification() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef intermediate = BufferRef.contiguous(100, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad loadIn = new ScalarLoad(in, byteOffset);
        Store storeIntermediate = new Store(intermediate, byteOffset, loadIn);
        Store storeOut = new Store(out, byteOffset, new ScalarLoad(intermediate, byteOffset));

        Block loopBody = new Block(List.of(storeIntermediate, storeOut));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        // Create layout with unaligned offset (not multiple of 64)
        java.util.Map<BufferRef, Long> badOffsets = new java.util.HashMap<>();
        badOffsets.put(intermediate, 32L); // 32 is not aligned to 64
        ScratchLayout badLayout = new ScratchLayout(badOffsets, 128L);

        ScratchVerificationPass.VerificationResult result =
                verificationPass.verify(graph, badLayout);

        assertFalse(result.isValid(), "Should fail with unaligned offset");
        assertTrue(
                result.errors().get(0).contains("Unaligned"),
                "Error should mention unaligned offset");
    }

    @Test
    void testOverlappingLifetimesWithOverlappingMemory_failsVerification() {
        // Build graph where two intermediates have overlapping lifetimes
        // and would need non-overlapping memory
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out1 = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef out2 = builder.addContiguousOutput(DataType.FP32, 4);

        BufferRef temp1 = BufferRef.contiguous(100, DataType.FP32, 4);
        BufferRef temp2 = BufferRef.contiguous(101, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        // stmt 0: store temp1 (temp1 live)
        ScalarLoad loadIn1 = new ScalarLoad(in, byteOffset);
        Store storeTemp1 = new Store(temp1, byteOffset, loadIn1);

        // stmt 1: store temp2 (temp1 still live, temp2 live)
        ScalarLoad loadIn2 = new ScalarLoad(in, byteOffset);
        Store storeTemp2 = new Store(temp2, byteOffset, loadIn2);

        // stmt 2: read temp1 (temp1 dead after)
        ScalarLoad loadTemp1 = new ScalarLoad(temp1, byteOffset);
        Store storeOut1 = new Store(out1, byteOffset, loadTemp1);

        // stmt 3: read temp2
        ScalarLoad loadTemp2 = new ScalarLoad(temp2, byteOffset);
        Store storeOut2 = new Store(out2, byteOffset, loadTemp2);

        Block loopBody = new Block(List.of(storeTemp1, storeTemp2, storeOut1, storeOut2));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        // Create layout with overlapping memory (both at offset 0)
        java.util.Map<BufferRef, Long> badOffsets = new java.util.HashMap<>();
        badOffsets.put(temp1, 0L);
        badOffsets.put(temp2, 0L); // Same offset - overlapping!
        ScratchLayout badLayout = new ScratchLayout(badOffsets, 64L);

        ScratchVerificationPass.VerificationResult result =
                verificationPass.verify(graph, badLayout);

        assertFalse(
                result.isValid(), "Should fail with overlapping memory for overlapping lifetimes");
        assertTrue(
                result.errors().get(0).contains("Overlapping scratch allocations"),
                "Error should mention overlapping allocations");
    }

    @Test
    void testNonOverlappingLifetimesWithOverlappingMemory_passesVerification() {
        // Build graph where two intermediates have non-overlapping lifetimes
        // so they can share memory
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out1 = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef out2 = builder.addContiguousOutput(DataType.FP32, 4);

        BufferRef temp1 = BufferRef.contiguous(100, DataType.FP32, 4);
        BufferRef temp2 = BufferRef.contiguous(101, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        // stmt 0: store temp1 (temp1 live)
        ScalarLoad loadIn1 = new ScalarLoad(in, byteOffset);
        Store storeTemp1 = new Store(temp1, byteOffset, loadIn1);

        // stmt 1: read temp1 (temp1 dead after)
        ScalarLoad loadTemp1 = new ScalarLoad(temp1, byteOffset);
        Store storeOut1 = new Store(out1, byteOffset, loadTemp1);

        // stmt 2: store temp2 (temp2 live, temp1 is dead so can reuse memory)
        ScalarLoad loadIn2 = new ScalarLoad(in, byteOffset);
        Store storeTemp2 = new Store(temp2, byteOffset, loadIn2);

        // stmt 3: read temp2
        ScalarLoad loadTemp2 = new ScalarLoad(temp2, byteOffset);
        Store storeOut2 = new Store(out2, byteOffset, loadTemp2);

        Block loopBody = new Block(List.of(storeTemp1, storeOut1, storeTemp2, storeOut2));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        // Create layout with overlapping memory (both at offset 0) - this is OK for non-overlapping
        // lifetimes
        java.util.Map<BufferRef, Long> offsets = new java.util.HashMap<>();
        offsets.put(temp1, 0L);
        offsets.put(temp2, 0L); // Same offset - OK because lifetimes don't overlap
        ScratchLayout layout = new ScratchLayout(offsets, 64L);

        ScratchVerificationPass.VerificationResult result = verificationPass.verify(graph, layout);

        assertTrue(
                result.isValid(),
                "Should pass with overlapping memory for non-overlapping lifetimes");
    }

    @Test
    void testVerifyOrThrow_throwsOnInvalid() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef intermediate = BufferRef.contiguous(100, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        ScalarLoad loadIn = new ScalarLoad(in, byteOffset);
        Store storeIntermediate = new Store(intermediate, byteOffset, loadIn);
        Store storeOut = new Store(out, byteOffset, new ScalarLoad(intermediate, byteOffset));

        Block loopBody = new Block(List.of(storeIntermediate, storeOut));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        // Use empty layout to trigger failure
        assertThrows(
                IllegalStateException.class,
                () -> verificationPass.verifyOrThrow(graph, ScratchLayout.EMPTY),
                "verifyOrThrow should throw on invalid layout");
    }

    @Test
    void testWarningsForUnusedBuffer() {
        LIRGraph.Builder builder = LIRGraph.builder();
        BufferRef in = builder.addContiguousInput(DataType.FP32, 4);
        BufferRef out = builder.addContiguousOutput(DataType.FP32, 4);
        BufferRef unused = BufferRef.contiguous(100, DataType.FP32, 4);
        BufferRef used = BufferRef.contiguous(101, DataType.FP32, 4);

        IndexVar i = new IndexVar("i");
        IndexExpr byteOffset = IndexBinary.multiply(i, new IndexConst(4));

        // Store to both intermediates
        ScalarLoad loadIn = new ScalarLoad(in, byteOffset);
        Store storeUnused = new Store(unused, byteOffset, loadIn);
        Store storeUsed = new Store(used, byteOffset, loadIn);

        // But only read from 'used'
        ScalarLoad loadUsed = new ScalarLoad(used, byteOffset);
        Store storeOut = new Store(out, byteOffset, loadUsed);

        Block loopBody = new Block(List.of(storeUnused, storeUsed, storeOut));
        Loop loop = Loop.parallel("i", 4, loopBody);

        LIRGraph graph = builder.build(loop);

        ScratchLayout layout = analysisPass.analyze(graph);
        ScratchVerificationPass.VerificationResult result = verificationPass.verify(graph, layout);

        // Should be valid but have warning
        assertTrue(result.isValid(), "Layout should be valid");
        assertFalse(result.warnings().isEmpty(), "Should have warnings for unused buffer");
        assertTrue(
                result.warnings().stream().anyMatch(w -> w.contains("Unused scratch buffer")),
                "Warning should mention unused buffer: " + result.warnings());
    }
}
