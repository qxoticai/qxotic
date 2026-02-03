package ai.qxotic.jota.ir.lir.scratch;

import ai.qxotic.jota.ir.lir.*;
import java.util.*;

/**
 * Verification pass for scratch memory usage in LIR graphs.
 *
 * <p>This pass validates that scratch memory is used correctly:
 *
 * <ul>
 *   <li>All intermediate buffers (not inputs/outputs) are tracked in the scratch layout
 *   <li>No buffer is both a scratch buffer and an input/output buffer
 *   <li>All scratch buffer offsets are within the total scratch size
 *   <li>No overlapping scratch allocations exist for buffers with overlapping lifetimes
 *   <li>All scratch buffers are accessed (no unused scratch allocations)
 * </ul>
 *
 * <p>This pass should run after scratch analysis to verify the layout is correct before kernel
 * compilation.
 */
public final class ScratchVerificationPass {

    /** Result of scratch verification. */
    public record VerificationResult(boolean valid, List<String> errors, List<String> warnings) {
        public VerificationResult {
            errors = List.copyOf(errors);
            warnings = List.copyOf(warnings);
        }

        /** Returns true if verification passed with no errors. */
        public boolean isValid() {
            return valid && errors.isEmpty();
        }

        /** Throws IllegalStateException if verification failed. */
        public void throwIfInvalid() {
            if (!isValid()) {
                throw new IllegalStateException(
                        "Scratch verification failed:\n" + String.join("\n", errors));
            }
        }
    }

    /**
     * Verifies scratch memory usage for the given graph and layout.
     *
     * @param graph the LIR graph
     * @param layout the scratch layout
     * @return verification result
     */
    public VerificationResult verify(LIRGraph graph, ScratchLayout layout) {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        // Collect input/output buffers
        Set<BufferRef> inputBuffers = new HashSet<>();
        for (LIRInput input : graph.inputs()) {
            if (input instanceof BufferRef buf) {
                inputBuffers.add(buf);
            }
        }
        Set<BufferRef> outputBuffers = new HashSet<>(graph.outputs());

        // Find all buffers used in the body
        Set<BufferRef> allBuffers = new HashSet<>();
        collectBuffers(graph.body(), allBuffers);

        // Identify intermediates
        Set<BufferRef> intermediates = new HashSet<>(allBuffers);
        intermediates.removeAll(inputBuffers);
        intermediates.removeAll(outputBuffers);

        // Verify 1: All intermediates are tracked in scratch layout
        for (BufferRef intermediate : intermediates) {
            if (!layout.isScratchBuffer(intermediate)) {
                errors.add(
                        "Intermediate buffer not tracked in scratch layout: "
                                + bufferDescription(intermediate));
            }
        }

        // Verify 2: No input/output buffer is also a scratch buffer
        for (BufferRef input : inputBuffers) {
            if (layout.isScratchBuffer(input)) {
                errors.add(
                        "Input buffer incorrectly marked as scratch: " + bufferDescription(input));
            }
        }
        for (BufferRef output : outputBuffers) {
            if (layout.isScratchBuffer(output)) {
                errors.add(
                        "Output buffer incorrectly marked as scratch: "
                                + bufferDescription(output));
            }
        }

        // Verify 3: All scratch buffer offsets are within bounds
        if (layout.requiresScratch()) {
            for (Map.Entry<BufferRef, Long> entry : layout.offsets().entrySet()) {
                BufferRef buffer = entry.getKey();
                long offset = entry.getValue();
                long size = buffer.dataType().byteSizeFor(buffer.shape());
                long end = offset + size;

                if (offset < 0) {
                    errors.add(
                            "Negative offset for scratch buffer: "
                                    + bufferDescription(buffer)
                                    + " offset="
                                    + offset);
                }

                if (end > layout.totalByteSize()) {
                    errors.add(
                            "Scratch buffer extends beyond total size: "
                                    + bufferDescription(buffer)
                                    + " offset="
                                    + offset
                                    + " size="
                                    + size
                                    + " end="
                                    + end
                                    + " total="
                                    + layout.totalByteSize());
                }
            }
        }

        // Verify 4: Check for overlapping allocations (only if liveness info available)
        if (layout.requiresScratch() && intermediates.size() > 1) {
            checkOverlappingAllocations(graph, layout, intermediates, errors);
        }

        // Verify 5: All scratch buffers are read (stores without reads are unused)
        Set<BufferRef> readBuffers = new HashSet<>();
        collectReadBuffers(graph.body(), readBuffers);
        for (BufferRef intermediate : intermediates) {
            if (!readBuffers.contains(intermediate)) {
                warnings.add(
                        "Unused scratch buffer (no reads): " + bufferDescription(intermediate));
            }
        }

        // Verify 6: Check alignment
        if (layout.requiresScratch()) {
            for (Map.Entry<BufferRef, Long> entry : layout.offsets().entrySet()) {
                long offset = entry.getValue();
                if (offset % ScratchLayout.ALIGNMENT != 0) {
                    errors.add(
                            "Unaligned scratch buffer offset: "
                                    + bufferDescription(entry.getKey())
                                    + " offset="
                                    + offset
                                    + " (not aligned to "
                                    + ScratchLayout.ALIGNMENT
                                    + ")");
                }
            }

            if (layout.totalByteSize() % ScratchLayout.ALIGNMENT != 0) {
                warnings.add(
                        "Total scratch size not aligned: "
                                + layout.totalByteSize()
                                + " (should be multiple of "
                                + ScratchLayout.ALIGNMENT
                                + ")");
            }
        }

        boolean valid = errors.isEmpty();
        return new VerificationResult(valid, errors, warnings);
    }

    /**
     * Verifies scratch memory and throws if invalid. Convenience method for use in pipelines.
     *
     * @param graph the LIR graph
     * @param layout the scratch layout
     * @throws IllegalStateException if verification fails
     */
    public void verifyOrThrow(LIRGraph graph, ScratchLayout layout) {
        verify(graph, layout).throwIfInvalid();
    }

    private void collectBuffers(LIRNode node, Set<BufferRef> buffers) {
        switch (node) {
            case Store store -> {
                buffers.add(store.buffer());
                collectBuffersFromScalar(store.value(), buffers);
            }
            case ScalarLet let -> collectBuffersFromScalar(let.value(), buffers);
            case StructuredFor sfor -> {
                for (LoopIterArg arg : sfor.iterArgs()) {
                    collectBuffersFromScalar(arg.init(), buffers);
                }
                collectBuffers(sfor.body(), buffers);
            }
            case TiledLoop tiled -> collectBuffers(tiled.body(), buffers);
            case Block block -> block.statements().forEach(s -> collectBuffers(s, buffers));
            case Yield yield -> yield.values().forEach(v -> collectBuffersFromScalar(v, buffers));
            default -> {}
        }
    }

    private void collectBuffersFromScalar(ScalarExpr expr, Set<BufferRef> buffers) {
        switch (expr) {
            case ScalarLoad load -> buffers.add(load.buffer());
            case ScalarUnary u -> collectBuffersFromScalar(u.input(), buffers);
            case ScalarBinary b -> {
                collectBuffersFromScalar(b.left(), buffers);
                collectBuffersFromScalar(b.right(), buffers);
            }
            case ScalarTernary t -> {
                collectBuffersFromScalar(t.condition(), buffers);
                collectBuffersFromScalar(t.trueValue(), buffers);
                collectBuffersFromScalar(t.falseValue(), buffers);
            }
            case ScalarCast c -> collectBuffersFromScalar(c.input(), buffers);
            default -> {}
        }
    }

    private void collectAccessedBuffers(LIRNode node, Set<BufferRef> accessed) {
        switch (node) {
            case Store store -> {
                accessed.add(store.buffer());
                collectAccessedFromScalar(store.value(), accessed);
            }
            case ScalarLet let -> collectAccessedFromScalar(let.value(), accessed);
            case StructuredFor sfor -> {
                for (LoopIterArg arg : sfor.iterArgs()) {
                    collectAccessedFromScalar(arg.init(), accessed);
                }
                collectAccessedBuffers(sfor.body(), accessed);
            }
            case TiledLoop tiled -> collectAccessedBuffers(tiled.body(), accessed);
            case Block block ->
                    block.statements().forEach(s -> collectAccessedBuffers(s, accessed));
            case Yield yield -> yield.values().forEach(v -> collectAccessedFromScalar(v, accessed));
            default -> {}
        }
    }

    private void collectAccessedFromScalar(ScalarExpr expr, Set<BufferRef> accessed) {
        switch (expr) {
            case ScalarLoad load -> accessed.add(load.buffer());
            case ScalarUnary u -> collectAccessedFromScalar(u.input(), accessed);
            case ScalarBinary b -> {
                collectAccessedFromScalar(b.left(), accessed);
                collectAccessedFromScalar(b.right(), accessed);
            }
            case ScalarTernary t -> {
                collectAccessedFromScalar(t.condition(), accessed);
                collectAccessedFromScalar(t.trueValue(), accessed);
                collectAccessedFromScalar(t.falseValue(), accessed);
            }
            case ScalarCast c -> collectAccessedFromScalar(c.input(), accessed);
            default -> {}
        }
    }

    private void collectReadBuffers(LIRNode node, Set<BufferRef> read) {
        switch (node) {
            case Store store -> collectReadFromScalar(store.value(), read);
            case ScalarLet let -> collectReadFromScalar(let.value(), read);
            case StructuredFor sfor -> {
                for (LoopIterArg arg : sfor.iterArgs()) {
                    collectReadFromScalar(arg.init(), read);
                }
                collectReadBuffers(sfor.body(), read);
            }
            case TiledLoop tiled -> collectReadBuffers(tiled.body(), read);
            case Block block -> block.statements().forEach(s -> collectReadBuffers(s, read));
            case Yield yield -> yield.values().forEach(v -> collectReadFromScalar(v, read));
            default -> {}
        }
    }

    private void collectReadFromScalar(ScalarExpr expr, Set<BufferRef> read) {
        switch (expr) {
            case ScalarLoad load -> read.add(load.buffer());
            case ScalarUnary u -> collectReadFromScalar(u.input(), read);
            case ScalarBinary b -> {
                collectReadFromScalar(b.left(), read);
                collectReadFromScalar(b.right(), read);
            }
            case ScalarTernary t -> {
                collectReadFromScalar(t.condition(), read);
                collectReadFromScalar(t.trueValue(), read);
                collectReadFromScalar(t.falseValue(), read);
            }
            case ScalarCast c -> collectReadFromScalar(c.input(), read);
            default -> {}
        }
    }

    private void checkOverlappingAllocations(
            LIRGraph graph,
            ScratchLayout layout,
            Set<BufferRef> intermediates,
            List<String> errors) {
        // Compute liveness intervals for intermediates
        Map<BufferRef, LivenessInterval> liveness = computeLiveness(graph.body(), intermediates);

        // Check for overlapping allocations with overlapping lifetimes
        List<BufferRef> buffers = new ArrayList<>(intermediates);
        for (int i = 0; i < buffers.size(); i++) {
            for (int j = i + 1; j < buffers.size(); j++) {
                BufferRef buf1 = buffers.get(i);
                BufferRef buf2 = buffers.get(j);

                Long offset1 = layout.offsets().get(buf1);
                Long offset2 = layout.offsets().get(buf2);

                if (offset1 == null || offset2 == null) {
                    continue; // Already reported as error
                }

                long size1 = buf1.dataType().byteSizeFor(buf1.shape());
                long size2 = buf2.dataType().byteSizeFor(buf2.shape());

                // Check if memory ranges overlap
                long end1 = offset1 + size1;
                long end2 = offset2 + size2;
                boolean memoryOverlap = !(end1 <= offset2 || end2 <= offset1);

                if (memoryOverlap) {
                    // Check if lifetimes overlap
                    LivenessInterval live1 = liveness.get(buf1);
                    LivenessInterval live2 = liveness.get(buf2);

                    if (live1 != null
                            && live2 != null
                            && live1.overlaps(live2)
                            && !buf1.equals(buf2)) {
                        errors.add(
                                "Overlapping scratch allocations for buffers with overlapping lifetimes: "
                                        + bufferDescription(buf1)
                                        + " ["
                                        + live1.firstUse()
                                        + ", "
                                        + live1.lastUse()
                                        + "] at offset "
                                        + offset1
                                        + " and "
                                        + bufferDescription(buf2)
                                        + " ["
                                        + live2.firstUse()
                                        + ", "
                                        + live2.lastUse()
                                        + "] at offset "
                                        + offset2);
                    }
                }
            }
        }
    }

    private Map<BufferRef, LivenessInterval> computeLiveness(LIRNode body, Set<BufferRef> targets) {
        Map<BufferRef, Integer> firstDef = new HashMap<>();
        Map<BufferRef, Integer> lastUse = new HashMap<>();
        int[] idx = {0};

        computeLivenessRec(body, targets, firstDef, lastUse, idx);

        Map<BufferRef, LivenessInterval> result = new HashMap<>();
        for (BufferRef buf : targets) {
            int def = firstDef.getOrDefault(buf, 0);
            int use = lastUse.getOrDefault(buf, def);
            result.put(buf, new LivenessInterval(buf, def, use));
        }
        return result;
    }

    private void computeLivenessRec(
            LIRNode node,
            Set<BufferRef> targets,
            Map<BufferRef, Integer> firstDef,
            Map<BufferRef, Integer> lastUse,
            int[] idx) {
        switch (node) {
            case Store store -> {
                int i = idx[0]++;
                if (targets.contains(store.buffer())) {
                    firstDef.putIfAbsent(store.buffer(), i);
                }
                recordUses(store.value(), targets, lastUse, i);
            }
            case ScalarLet let -> {
                int i = idx[0]++;
                recordUses(let.value(), targets, lastUse, i);
            }
            case StructuredFor sfor -> {
                int i = idx[0]++;
                for (LoopIterArg arg : sfor.iterArgs()) {
                    recordUses(arg.init(), targets, lastUse, i);
                }
                computeLivenessRec(sfor.body(), targets, firstDef, lastUse, idx);
            }
            case TiledLoop tiled -> {
                idx[0]++;
                computeLivenessRec(tiled.body(), targets, firstDef, lastUse, idx);
            }
            case Block block ->
                    block.statements()
                            .forEach(s -> computeLivenessRec(s, targets, firstDef, lastUse, idx));
            case Yield yield -> {
                int i = idx[0]++;
                yield.values().forEach(v -> recordUses(v, targets, lastUse, i));
            }
            default -> {}
        }
    }

    private void recordUses(
            ScalarExpr expr, Set<BufferRef> targets, Map<BufferRef, Integer> lastUse, int idx) {
        switch (expr) {
            case ScalarLoad load -> {
                if (targets.contains(load.buffer())) {
                    lastUse.merge(load.buffer(), idx, Math::max);
                }
            }
            case ScalarUnary u -> recordUses(u.input(), targets, lastUse, idx);
            case ScalarBinary b -> {
                recordUses(b.left(), targets, lastUse, idx);
                recordUses(b.right(), targets, lastUse, idx);
            }
            case ScalarTernary t -> {
                recordUses(t.condition(), targets, lastUse, idx);
                recordUses(t.trueValue(), targets, lastUse, idx);
                recordUses(t.falseValue(), targets, lastUse, idx);
            }
            case ScalarCast c -> recordUses(c.input(), targets, lastUse, idx);
            default -> {}
        }
    }

    private String bufferDescription(BufferRef buffer) {
        return "BufferRef[id="
                + buffer.id()
                + ", type="
                + buffer.dataType()
                + ", shape="
                + buffer.shape()
                + "]";
    }
}
