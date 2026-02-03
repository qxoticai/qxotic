package ai.qxotic.jota.ir.lir.scratch;

import ai.qxotic.jota.ir.lir.*;
import java.util.*;

/**
 * Analysis pass that identifies intermediate buffers and computes scratch layout using linear scan
 * allocation with liveness-based memory reuse.
 */
public final class ScratchAnalysisPass {

    /** Analyzes the graph and returns the scratch layout. */
    public ScratchLayout analyze(LIRGraph graph) {
        // Collect input/output buffers
        Set<BufferRef> inputBuffers = newIdentitySet();
        for (LIRInput input : graph.inputs()) {
            if (input instanceof BufferRef buf) {
                inputBuffers.add(buf);
            }
        }
        Set<BufferRef> outputBuffers = newIdentitySet();
        outputBuffers.addAll(graph.outputs());

        // Find all buffers used in the body
        Set<BufferRef> allBuffers = newIdentitySet();
        collectBuffers(graph.body(), allBuffers);

        // Intermediates = all buffers - inputs - outputs
        Set<BufferRef> intermediates = newIdentitySet();
        intermediates.addAll(allBuffers);
        intermediates.removeAll(inputBuffers);
        intermediates.removeAll(outputBuffers);

        if (intermediates.isEmpty()) {
            return ScratchLayout.EMPTY;
        }

        // Compute liveness and assign offsets
        Map<BufferRef, LivenessInterval> liveness = computeLiveness(graph.body(), intermediates);
        return assignOffsets(intermediates, liveness);
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
        collectBuffersFromScalarIter(expr, buffers);
    }

    private void collectBuffersFromScalarIter(ScalarExpr expr, Set<BufferRef> buffers) {
        Deque<ScalarExpr> stack = new ArrayDeque<>();
        Set<ScalarExpr> visited = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        stack.add(expr);
        while (!stack.isEmpty()) {
            ScalarExpr current = stack.removeLast();
            if (!visited.add(current)) {
                continue;
            }
            switch (current) {
                case ScalarLoad load -> buffers.add(load.buffer());
                case ScalarUnary u -> stack.add(u.input());
                case ScalarBinary b -> {
                    stack.add(b.left());
                    stack.add(b.right());
                }
                case ScalarTernary t -> {
                    stack.add(t.condition());
                    stack.add(t.trueValue());
                    stack.add(t.falseValue());
                }
                case ScalarCast c -> stack.add(c.input());
                default -> {}
            }
        }
    }

    private Map<BufferRef, LivenessInterval> computeLiveness(LIRNode body, Set<BufferRef> targets) {
        Map<BufferRef, Integer> firstDef = new IdentityHashMap<>();
        Map<BufferRef, Integer> lastUse = new IdentityHashMap<>();
        int[] idx = {0};

        computeLivenessRec(body, targets, firstDef, lastUse, idx);

        Map<BufferRef, LivenessInterval> result = new IdentityHashMap<>();
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
        recordUsesIter(expr, targets, lastUse, idx);
    }

    private void recordUsesIter(
            ScalarExpr expr, Set<BufferRef> targets, Map<BufferRef, Integer> lastUse, int idx) {
        Deque<ScalarExpr> stack = new ArrayDeque<>();
        Set<ScalarExpr> visited = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        stack.add(expr);
        while (!stack.isEmpty()) {
            ScalarExpr current = stack.removeLast();
            if (!visited.add(current)) {
                continue;
            }
            switch (current) {
                case ScalarLoad load -> {
                    if (targets.contains(load.buffer())) {
                        lastUse.merge(load.buffer(), idx, Math::max);
                    }
                }
                case ScalarUnary u -> stack.add(u.input());
                case ScalarBinary b -> {
                    stack.add(b.left());
                    stack.add(b.right());
                }
                case ScalarTernary t -> {
                    stack.add(t.condition());
                    stack.add(t.trueValue());
                    stack.add(t.falseValue());
                }
                case ScalarCast c -> stack.add(c.input());
                default -> {}
            }
        }
    }

    /** Linear scan allocation with memory reuse for non-overlapping lifetimes. */
    private ScratchLayout assignOffsets(
            Set<BufferRef> buffers, Map<BufferRef, LivenessInterval> liveness) {
        List<BufferRef> sorted = new ArrayList<>(buffers);
        sorted.sort(Comparator.comparingInt(b -> liveness.get(b).firstUse()));

        // Active intervals sorted by end time
        PriorityQueue<long[]> active = new PriorityQueue<>(Comparator.comparingLong(a -> a[0]));
        // Free regions: size -> list of offsets
        TreeMap<Long, Deque<Long>> free = new TreeMap<>();

        Map<BufferRef, Long> offsets = new IdentityHashMap<>();
        long highWater = 0;

        for (BufferRef buf : sorted) {
            LivenessInterval interval = liveness.get(buf);
            long size = alignUp(buf.dataType().byteSizeFor(buf.shape()));

            // Expire dead intervals
            while (!active.isEmpty() && active.peek()[0] < interval.firstUse()) {
                long[] expired = active.poll();
                free.computeIfAbsent(expired[2], k -> new ArrayDeque<>()).addLast(expired[1]);
            }

            // Try to reuse (best-fit)
            long offset = allocateFromFree(free, size);
            if (offset < 0) {
                offset = highWater;
                highWater = offset + size;
            }

            offsets.put(buf, offset);
            active.add(new long[] {interval.lastUse(), offset, size});
        }

        return new ScratchLayout(offsets, highWater);
    }

    private static Set<BufferRef> newIdentitySet() {
        return java.util.Collections.newSetFromMap(new IdentityHashMap<>());
    }

    private long allocateFromFree(TreeMap<Long, Deque<Long>> free, long size) {
        var entry = free.ceilingEntry(size);
        if (entry != null && !entry.getValue().isEmpty()) {
            long offset = entry.getValue().removeFirst();
            if (entry.getValue().isEmpty()) {
                free.remove(entry.getKey());
            }
            // Return remainder to free list
            long remainder = entry.getKey() - size;
            if (remainder > 0) {
                free.computeIfAbsent(remainder, k -> new ArrayDeque<>()).addLast(offset + size);
            }
            return offset;
        }
        return -1;
    }

    private static long alignUp(long size) {
        return (size + ScratchLayout.ALIGNMENT - 1)
                / ScratchLayout.ALIGNMENT
                * ScratchLayout.ALIGNMENT;
    }
}
