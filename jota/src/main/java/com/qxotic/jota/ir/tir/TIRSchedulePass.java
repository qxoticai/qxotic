package com.qxotic.jota.ir.tir;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Builds a kernel schedule from a traced TIR graph.
 *
 * <p>This pass emits a deterministic sequence of kernel steps in topological order. Each step has a
 * single materialized output represented by a {@link ValueId}. Unsupported kernels fail fast.
 */
public final class TIRSchedulePass {

    private final LoweringSupportChecker loweringSupportChecker;

    public TIRSchedulePass() {
        this(new LoweringSupportChecker());
    }

    TIRSchedulePass(LoweringSupportChecker loweringSupportChecker) {
        this.loweringSupportChecker =
                Objects.requireNonNull(loweringSupportChecker, "loweringSupportChecker");
    }

    public ScheduledProgram run(TIRGraph graph) {
        Objects.requireNonNull(graph, "graph");
        if (graph.outputs().isEmpty()) {
            throw new IllegalArgumentException("Cannot schedule graph with no outputs");
        }

        TIRNode output = graph.outputs().getFirst();
        List<TIRNode> topo = collectTopo(output);
        List<TIRNode> roots = stepRoots(output, topo);
        List<KernelStep> steps = new ArrayList<>(roots.size());
        Map<TIRNode, ValueId> producedValues = new IdentityHashMap<>();

        int nextValueId = 0;
        for (TIRNode root : roots) {
            TIRStepGraphBuilder.Result result = TIRStepGraphBuilder.build(root, producedValues);
            if (countReductions(result.graph().outputs().getFirst()) > 1) {
                throw new UnsupportedOperationException(
                        "Unsupported scheduled kernel for "
                                + root.getClass().getSimpleName()
                                + ": more than one reduction in kernel");
            }
            loweringSupportChecker.verifyOrThrow(result.graph(), describe(root));

            ValueId out = new ValueId(nextValueId++);
            steps.add(new KernelStep(result.graph(), result.inputs(), out));
            producedValues.put(root, out);
        }

        ScheduledOutputRef scheduledOutput = resolveOutputRef(output, producedValues);
        return new ScheduledProgram(steps, scheduledOutput);
    }

    private static List<TIRNode> collectTopo(TIRNode output) {
        List<TIRNode> topo = new ArrayList<>();
        IdentityHashMap<TIRNode, Boolean> visited = new IdentityHashMap<>();
        dfs(output, visited, topo);
        return topo;
    }

    private static void dfs(
            TIRNode node, IdentityHashMap<TIRNode, Boolean> visited, List<TIRNode> topo) {
        if (visited.put(node, Boolean.TRUE) != null) {
            return;
        }
        for (TIRNode input : TIRNodeUtils.inputsOf(node)) {
            dfs(input, visited, topo);
        }
        topo.add(node);
    }

    private static List<TIRNode> stepRoots(TIRNode output, List<TIRNode> topo) {
        List<TIRNode> roots =
                topo.stream()
                        .filter(TIRNodeUtils::isComputeNode)
                        .collect(Collectors.toCollection(ArrayList::new));
        if (!roots.contains(output) && !TIRNodeUtils.isGraphInputNode(output)) {
            roots.add(output);
        }
        return roots;
    }

    private static ScheduledOutputRef resolveOutputRef(
            TIRNode output, Map<TIRNode, ValueId> producedValues) {
        ValueId produced = producedValues.get(output);
        if (produced != null) {
            return new ScheduledOutputRef.ValueOutput(produced);
        }
        if (output instanceof TensorInput input) {
            return new ScheduledOutputRef.TensorInputOutput(input.id());
        }
        if (output instanceof ScalarInput input) {
            return new ScheduledOutputRef.ScalarInputOutput(input.id());
        }
        throw new IllegalStateException(
                "Output was not materialized by schedule: " + output.getClass().getSimpleName());
    }

    private static String describe(TIRNode node) {
        return "for "
                + node.getClass().getSimpleName()
                + "("
                + node.dataType()
                + " "
                + node.shape()
                + ")";
    }

    private static int countReductions(TIRNode root) {
        Set<TIRNode> visited = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        return countReductions(root, visited);
    }

    private static int countReductions(TIRNode node, Set<TIRNode> visited) {
        if (!visited.add(node)) {
            return 0;
        }
        int count = node instanceof ReductionOp ? 1 : 0;
        for (TIRNode input : TIRNodeUtils.inputsOf(node)) {
            count += countReductions(input, visited);
        }
        return count;
    }
}
