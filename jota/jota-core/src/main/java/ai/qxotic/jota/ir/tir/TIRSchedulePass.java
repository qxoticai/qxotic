package ai.qxotic.jota.ir.tir;

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
        List<KernelStep> steps = new ArrayList<>();
        Map<TIRNode, ValueId> producedValues = new IdentityHashMap<>();

        int nextValueId = 0;
        for (int i = 0; i < roots.size(); ) {
            int best = i;
            TIRStepGraphBuilder.Result bestResult = null;

            for (int j = i; j < roots.size(); j++) {
                if (!canFuseRange(roots, i, j)) {
                    break;
                }

                TIRNode candidateRoot = roots.get(j);
                TIRStepGraphBuilder.Result candidate =
                        TIRStepGraphBuilder.build(candidateRoot, producedValues);
                if (countReductions(candidate.graph().outputs().getFirst()) > 1) {
                    break;
                }

                try {
                    loweringSupportChecker.verifyOrThrow(candidate.graph(), describe(candidateRoot));
                } catch (UnsupportedOperationException ex) {
                    break;
                }

                best = j;
                bestResult = candidate;
            }

            if (bestResult == null) {
                TIRNode root = roots.get(i);
                TIRStepGraphBuilder.Result fallback = TIRStepGraphBuilder.build(root, producedValues);
                if (countReductions(fallback.graph().outputs().getFirst()) > 1) {
                    throw new UnsupportedOperationException(
                            "Unsupported scheduled kernel for "
                                    + root.getClass().getSimpleName()
                                    + ": more than one reduction in fused kernel");
                }
                loweringSupportChecker.verifyOrThrow(fallback.graph(), describe(root));
                bestResult = fallback;
                best = i;
            }

            ValueId out = new ValueId(nextValueId++);
            steps.add(new KernelStep(bestResult.graph(), bestResult.inputs(), out));
            producedValues.put(roots.get(best), out);
            i = best + 1;
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
                topo.stream().filter(TIRNodeUtils::isComputeNode).collect(Collectors.toCollection(ArrayList::new));
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

    private static boolean canFuseRange(List<TIRNode> roots, int startInclusive, int endInclusive) {
        TIRNode candidateRoot = roots.get(endInclusive);
        for (int i = startInclusive; i < endInclusive; i++) {
            if (!isAncestor(roots.get(i), candidateRoot)) {
                return false;
            }
        }
        return true;
    }

    private static boolean isAncestor(TIRNode ancestor, TIRNode node) {
        if (ancestor == node) {
            return true;
        }
        for (TIRNode input : TIRNodeUtils.inputsOf(node)) {
            if (isAncestor(ancestor, input)) {
                return true;
            }
        }
        return false;
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
