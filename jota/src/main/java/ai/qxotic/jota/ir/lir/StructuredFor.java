package ai.qxotic.jota.ir.lir;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Structured loop with explicit bounds, step, and loop-carried values. */
public record StructuredFor(
        String indexName,
        IndexExpr lowerBound,
        IndexExpr upperBound,
        IndexExpr step,
        List<LoopIterArg> iterArgs,
        LIRNode body)
        implements LIRNode {

    public StructuredFor {
        Objects.requireNonNull(indexName, "indexName cannot be null");
        if (indexName.isEmpty()) {
            throw new IllegalArgumentException("indexName cannot be empty");
        }
        Objects.requireNonNull(lowerBound, "lowerBound cannot be null");
        Objects.requireNonNull(upperBound, "upperBound cannot be null");
        Objects.requireNonNull(step, "step cannot be null");
        Objects.requireNonNull(iterArgs, "iterArgs cannot be null");
        iterArgs = List.copyOf(iterArgs);
        Objects.requireNonNull(body, "body cannot be null");

        validateIterArgs(iterArgs);
        validateYield(body, iterArgs.size());
    }

    /** Creates a loop with constant bounds and step. */
    public static StructuredFor of(
            String indexName,
            long lowerBound,
            long upperBound,
            long step,
            List<LoopIterArg> iterArgs,
            LIRNode body) {
        return new StructuredFor(
                indexName,
                new IndexConst(lowerBound),
                new IndexConst(upperBound),
                new IndexConst(step),
                iterArgs,
                body);
    }

    /** Creates a loop with constant bounds, step, and no iter args. */
    public static StructuredFor simple(
            String indexName, long lowerBound, long upperBound, long step, LIRNode body) {
        return of(indexName, lowerBound, upperBound, step, List.of(), body);
    }

    private static void validateIterArgs(List<LoopIterArg> iterArgs) {
        Set<String> seen = new HashSet<>();
        for (LoopIterArg arg : iterArgs) {
            if (!seen.add(arg.name())) {
                throw new IllegalArgumentException("Duplicate iter arg name: " + arg.name());
            }
        }
    }

    private static void validateYield(LIRNode body, int expectedArity) {
        Yield yield = extractYield(body);
        if (yield.values().size() != expectedArity) {
            throw new IllegalArgumentException(
                    "Yield arity "
                            + yield.values().size()
                            + " does not match iter args "
                            + expectedArity);
        }
    }

    private static Yield extractYield(LIRNode body) {
        if (body instanceof Yield yield) {
            return yield;
        }
        if (body instanceof Block block) {
            if (block.statements().isEmpty()) {
                throw new IllegalArgumentException("Loop body cannot be empty; yield required");
            }
            LIRNode last = block.statements().getLast();
            if (last instanceof Yield yield) {
                return yield;
            }
        }
        throw new IllegalArgumentException("Structured loop body must end with Yield");
    }
}
