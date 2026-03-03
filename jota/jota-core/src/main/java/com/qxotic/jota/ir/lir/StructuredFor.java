package com.qxotic.jota.ir.lir;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Structured loop with explicit bounds, step, and loop-carried values. */
public final class StructuredFor extends LIRExprNode {
    private final String indexName;
    private final List<LoopIterArg> iterArgs;
    private final Block body;
    private final int iterArgCount;

    StructuredFor(
            int id,
            String indexName,
            LIRExprNode lowerBound,
            LIRExprNode upperBound,
            LIRExprNode step,
            List<LoopIterArg> iterArgs,
            Block body) {
        super(
                id,
                LIRExprKind.STRUCTURED_FOR,
                null,
                buildInputs(lowerBound, upperBound, step, iterArgs, body),
                false,
                false);
        Objects.requireNonNull(indexName, "indexName cannot be null");
        if (indexName.isEmpty()) {
            throw new IllegalArgumentException("indexName cannot be empty");
        }
        Objects.requireNonNull(iterArgs, "iterArgs cannot be null");
        Objects.requireNonNull(body, "body cannot be null");
        this.indexName = indexName;
        this.iterArgs = List.copyOf(iterArgs);
        this.body = body;
        this.iterArgCount = iterArgs.size();

        validateIterArgs(this.iterArgs);
        validateYield(body, this.iterArgCount);
    }

    public String indexName() {
        return indexName;
    }

    public LIRExprNode lowerBound() {
        return inputs()[0];
    }

    public LIRExprNode upperBound() {
        return inputs()[1];
    }

    public LIRExprNode step() {
        return inputs()[2];
    }

    public List<LoopIterArg> iterArgs() {
        return iterArgs;
    }

    public Block body() {
        return body;
    }

    public int iterArgCount() {
        return iterArgCount;
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }

    private static LIRExprNode[] buildInputs(
            LIRExprNode lowerBound,
            LIRExprNode upperBound,
            LIRExprNode step,
            List<LoopIterArg> iterArgs,
            Block body) {
        Objects.requireNonNull(lowerBound, "lowerBound cannot be null");
        Objects.requireNonNull(upperBound, "upperBound cannot be null");
        Objects.requireNonNull(step, "step cannot be null");
        Objects.requireNonNull(iterArgs, "iterArgs cannot be null");
        Objects.requireNonNull(body, "body cannot be null");

        int size = 3 + iterArgs.size() + 1;
        LIRExprNode[] inputs = new LIRExprNode[size];
        inputs[0] = lowerBound;
        inputs[1] = upperBound;
        inputs[2] = step;
        int idx = 3;
        for (LoopIterArg arg : iterArgs) {
            inputs[idx++] = Objects.requireNonNull(arg.init(), "iter arg init cannot be null");
        }
        inputs[idx] = body;
        return inputs;
    }

    private static void validateIterArgs(List<LoopIterArg> iterArgs) {
        Set<String> seen = new HashSet<>();
        for (LoopIterArg arg : iterArgs) {
            if (!seen.add(arg.name())) {
                throw new IllegalArgumentException("Duplicate iter arg name: " + arg.name());
            }
        }
    }

    private static void validateYield(Block body, int expectedArity) {
        Yield yield = extractYield(body);
        if (yield.values().size() != expectedArity) {
            throw new IllegalArgumentException(
                    "Yield arity "
                            + yield.values().size()
                            + " does not match iter args "
                            + expectedArity);
        }
    }

    private static Yield extractYield(Block body) {
        if (body.statements().isEmpty()) {
            throw new IllegalArgumentException("Loop body cannot be empty; yield required");
        }
        LIRExprNode last = body.statements().getLast();
        if (last instanceof Yield yield) {
            return yield;
        }
        throw new IllegalArgumentException("Structured loop body must end with Yield");
    }
}
