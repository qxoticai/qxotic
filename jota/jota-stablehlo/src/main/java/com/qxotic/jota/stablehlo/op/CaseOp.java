package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code stablehlo.case}. Generated from the op table; see {@link Op}. */
public record CaseOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.case";

    public CaseOp {
        OpSupport.checkName(operation, NAME);
    }

    public Value index() {
        return operation.operand(0);
    }

    public List<Region> branches() {
        return operation.regions();
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCase(this);
    }
}
