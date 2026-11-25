package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code stablehlo.while}. Generated from the op table; see {@link Op}. */
public record WhileOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.while";

    public WhileOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> operand() {
        return OpSupport.slice(operation, 0, 0);
    }

    public Region cond() {
        return operation.region(0);
    }

    public Region body() {
        return operation.region(1);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitWhile(this);
    }
}
