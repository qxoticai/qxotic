package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.ScatterDimensionNumbers;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;

/** Typed view of {@code stablehlo.scatter}. Generated from the op table; see {@link Op}. */
public record ScatterOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.scatter";

    public ScatterOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.scatterGroup(operation, 0);
    }

    public Value scatterIndices() {
        return OpSupport.scatterGroup(operation, 1).getFirst();
    }

    public List<Value> updates() {
        return OpSupport.scatterGroup(operation, 2);
    }

    /** Attribute {@code scatter_dimension_numbers}. */
    public ScatterDimensionNumbers scatterDimensionNumbers() {
        return ScatterDimensionNumbers.from(
                OpSupport.required(operation, "scatter_dimension_numbers"));
    }

    /** Attribute {@code indices_are_sorted}. */
    public boolean indicesAreSorted() {
        return OpSupport.bool(operation, "indices_are_sorted");
    }

    /** Attribute {@code unique_indices}. */
    public boolean uniqueIndices() {
        return OpSupport.bool(operation, "unique_indices");
    }

    public Region updateComputation() {
        return operation.region(0);
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitScatter(this);
    }
}
