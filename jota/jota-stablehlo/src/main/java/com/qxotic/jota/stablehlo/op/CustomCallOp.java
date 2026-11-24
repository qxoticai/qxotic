package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import java.util.List;
import java.util.Optional;

/** Typed view of {@code stablehlo.custom_call}. Generated from the op table; see {@link Op}. */
public record CustomCallOp(Operation operation) implements Op {

    public static final String NAME = "stablehlo.custom_call";

    public CustomCallOp {
        OpSupport.checkName(operation, NAME);
    }

    public List<Value> inputs() {
        return OpSupport.slice(operation, 0, 0);
    }

    /** Attribute {@code call_target_name}. */
    public String callTargetName() {
        return OpSupport.str(operation, "call_target_name");
    }

    /** Attribute {@code has_side_effect}. */
    public boolean hasSideEffect() {
        return OpSupport.bool(operation, "has_side_effect");
    }

    /** Attribute {@code backend_config}. */
    public Optional<Attribute> backendConfig() {
        return operation.attribute("backend_config");
    }

    /** Attribute {@code api_version}. */
    public Optional<Attribute> apiVersion() {
        return operation.attribute("api_version");
    }

    /** Attribute {@code called_computations}. */
    public Optional<Attribute> calledComputations() {
        return operation.attribute("called_computations");
    }

    /** Attribute {@code operand_layouts}. */
    public Optional<Attribute> operandLayouts() {
        return operation.attribute("operand_layouts");
    }

    /** Attribute {@code result_layouts}. */
    public Optional<Attribute> resultLayouts() {
        return operation.attribute("result_layouts");
    }

    /** Attribute {@code output_operand_aliases}. */
    public Optional<Attribute> outputOperandAliases() {
        return operation.attribute("output_operand_aliases");
    }

    @Override
    public <R> R accept(OpVisitor<R> visitor) {
        return visitor.visitCustomCall(this);
    }
}
