package com.qxotic.jota.stablehlo.ir;

import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/** Rebuilds immutable IR trees with modified operations. */
public final class IrRewriter {

    /** New properties or attributes of an operation (given the current ones). */
    @FunctionalInterface
    public interface AttributeRewrite {
        List<NamedAttribute> apply(Operation op, List<NamedAttribute> current);
    }

    private final Map<Value, Value> values = new IdentityHashMap<>();
    private final AttributeRewrite properties;
    private final AttributeRewrite attributes;

    private IrRewriter(AttributeRewrite properties, AttributeRewrite attributes) {
        this.properties = properties;
        this.attributes = attributes;
    }

    /** A copy of {@code program} whose operations' properties/attributes were rewritten. */
    public static Program rewriteAttributes(
            Program program, AttributeRewrite properties, AttributeRewrite attributes) {
        return new Program(new IrRewriter(properties, attributes).copy(program.operation()));
    }

    private Operation copy(Operation op) {
        List<Value> operands = new ArrayList<>(op.numOperands());
        for (Value v : op.operands()) {
            Value mapped = values.get(v);
            operands.add(mapped != null ? mapped : v);
        }
        List<Region> regions = new ArrayList<>();
        for (int i = 0; i < op.regions().size(); i++) {
            regions.add(new Region());
        }
        Operation out =
                Operation.create(
                        op.name(),
                        operands,
                        op.resultTypes(),
                        op.results().stream().map(Value::nameHint).toList(),
                        properties.apply(op, op.properties()),
                        attributes.apply(op, op.attributes()),
                        regions);
        for (int i = 0; i < op.numResults(); i++) {
            values.put(op.result(i), out.result(i));
        }
        for (int i = 0; i < op.regions().size(); i++) {
            for (Block b : op.region(i).blocks()) {
                Block nb = new Block();
                nb.setLabel(b.label());
                for (Value arg : b.arguments()) {
                    values.put(arg, nb.addArgument(arg.type(), arg.nameHint()));
                }
                for (Operation inner : b.operations()) {
                    nb.addOperation(copy(inner));
                }
                regions.get(i).addBlock(nb);
            }
        }
        return out;
    }
}
