package com.qxotic.jota.stablehlo.ir;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.op.Op;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * A generic operation: name, operands, results, attributes and regions, exactly as in MLIR's
 * generic form {@code %r = "dialect.op"(%a, %b) <{properties}> ({regions}) {attributes} : (A, B) ->
 * R}.
 *
 * <p>Inherent attributes (the ones an op definition declares, e.g. {@code comparison_direction})
 * live in {@link #properties()}; discardable attributes ({@code mhlo.sharding}...) live in {@link
 * #attributes()}. {@link #attribute(String)} looks in both. The typed view of an operation is
 * {@link #asOp()}.
 */
public final class Operation {

    private final String name;
    private final List<Value> operands;
    private final List<Value> results;
    private final List<NamedAttribute> properties;
    private final List<NamedAttribute> attributes;
    private final List<Region> regions;

    private Operation(
            String name,
            List<Value> operands,
            List<Type> resultTypes,
            List<String> resultNames,
            List<NamedAttribute> properties,
            List<NamedAttribute> attributes,
            List<Region> regions) {
        this.name = Objects.requireNonNull(name, "name");
        this.operands = List.copyOf(operands);
        this.properties = List.copyOf(properties);
        this.attributes = List.copyOf(attributes);
        this.regions = List.copyOf(regions);
        List<Value> res = new ArrayList<>(resultTypes.size());
        for (int i = 0; i < resultTypes.size(); i++) {
            String hint = resultNames != null && i < resultNames.size() ? resultNames.get(i) : null;
            Value v = new Value(resultTypes.get(i), hint);
            v.setOwner(this, i);
            res.add(v);
        }
        this.results = List.copyOf(res);
    }

    /** Creates an operation; result values are created from {@code resultTypes}. */
    public static Operation create(
            String name,
            List<Value> operands,
            List<Type> resultTypes,
            List<NamedAttribute> properties,
            List<NamedAttribute> attributes,
            List<Region> regions) {
        return new Operation(name, operands, resultTypes, null, properties, attributes, regions);
    }

    /** As {@link #create} with source names for the results. */
    public static Operation create(
            String name,
            List<Value> operands,
            List<Type> resultTypes,
            List<String> resultNames,
            List<NamedAttribute> properties,
            List<NamedAttribute> attributes,
            List<Region> regions) {
        return new Operation(
                name, operands, resultTypes, resultNames, properties, attributes, regions);
    }

    /** Full name {@code dialect.mnemonic}. */
    public String name() {
        return name;
    }

    public String dialect() {
        int dot = name.indexOf('.');
        return dot < 0 ? "" : name.substring(0, dot);
    }

    public String mnemonic() {
        int dot = name.indexOf('.');
        return dot < 0 ? name : name.substring(dot + 1);
    }

    public List<Value> operands() {
        return operands;
    }

    public Value operand(int i) {
        return operands.get(i);
    }

    public int numOperands() {
        return operands.size();
    }

    public List<Value> results() {
        return results;
    }

    public Value result(int i) {
        return results.get(i);
    }

    public int numResults() {
        return results.size();
    }

    public List<Type> resultTypes() {
        return results.stream().map(Value::type).toList();
    }

    public List<Type> operandTypes() {
        return operands.stream().map(Value::type).toList();
    }

    public List<NamedAttribute> properties() {
        return properties;
    }

    public List<NamedAttribute> attributes() {
        return attributes;
    }

    /** Properties then attributes. */
    public List<NamedAttribute> allAttributes() {
        List<NamedAttribute> all = new ArrayList<>(properties);
        all.addAll(attributes);
        return all;
    }

    /** Looks an attribute up by name in the properties, then in the discardable attributes. */
    public Optional<Attribute> attribute(String attrName) {
        for (NamedAttribute a : properties) {
            if (a.name().equals(attrName)) {
                return Optional.of(a.value());
            }
        }
        for (NamedAttribute a : attributes) {
            if (a.name().equals(attrName)) {
                return Optional.of(a.value());
            }
        }
        return Optional.empty();
    }

    public List<Region> regions() {
        return regions;
    }

    public Region region(int i) {
        return regions.get(i);
    }

    /** Visits this operation, then every operation nested in its regions, in pre-order. */
    public void walk(java.util.function.Consumer<Operation> visitor) {
        visitor.accept(this);
        for (Region region : regions) {
            for (Block block : region.blocks()) {
                for (Operation op : block.operations()) {
                    op.walk(visitor);
                }
            }
        }
    }

    /** The typed view ({@code AddOp}, {@code GatherOp}...; {@code GenericOp} when unknown). */
    public Op asOp() {
        return Op.wrap(this);
    }

    @Override
    public String toString() {
        return name + operandTypes() + " -> " + resultTypes();
    }
}
