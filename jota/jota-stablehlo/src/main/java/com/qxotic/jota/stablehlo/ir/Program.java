package com.qxotic.jota.stablehlo.ir;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.op.FuncOp;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * A StableHLO program (spec "Programs"): a {@code builtin.module} operation whose single block
 * holds functions.
 */
public record Program(Operation operation) {

    public Program {
        Objects.requireNonNull(operation, "operation");
        if (!operation.name().equals("builtin.module")) {
            throw new IllegalArgumentException("not a builtin.module: " + operation.name());
        }
    }

    /** The module symbol name ({@code module @jit_main}), if any. */
    public Optional<String> name() {
        return operation.attribute("sym_name").map(a -> ((StringAttr) a).value());
    }

    public Block body() {
        return operation.region(0).block();
    }

    public List<Operation> operations() {
        return body().operations();
    }

    public List<FuncOp> functions() {
        return operations().stream()
                .filter(op -> op.name().equals("func.func"))
                .map(FuncOp::new)
                .toList();
    }

    public Optional<FuncOp> function(String symbol) {
        return functions().stream().filter(f -> f.symName().equals(symbol)).findFirst();
    }

    public Optional<Attribute> attribute(String name) {
        return operation.attribute(name);
    }
}
