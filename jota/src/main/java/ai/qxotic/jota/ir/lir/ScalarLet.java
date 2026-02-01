package ai.qxotic.jota.ir.lir;

import java.util.Objects;

/**
 * Defines a named scalar value (SSA-style definition).
 *
 * <p>The bound value can be referenced by {@link ScalarRef} nodes in subsequent statements.
 *
 * <p>Example:
 *
 * <pre>
 * %hoisted = multiply fp32 %a, %b
 * for %i in [0, N) {
 *   store %out[%i], %hoisted
 * }
 * </pre>
 *
 * @param name the name to bind
 * @param value the scalar expression to compute
 */
public record ScalarLet(String name, ScalarExpr value) implements LIRNode {

    public ScalarLet {
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(value, "value cannot be null");
    }
}
