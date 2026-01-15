package ai.llm4j.jota;

import java.util.Objects;

@FunctionalInterface
public interface FloatPredicate {

    boolean test(float value);

    default FloatPredicate and(FloatPredicate that) {
        Objects.requireNonNull(that);
        return (value) -> this.test(value) && that.test(value);
    }

    default FloatPredicate or(FloatPredicate that) {
        Objects.requireNonNull(that);
        return (value) -> this.test(value) || that.test(value);
    }

    default FloatPredicate negate() {
        return (value) -> !test(value);
    }

    FloatPredicate TRUE = unused -> true;
    FloatPredicate FALSE = unused -> false;
    FloatPredicate IS_NAN = Float::isNaN;
}
