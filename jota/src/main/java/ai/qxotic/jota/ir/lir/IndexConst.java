package ai.qxotic.jota.ir.lir;

/** Constant index value. */
public record IndexConst(long value) implements IndexExpr {

    public static final IndexConst ZERO = new IndexConst(0);
    public static final IndexConst ONE = new IndexConst(1);

    @Override
    public boolean isZero() {
        return value == 0;
    }

    @Override
    public boolean isOne() {
        return value == 1;
    }

    @Override
    public boolean isConstant() {
        return true;
    }

    @Override
    public long constantValue() {
        return value;
    }

    /** Factory that returns cached constants for 0 and 1. */
    public static IndexConst of(long value) {
        if (value == 0) return ZERO;
        if (value == 1) return ONE;
        return new IndexConst(value);
    }
}
