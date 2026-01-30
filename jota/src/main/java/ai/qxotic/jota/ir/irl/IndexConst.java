package ai.qxotic.jota.ir.irl;

/** Constant index value. */
public record IndexConst(long value) implements IndexExpr {

    public static final IndexConst ZERO = new IndexConst(0);
    public static final IndexConst ONE = new IndexConst(1);
}
