package com.qxotic.jota.stablehlo.type;

/**
 * Integer element type. {@code i1} is the boolean type; signless {@code iN} (MLIR spelling) is
 * StableHLO's signed {@code siN}; {@code uiN} is unsigned.
 */
public record IntegerType(int bitWidth, Signedness signedness) implements ElementType {

    /** MLIR integer signedness. StableHLO treats signless as signed. */
    public enum Signedness {
        SIGNLESS,
        SIGNED,
        UNSIGNED
    }

    public static final IntegerType I1 = signless(1);
    public static final IntegerType I8 = signless(8);
    public static final IntegerType I16 = signless(16);
    public static final IntegerType I32 = signless(32);
    public static final IntegerType I64 = signless(64);

    public IntegerType {
        if (bitWidth < 1 || bitWidth > 64) {
            throw new IllegalArgumentException("unsupported integer bit width: " + bitWidth);
        }
    }

    public static IntegerType signless(int bits) {
        return new IntegerType(bits, Signedness.SIGNLESS);
    }

    public static IntegerType unsigned(int bits) {
        return new IntegerType(bits, Signedness.UNSIGNED);
    }

    public boolean isBoolean() {
        return bitWidth == 1;
    }

    public boolean isUnsigned() {
        return signedness == Signedness.UNSIGNED;
    }

    /** Smallest representable value (as a signed long; i1 counts as unsigned 0..1). */
    public long minValue() {
        if (isUnsigned() || isBoolean()) {
            return 0;
        }
        return bitWidth == 64 ? Long.MIN_VALUE : -(1L << (bitWidth - 1));
    }

    @Override
    public int storageBytes() {
        return (bitWidth + 7) / 8;
    }

    @Override
    public String mlir() {
        return switch (signedness) {
            case SIGNLESS -> "i" + bitWidth;
            case SIGNED -> "si" + bitWidth;
            case UNSIGNED -> "ui" + bitWidth;
        };
    }

    @Override
    public String toString() {
        return mlir();
    }
}
