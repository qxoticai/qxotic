package com.qxotic.jota.stablehlo.attr;

import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import java.util.Objects;

/** Typed float {@code 1.0e-05 : f32}, stored as exact raw bits of its format. */
public record FloatAttr(long bits, FloatType type) implements Attribute {

    public FloatAttr {
        Objects.requireNonNull(type, "type");
    }

    public static FloatAttr of(double value, FloatType type) {
        return new FloatAttr(type.format().encode(value), type);
    }

    public double value() {
        return type.format().decode(bits);
    }

    /**
     * As MLIR prints it: {@code 1.000000e+00} when six digits round-trip, the natural-precision
     * decimal otherwise, the hexadecimal bit pattern for infinities and NaNs.
     */
    public String literal() {
        return literal(bits, type.format());
    }

    static String literal(long bits, FloatFormat format) {
        return FloatLiteral.of(bits, format);
    }

    static String hex(long bits, int bitWidth) {
        int digits = (bitWidth + 3) / 4;
        StringBuilder h = new StringBuilder(Long.toHexString(bits).toUpperCase());
        while (h.length() < digits) {
            h.insert(0, '0');
        }
        return "0x" + h;
    }

    @Override
    public String toString() {
        return literal() + " : " + type.mlir();
    }
}
