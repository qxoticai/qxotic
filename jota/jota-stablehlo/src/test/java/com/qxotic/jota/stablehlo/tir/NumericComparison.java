package com.qxotic.jota.stablehlo.tir;

import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.type.ElementType;
import com.qxotic.jota.stablehlo.type.FloatType;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Runs the {@code @main} and {@code @expected} functions of a {@code stablehlo/testdata} program
 * through TIR and compares the results (exact for integers and booleans, relative tolerance for
 * floats: XLA's transcendental functions are not bit-identical to Java's).
 */
public final class NumericComparison {

    private NumericComparison() {}

    /** Null when the results agree, otherwise a description of the first difference. */
    public static String check(Program program, MemoryDomain<MemorySegment> domain) {
        TirConversion main = StableHloToTirConverter.convert(program, "main");
        TirConversion expected = StableHloToTirConverter.convert(program, "expected");
        DenseElementsAttr got =
                TirConversion.toDense(
                        main.interpret(List.of(), domain).getFirst(),
                        main.resultTypes().getFirst());
        DenseElementsAttr want =
                TirConversion.toDense(
                        expected.interpret(List.of(), domain).getFirst(),
                        expected.resultTypes().getFirst());
        return compare(got, want);
    }

    static String compare(DenseElementsAttr got, DenseElementsAttr want) {
        long n = want.numElements();
        if (got.numElements() != n) {
            return "element count " + got.numElements() + " vs " + n;
        }
        ElementType et = want.type().elementType();
        double tolerance =
                et instanceof FloatType ft
                        ? switch (ft.format()) {
                            case F16, BF16 -> 2e-2;
                            case F32 -> 1e-4;
                            default -> 1e-9;
                        }
                        : 0;
        for (long i = 0; i < n; i++) {
            if (et instanceof FloatType) {
                double x = got.getDouble(i);
                double y = want.getDouble(i);
                if ((Double.isNaN(x) && Double.isNaN(y))
                        || x == y
                        || Math.abs(x - y) <= tolerance * Math.max(1, Math.abs(y))) {
                    continue;
                }
                return "element " + i + ": got " + x + ", expected " + y;
            }
            if (got.getLong(i) != want.getLong(i)) {
                return "element " + i + ": got " + got.getLong(i) + ", expected " + want.getLong(i);
            }
        }
        return null;
    }
}
