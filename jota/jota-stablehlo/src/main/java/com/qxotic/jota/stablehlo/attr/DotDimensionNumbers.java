package com.qxotic.jota.stablehlo.attr;

import java.util.List;

/** {@code #stablehlo.dot<...>} (spec "dot_general", inputs I3-I6). */
public record DotDimensionNumbers(
        long[] lhsBatchingDimensions,
        long[] rhsBatchingDimensions,
        long[] lhsContractingDimensions,
        long[] rhsContractingDimensions) {

    public static DotDimensionNumbers from(Attribute attribute) {
        DialectAttr d = DimensionNumbers.expect(attribute, "dot");
        return new DotDimensionNumbers(
                DimensionNumbers.longs(d, "lhs_batching_dimensions"),
                DimensionNumbers.longs(d, "rhs_batching_dimensions"),
                DimensionNumbers.longs(d, "lhs_contracting_dimensions"),
                DimensionNumbers.longs(d, "rhs_contracting_dimensions"));
    }

    public DialectAttr toAttribute() {
        List<NamedAttribute> p = DimensionNumbers.list();
        DimensionNumbers.add(p, "lhs_batching_dimensions", lhsBatchingDimensions);
        DimensionNumbers.add(p, "rhs_batching_dimensions", rhsBatchingDimensions);
        DimensionNumbers.add(p, "lhs_contracting_dimensions", lhsContractingDimensions);
        DimensionNumbers.add(p, "rhs_contracting_dimensions", rhsContractingDimensions);
        return DialectAttr.ofParams("stablehlo", "dot", p);
    }
}
