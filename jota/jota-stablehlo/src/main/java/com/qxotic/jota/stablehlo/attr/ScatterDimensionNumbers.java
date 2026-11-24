package com.qxotic.jota.stablehlo.attr;

import java.util.List;

/** {@code #stablehlo.scatter<...>} (spec "scatter", inputs I4-I9). */
public record ScatterDimensionNumbers(
        long[] updateWindowDims,
        long[] insertedWindowDims,
        long[] inputBatchingDims,
        long[] scatterIndicesBatchingDims,
        long[] scatterDimsToOperandDims,
        long indexVectorDim) {

    public static ScatterDimensionNumbers from(Attribute attribute) {
        DialectAttr d = DimensionNumbers.expect(attribute, "scatter");
        return new ScatterDimensionNumbers(
                DimensionNumbers.longs(d, "update_window_dims"),
                DimensionNumbers.longs(d, "inserted_window_dims"),
                DimensionNumbers.longs(d, "input_batching_dims"),
                DimensionNumbers.longs(d, "scatter_indices_batching_dims"),
                DimensionNumbers.longs(d, "scatter_dims_to_operand_dims"),
                DimensionNumbers.scalar(d, "index_vector_dim", 0));
    }

    public DialectAttr toAttribute() {
        List<NamedAttribute> p = DimensionNumbers.list();
        DimensionNumbers.add(p, "update_window_dims", updateWindowDims);
        DimensionNumbers.add(p, "inserted_window_dims", insertedWindowDims);
        DimensionNumbers.add(p, "input_batching_dims", inputBatchingDims);
        DimensionNumbers.add(p, "scatter_indices_batching_dims", scatterIndicesBatchingDims);
        DimensionNumbers.add(p, "scatter_dims_to_operand_dims", scatterDimsToOperandDims);
        if (indexVectorDim != 0) { // the StableHLO printer omits the default
            p.add(new NamedAttribute("index_vector_dim", IntegerAttr.i64(indexVectorDim)));
        }
        return DialectAttr.ofParams("stablehlo", "scatter", p);
    }
}
