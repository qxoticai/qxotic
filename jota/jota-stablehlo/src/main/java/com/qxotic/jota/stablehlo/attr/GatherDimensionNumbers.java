package com.qxotic.jota.stablehlo.attr;

import java.util.List;

/** {@code #stablehlo.gather<...>} (spec "gather", inputs I3-I8). */
public record GatherDimensionNumbers(
        long[] offsetDims,
        long[] collapsedSliceDims,
        long[] operandBatchingDims,
        long[] startIndicesBatchingDims,
        long[] startIndexMap,
        long indexVectorDim) {

    public static GatherDimensionNumbers from(Attribute attribute) {
        DialectAttr d = DimensionNumbers.expect(attribute, "gather");
        return new GatherDimensionNumbers(
                DimensionNumbers.longs(d, "offset_dims"),
                DimensionNumbers.longs(d, "collapsed_slice_dims"),
                DimensionNumbers.longs(d, "operand_batching_dims"),
                DimensionNumbers.longs(d, "start_indices_batching_dims"),
                DimensionNumbers.longs(d, "start_index_map"),
                DimensionNumbers.scalar(d, "index_vector_dim", 0));
    }

    public DialectAttr toAttribute() {
        List<NamedAttribute> p = DimensionNumbers.list();
        DimensionNumbers.add(p, "offset_dims", offsetDims);
        DimensionNumbers.add(p, "collapsed_slice_dims", collapsedSliceDims);
        DimensionNumbers.add(p, "operand_batching_dims", operandBatchingDims);
        DimensionNumbers.add(p, "start_indices_batching_dims", startIndicesBatchingDims);
        DimensionNumbers.add(p, "start_index_map", startIndexMap);
        if (indexVectorDim != 0) { // the StableHLO printer omits the default
            p.add(new NamedAttribute("index_vector_dim", IntegerAttr.i64(indexVectorDim)));
        }
        return DialectAttr.ofParams("stablehlo", "gather", p);
    }
}
