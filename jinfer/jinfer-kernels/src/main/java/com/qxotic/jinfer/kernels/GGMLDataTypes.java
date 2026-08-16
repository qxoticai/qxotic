package com.qxotic.jinfer.kernels;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jota.DataType;

/** Conversion between GGUF storage types and their Jota views. */
public final class GGMLDataTypes {

    private static final Pair[] SUPPORTED = {
        new Pair(GGMLType.F32, DataType.FP32),
        new Pair(GGMLType.F16, DataType.FP16),
        new Pair(GGMLType.BF16, DataType.BF16),
        new Pair(GGMLType.Q4_0, DataType.Q4_0),
        new Pair(GGMLType.Q4_1, DataType.Q4_1),
        new Pair(GGMLType.Q5_1, DataType.Q5_1),
        new Pair(GGMLType.Q8_0, DataType.Q8_0),
        new Pair(GGMLType.Q4_K, DataType.Q4_K),
        new Pair(GGMLType.Q5_K, DataType.Q5_K),
        new Pair(GGMLType.Q6_K, DataType.Q6_K),
        new Pair(GGMLType.MXFP4, DataType.MXFP4),
        new Pair(GGMLType.NVFP4, DataType.NVFP4),
        new Pair(GGMLType.Q1_0, DataType.Q1_0),
        new Pair(GGMLType.TQ1_0, DataType.TQ1_0),
        new Pair(GGMLType.TQ2_0, DataType.TQ2_0)
    };

    private GGMLDataTypes() {}

    public static DataType toDataType(GGMLType type) {
        for (Pair pair : SUPPORTED) if (pair.ggmlType == type) return pair.dataType;
        throw new UnsupportedOperationException("unsupported inference GGMLType " + type);
    }

    public static GGMLType toGGMLType(DataType type) {
        for (Pair pair : SUPPORTED) if (pair.dataType == type) return pair.ggmlType;
        throw new UnsupportedOperationException("unsupported inference DataType " + type);
    }

    private record Pair(GGMLType ggmlType, DataType dataType) {
        private Pair {
            if (ggmlType.getElementsPerBlock() != dataType.elementsPerBlock()
                    || ggmlType.getBlockByteSize() != dataType.byteSize()) {
                throw new ExceptionInInitializerError(
                        ggmlType + " storage geometry disagrees with " + dataType);
            }
        }
    }
}
