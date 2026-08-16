package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jota.DataType;
import org.junit.jupiter.api.Test;

class GGMLDataTypesTest {

    @Test
    void supportedTypesRoundTrip() {
        GGMLType[] types = {
            GGMLType.F32,
            GGMLType.F16,
            GGMLType.BF16,
            GGMLType.Q4_0,
            GGMLType.Q4_1,
            GGMLType.Q5_1,
            GGMLType.Q8_0,
            GGMLType.Q4_K,
            GGMLType.Q5_K,
            GGMLType.Q6_K,
            GGMLType.MXFP4,
            GGMLType.NVFP4,
            GGMLType.Q1_0,
            GGMLType.TQ1_0,
            GGMLType.TQ2_0
        };
        for (GGMLType type : types) {
            DataType dataType = GGMLDataTypes.toDataType(type);
            assertEquals(type, GGMLDataTypes.toGGMLType(dataType));
            assertEquals(type.getElementsPerBlock(), dataType.elementsPerBlock(), type.name());
            assertEquals(type.getBlockByteSize(), dataType.byteSize(), type.name());
        }
    }

    @Test
    void unsupportedTypesFailClearly() {
        assertThrows(
                UnsupportedOperationException.class, () -> GGMLDataTypes.toDataType(GGMLType.Q5_0));
        assertThrows(
                UnsupportedOperationException.class, () -> GGMLDataTypes.toGGMLType(DataType.I8));
    }
}
