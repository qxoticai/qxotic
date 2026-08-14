package com.qxotic.jam;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.qxotic.jota.DataType;
import org.junit.jupiter.api.Test;

/**
 * jam's public dtype surface is GGML's int tags; jota's is the {@link DataType} constants. The two
 * describe the same on-disk formats - this test is the only place they meet (jota-core is a
 * TEST-scope dependency; jam itself stays zero-dep). Per dtype: the JAM tag resolves to the
 * expected {@link GGMLType} row and the block geometry agrees with jota's, so a drift on either
 * side fails here instead of corrupting a matmul.
 */
class GGMLTypeJotaParityTest {

    /** JAM int tag, its GGMLType row, the matching jota DataType. */
    private static Object[][] rows() {
        return new Object[][] {
            {JAM.F32, GGMLType.F32, DataType.FP32},
            {JAM.F16, GGMLType.F16, DataType.FP16},
            {JAM.BF16, GGMLType.BF16, DataType.BF16},
            {JAM.Q4_0, GGMLType.Q4_0, DataType.Q4_0},
            {JAM.Q8_0, GGMLType.Q8_0, DataType.Q8_0},
            {JAM.Q4_K, GGMLType.Q4_K, DataType.Q4_K},
            {JAM.Q5_K, GGMLType.Q5_K, DataType.Q5_K},
            {JAM.Q6_K, GGMLType.Q6_K, DataType.Q6_K},
            {JAM.MXFP4, GGMLType.MXFP4, DataType.MXFP4},
            {JAM.NVFP4, GGMLType.NVFP4, DataType.NVFP4},
            {JAM.Q1_0, GGMLType.Q1_0, DataType.Q1_0},
        };
    }

    @Test
    void everySupportedDtypeAgreesWithJota() {
        Object[][] rows = rows();
        assertEquals(
                GGMLType.values().length,
                rows.length,
                "a GGMLType row without a jota pairing (or a stale one)");
        for (Object[] row : rows()) {
            int tag = (Integer) row[0];
            GGMLType type = (GGMLType) row[1];
            DataType dt = (DataType) row[2];
            assertSame(type, GGMLType.byCode(tag), type + " <- tag " + tag);
            assertEquals(type.ggml, tag, type + " ggml code == JAM tag");
            assertEquals(type.blockElems, dt.elementsPerBlock(), type + " elements/block vs jota");
            assertEquals(type.blockBytes, dt.byteSize(), type + " bytes/block vs jota");
            // ...and through rowBytes, the conversion the mm bounds checks actually run
            assertEquals(
                    3 * dt.byteSize(),
                    type.rowBytes(3L * type.blockElems),
                    type + " rowBytes vs jota");
        }
    }
}
