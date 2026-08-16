package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jam.JAM;
import com.qxotic.jota.DataType;
import org.junit.jupiter.api.Test;

/**
 * The jinfer end of the DataType-to-ggml-tag contract: {@link MatMul#jamTag} is the single adapter
 * between jota's lingua franca and jam's wire codes, and {@link MatMul#jamApplies} is a second
 * hand-maintained list of the same dtype set - this test pins both against ONE table. The loop
 * closes on the jam side: jam-core's GGMLTypeJotaParityTest pins each JAM tag's block geometry
 * against jota's, so DataType geometry == geometry of jamTag(DataType) holds transitively.
 */
class MatMulJamTagTest {

    /** Every DataType jam runs, paired with its JAM tag - an independent copy of jamTag's chain. */
    private static Object[][] supported() {
        return new Object[][] {
            {DataType.FP32, JAM.F32},
            {DataType.FP16, JAM.F16},
            {DataType.BF16, JAM.BF16},
            {DataType.Q4_0, JAM.Q4_0},
            {DataType.Q8_0, JAM.Q8_0},
            {DataType.Q4_K, JAM.Q4_K},
            {DataType.Q5_K, JAM.Q5_K},
            {DataType.Q6_K, JAM.Q6_K},
            {DataType.MXFP4, JAM.MXFP4},
            {DataType.NVFP4, JAM.NVFP4},
            {DataType.Q1_0, JAM.Q1_0},
        };
    }

    /** The jota dtypes jam must DECLINE (dense non-floats, FP64, and the unported quants). */
    private static DataType[] unsupported() {
        return new DataType[] {
            DataType.BOOL,
            DataType.I8,
            DataType.I16,
            DataType.I32,
            DataType.I64,
            DataType.FP64,
            DataType.Q4_1,
            DataType.Q5_1,
            DataType.TQ1_0,
            DataType.TQ2_0,
        };
    }

    @Test
    void everySupportedDtypeMapsToItsJamTag() {
        for (Object[] p : supported())
            assertEquals((int) p[1], MatMul.jamTag((DataType) p[0]), p[0] + " tag");
    }

    @Test
    void jamAppliesCoversExactlyTheSameDtypeSetAsJamTag() {
        for (Object[] p : supported()) {
            DataType dt = (DataType) p[0];
            long epb = dt.elementsPerBlock();
            assertTrue(
                    MatMul.jamApplies(dt, (int) (2 * epb), epb),
                    dt + ": jamApplies must accept what jamTag maps (aligned)");
        }
        for (DataType dt : unsupported()) {
            assertFalse(
                    MatMul.jamApplies(dt, 4096, 0),
                    dt + ": jamApplies must decline what jamTag cannot map");
            assertThrows(IllegalArgumentException.class, () -> MatMul.jamTag(dt), dt + " tag");
        }
    }

    @Test
    void jamAppliesDemandsBlockAlignmentOfKAndWeightOffset() {
        long epb = DataType.Q4_K.elementsPerBlock(); // 256
        assertTrue(MatMul.jamApplies(DataType.Q4_K, (int) epb, 0));
        assertFalse(MatMul.jamApplies(DataType.Q4_K, (int) epb + 1, 0), "k not a block multiple");
        assertFalse(MatMul.jamApplies(DataType.Q4_K, (int) epb, 8), "weight offset misaligned");
    }
}
