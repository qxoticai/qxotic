package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class IotaOpTest {

    private static void assumeArangeSupportedOnConfiguredDevice() {
        Assumptions.assumeTrue(
                ConfiguredTestDevice.resolve() == Device.PANAMA,
                "Tensor.iota currently panama-only in runtime-agnostic lane");
    }

    @Test
    void iotaDefaultsToI64() {
        assumeArangeSupportedOnConfiguredDevice();
        Tensor range = Tensor.iota(6);

        assertTrue(TensorTestInternals.isLazy(range));
        assertFalse(TensorTestInternals.isMaterialized(range));

        MemoryView<?> output = range.materialize();

        assertEquals(DataType.I64, output.dataType());
        assertEquals(Shape.flat(6), output.shape());
        assertEquals(0L, TensorTestReads.readLong(range, 0));
        assertEquals(5L, TensorTestReads.readLong(range, 5));
    }

    @Test
    void iotaCastsToFp32() {
        assumeArangeSupportedOnConfiguredDevice();
        Tensor range = Tensor.iota(6, DataType.FP32);
        MemoryView<?> output = range.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.flat(6), output.shape());
        assertEquals(0.0f, TensorTestReads.readFloat(range, 0), 0.0001f);
        assertEquals(5.0f, TensorTestReads.readFloat(range, 5), 0.0001f);
    }

    @Test
    void iotaRejectsBool() {
        assumeArangeSupportedOnConfiguredDevice();
        assertThrows(IllegalArgumentException.class, () -> Tensor.iota(4, DataType.BOOL));
    }

    @Test
    void iotaRejectsNegativeCounts() {
        assumeArangeSupportedOnConfiguredDevice();
        assertThrows(IllegalArgumentException.class, () -> Tensor.iota(-1));
    }
}
