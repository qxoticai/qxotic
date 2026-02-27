package com.qxotic.jota.tensor.contracts;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class MaterializationSemanticsContractTest {

    @Test
    void materializeMarksTensorAsMaterialized() {
        Tensor tensor = Tensor.iota(4, DataType.FP32);
        tensor.materialize();
        assertTrue(tensor.isMaterialized());
    }
}
