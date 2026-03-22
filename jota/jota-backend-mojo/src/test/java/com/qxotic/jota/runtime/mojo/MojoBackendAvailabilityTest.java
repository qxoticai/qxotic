package com.qxotic.jota.runtime.mojo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MojoBackendAvailabilityTest {

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MojoRuntime.isAvailable(), "libjota_mojo.so is not available");
        Assumptions.assumeTrue(
                ConfiguredTestDevice.hasRuntime(DeviceType.HIP), "HIP runtime is unavailable");
        Assumptions.assumeTrue(
                ConfiguredTestDevice.hasRuntime(DeviceType.MOJO),
                "Mojo runtime is unavailable (toolchain/compiler/runtime missing)");
    }

    @Test
    void mojoBackendRuntimeIsRegisteredAndAddressable() {
        assertTrue(ConfiguredTestDevice.hasRuntime(DeviceType.MOJO));
        assertEquals(
                DeviceType.MOJO.deviceIndex(0),
                Environment.runtimeFor(ConfiguredTestDevice.resolve(DeviceType.MOJO))
                        .device());

        Tensor onMojo =
                Tensor.iota(16, DataType.FP32).to(ConfiguredTestDevice.resolve(DeviceType.MOJO));
        assertEquals(DeviceType.MOJO.deviceIndex(0), onMojo.device());
    }
}
