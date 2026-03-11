package com.qxotic.jota.runtime.mojo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.tensor.Tensor;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MojoBackendAvailabilityTest {

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MojoRuntime.isAvailable(), "libjota_mojo.so is not available");
        Assumptions.assumeTrue(
                Environment.current().runtimes().hasRuntime(Device.HIP),
                "HIP runtime is unavailable");
    }

    @Test
    void mojoBackendRuntimeIsRegisteredAndAddressable() {
        assertTrue(Environment.current().runtimes().hasRuntime(Device.MOJO));
        assertEquals(Device.MOJO, Environment.current().runtimeFor(Device.MOJO).device());

        Tensor onMojo = Tensor.iota(16, DataType.FP32).to(Device.MOJO);
        assertEquals(Device.MOJO, onMojo.device());
    }
}
