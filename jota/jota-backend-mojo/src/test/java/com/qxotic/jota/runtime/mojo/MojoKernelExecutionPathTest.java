package com.qxotic.jota.runtime.mojo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MojoKernelExecutionPathTest {

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
    void executesThroughMojoKernelServiceRegistry() {
        DeviceRuntime runtime =
                Environment.current().runtimeFor(ConfiguredTestDevice.resolve(DeviceType.MOJO));
        KernelService kernelService = runtime.kernelService().orElseThrow();

        Tensor left =
                Tensor.iota(64, DataType.FP32).to(ConfiguredTestDevice.resolve(DeviceType.MOJO));
        Tensor right =
                Tensor.iota(64, DataType.FP32).to(ConfiguredTestDevice.resolve(DeviceType.MOJO));
        Tensor out = left.add(right);
        out.materialize();

        assertTrue(
                out.device().equals(ConfiguredTestDevice.resolve(DeviceType.MOJO))
                        || out.device().equals(ConfiguredTestDevice.resolve(DeviceType.HIP)));

        Map<String, ?> named = kernelService.namedKernelKeys();
        assertFalse(named.isEmpty(), "Expected Mojo kernel service to bind at least one kernel");
        String anyKernelName = named.keySet().iterator().next();
        KernelCacheKey key = kernelService.namedKernelKeys().get(anyKernelName);
        KernelProgram source = kernelService.loadRegisteredSource(key).orElseThrow();
        assertEquals("mojo", source.language());
        assertTrue(runtime.loadRegisteredExecutable(anyKernelName).isPresent());
    }
}
