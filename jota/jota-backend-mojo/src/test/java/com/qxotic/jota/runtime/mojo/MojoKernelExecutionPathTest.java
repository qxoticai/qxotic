package com.qxotic.jota.runtime.mojo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.tensor.Tensor;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MojoKernelExecutionPathTest {

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MojoRuntime.isAvailable(), "libjota_mojo.so is not available");
        Assumptions.assumeTrue(
                Environment.current().runtimes().hasRuntime(Device.HIP),
                "HIP runtime is unavailable");
    }

    @Test
    void executesThroughMojoKernelServiceRegistry() {
        DeviceRuntime runtime = Environment.current().runtimeFor(Device.MOJO);
        KernelService kernelService = runtime.kernelService().orElseThrow();

        Tensor left = Tensor.iota(64, DataType.FP32).to(Device.MOJO);
        Tensor right = Tensor.iota(64, DataType.FP32).to(Device.MOJO);
        Tensor out = left.add(right);
        out.materialize();

        assertTrue(out.device().equals(Device.MOJO) || out.device().equals(Device.HIP));

        Map<String, ?> named = kernelService.namedKernelKeys();
        assertFalse(named.isEmpty(), "Expected Mojo kernel service to bind at least one kernel");
        String anyKernelName = named.keySet().iterator().next();
        KernelCacheKey key = kernelService.namedKernelKeys().get(anyKernelName);
        KernelProgram source = kernelService.loadRegisteredSource(key).orElseThrow();
        assertEquals("mojo", source.language());
        assertTrue(runtime.loadRegisteredExecutable(anyKernelName).isPresent());
    }
}
