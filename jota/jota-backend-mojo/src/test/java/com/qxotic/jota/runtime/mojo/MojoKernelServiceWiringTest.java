package com.qxotic.jota.runtime.mojo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.testutil.ConfiguredTestDevice;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MojoKernelServiceWiringTest {

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MojoRuntime.isAvailable(), "libjota_mojo.so is not available");
        Assumptions.assumeTrue(
                ConfiguredTestDevice.hasRuntime(DeviceType.HIP), "HIP runtime is unavailable");
    }

    @Test
    void mojoRuntimeUsesMojoOwnedKernelStores() {
        MojoDeviceRuntime runtime = new MojoDeviceRuntime();
        KernelService kernelService = runtime.kernelService().orElseThrow();

        Path expectedRoot =
                KernelCachePaths.programRoot(ConfiguredTestDevice.resolve(DeviceType.MOJO));
        assertEquals(expectedRoot.resolve("source"), kernelService.sourceStore().root());
        assertEquals(expectedRoot.resolve("binary"), kernelService.binaryStore().root());
        assertTrue(kernelService.backend() instanceof MojoKernelBackend);
    }
}
