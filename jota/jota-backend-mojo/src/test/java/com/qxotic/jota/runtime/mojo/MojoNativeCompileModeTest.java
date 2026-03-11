package com.qxotic.jota.runtime.mojo;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.mojo.bridge.MojoRuntime;
import com.qxotic.jota.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MojoNativeCompileModeTest {

    private String previousMode;

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MojoRuntime.isAvailable(), "libjota_mojo.so is not available");
        Assumptions.assumeTrue(
                Environment.current().runtimes().hasRuntime(Device.HIP),
                "HIP runtime is unavailable");
        Assumptions.assumeTrue(isCommandAvailable("mojo", "--version"), "mojo is unavailable");
    }

    @AfterEach
    void tearDown() {
        if (previousMode == null) {
            System.clearProperty("jota.mojo.native.compile.mode");
            return;
        }
        System.setProperty("jota.mojo.native.compile.mode", previousMode);
    }

    @Test
    void nativeModeProducesMojoDerivedCodeObjectArtifacts() {
        previousMode = System.getProperty("jota.mojo.native.compile.mode");
        System.setProperty("jota.mojo.native.compile.mode", "native");

        DeviceRuntime runtime = Environment.current().runtimeFor(Device.MOJO);
        KernelService kernelService = runtime.kernelService().orElseThrow();

        Tensor left = Tensor.iota(513, DataType.FP32).to(Device.MOJO);
        Tensor right = Tensor.iota(513, DataType.FP32).to(Device.MOJO);
        left.add(right).materialize();

        assertTrue(!kernelService.namedKernelKeys().isEmpty(), "Expected a registered Mojo kernel");
        String kernelName = kernelService.namedKernelKeys().keySet().iterator().next();
        KernelCacheKey key = kernelService.namedKernelKeys().get(kernelName);
        assertTrue(key != null, "Expected a cache key for native-compiled kernel");

        Path mojoPath = MojoCachePaths.lirSourcePath(key.value());
        Path wrapperPath = MojoCachePaths.lirWrapperPath(key.value());
        Path asmPath = MojoCachePaths.lirAsmPath(key.value());
        Path binaryPath = MojoCachePaths.lirBinaryPath(key.value());
        Path entryPath = MojoCachePaths.lirEntryPath(key.value());

        assertTrue(Files.exists(mojoPath), "Expected Mojo source: " + mojoPath);
        assertTrue(Files.exists(wrapperPath), "Expected Mojo wrapper: " + wrapperPath);
        assertTrue(Files.exists(asmPath), "Expected Mojo asm output: " + asmPath);
        assertTrue(Files.exists(binaryPath), "Expected persisted Mojo ELF binary: " + binaryPath);
        assertTrue(Files.exists(entryPath), "Expected persisted Mojo entrypoint: " + entryPath);
    }

    private static boolean isCommandAvailable(String command, String arg) {
        ProcessBuilder pb = new ProcessBuilder(List.of(command, arg));
        pb.redirectErrorStream(true);
        try {
            Process process = pb.start();
            boolean finished = process.waitFor(2, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                return false;
            }
            return process.exitValue() == 0;
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return false;
        }
    }
}
