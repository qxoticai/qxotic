package com.qxotic.jota.runtime.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.TIRToLIRLowerer;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRStandardPipeline;
import com.qxotic.jota.ir.lir.scratch.ScratchAnalysisPass;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.tir.KernelStep;
import com.qxotic.jota.ir.tir.ScheduledProgram;
import com.qxotic.jota.ir.tir.TIRCSEPass;
import com.qxotic.jota.ir.tir.TIRConstantFoldingPass;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRSchedulePass;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.TensorTracing;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class CudaArgReductionSourceCompileTest {

    private static final String ARCH_PROPERTY = "jota.cuda.arch";
    private static final String ARCH_ENV = "JOTA_CUDA_ARCH";
    private static final long TIMEOUT_SECONDS = 20L;

    @Test
    void tracedArgmaxKernelCompilesToPtx() throws IOException, InterruptedException {
        Tensor input = Tensor.of(new float[] {1f, 5f, 2f, 9f, 3f, 4f}).view(Shape.of(2, 3));
        Tensor traced = Tracer.trace(input, t -> t.argmax(1));
        assertTracedKernelCompilesToPtx(traced, "argmax_compile_kernel");
    }

    @Test
    void tracedArgminKernelCompilesToPtx() throws IOException, InterruptedException {
        Tensor input = Tensor.of(new float[] {1f, 5f, 2f, 9f, 3f, 4f}).view(Shape.of(2, 3));
        Tensor traced = Tracer.trace(input, t -> t.argmin(1));
        assertTracedKernelCompilesToPtx(traced, "argmin_compile_kernel");
    }

    private static void assertTracedKernelCompilesToPtx(Tensor traced, String kernelName)
            throws IOException, InterruptedException {
        String nvcc = CudaKernelBackend.resolveNvccExecutable();
        Assumptions.assumeTrue(
                ExternalToolChecks.hasCommand(nvcc, "--version"), "nvcc is not available");

        TIRGraph tirGraph =
                TensorTracing.tracedGraph(traced)
                        .orElseThrow(() -> new IllegalStateException("Expected traced IR graph"));
        TIRGraph optimizedGraph = new TIRConstantFoldingPass().run(new TIRCSEPass().run(tirGraph));
        ScheduledProgram schedule = new TIRSchedulePass().run(optimizedGraph);
        assertTrue(!schedule.steps().isEmpty(), "Expected schedulable kernels in traced graph");

        Path tempDir = Files.createTempDirectory("jota-cuda-argreduce-");
        int stepIndex = 0;
        for (KernelStep step : schedule.steps()) {
            String stepKernelName = kernelName + "_step" + stepIndex++;
            LIRGraph lirGraph =
                    new LIRStandardPipeline().run(new TIRToLIRLowerer().lower(step.graph()));
            ScratchLayout scratchLayout = new ScratchAnalysisPass().analyze(lirGraph);
            String source = new CudaDialect().renderSource(lirGraph, scratchLayout, stepKernelName);

            Path sourcePath = tempDir.resolve(stepKernelName + ".cu");
            Path ptxPath = tempDir.resolve(stepKernelName + ".ptx");
            Files.writeString(sourcePath, source);

            List<String> command = new ArrayList<>();
            command.add(nvcc);
            command.add("-ptx");
            command.add("-O2");
            command.add("-diag-suppress=177");
            String arch = resolveArch();
            if (!arch.isBlank()) {
                command.add("-arch=" + arch);
            }
            command.add("-o");
            command.add(ptxPath.toString());
            command.add(sourcePath.toString());

            Process process =
                    new ProcessBuilder(command)
                            .redirectErrorStream(true)
                            .directory(tempDir.toFile())
                            .start();
            boolean finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new IllegalStateException("nvcc timed out for source: " + sourcePath);
            }

            String output = new String(process.getInputStream().readAllBytes());
            assertEquals(
                    0,
                    process.exitValue(),
                    "nvcc failed for generated kernel source "
                            + sourcePath
                            + "\ncommand="
                            + String.join(" ", command)
                            + "\noutput:\n"
                            + output);
            assertFalse(Files.notExists(ptxPath), "Expected PTX output file: " + ptxPath);
            assertFalse(Files.size(ptxPath) == 0, "PTX output was empty: " + ptxPath);
        }
    }

    private static String resolveArch() {
        String arch = System.getProperty(ARCH_PROPERTY);
        if (arch != null && !arch.isBlank()) {
            return arch.trim();
        }
        arch = System.getenv(ARCH_ENV);
        return arch == null ? "" : arch.trim();
    }
}
