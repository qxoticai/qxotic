package ai.qxotic.jota.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.tensor.BinaryOp;
import ai.qxotic.jota.tensor.ExpressionGraph;
import ai.qxotic.jota.tensor.GraphHasher;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelProgram;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

final class HipKernelCompiler {

    private static final String ENV_HIPCC = "HIPCC";
    private static final Path KERNEL_ROOT = Path.of("__kernels").resolve(Device.HIP.leafName());

    HipKernelSpec compile(ExpressionGraph graph) {
        KernelCacheKey key = GraphHasher.hash(graph);
        String kernelName = "hip_fused_" + key.value().substring(0, 12);

        Path kernelDir = KERNEL_ROOT.resolve(key.value());
        Path sourcePath = kernelDir.resolve(kernelName + ".hip");
        Path hsacoPath = kernelDir.resolve(kernelName + ".hsaco");

        ensureDirectory(kernelDir);

        HipElementwiseKernelGenerator generator = new HipElementwiseKernelGenerator();
        HipElementwiseKernelGenerator.GeneratedKernel generated =
                generator.generate(graph, kernelName);
        String source = generated.source();
        writeIfChanged(sourcePath, source);
        if (needsCompile(sourcePath, hsacoPath)) {
            compileSource(sourcePath, hsacoPath);
        }
        return new HipKernelSpec(hsacoPath, kernelName);
    }

    HipKernelSpec compile(KernelProgram program, KernelCacheKey key) {
        if (program.kind() != KernelProgram.Kind.SOURCE) {
            throw new UnsupportedOperationException("HIP compiler expects source program");
        }
        String kernelName = program.entryPoint();
        Path kernelDir = KERNEL_ROOT.resolve(key.value());
        Path sourcePath = kernelDir.resolve(kernelName + ".hip");
        Path hsacoPath = kernelDir.resolve(kernelName + ".hsaco");

        ensureDirectory(kernelDir);

        String source = (String) program.payload();
        writeIfChanged(sourcePath, source);
        if (needsCompile(sourcePath, hsacoPath)) {
            compileSource(sourcePath, hsacoPath);
        }
        return new HipKernelSpec(hsacoPath, kernelName);
    }

    HipKernelSpec compileMetadata(BinaryOp op) {
        String opName = op.name().toLowerCase(Locale.ROOT);
        String kernelName = "hip_" + opName + "_meta";
        Path kernelDir = KERNEL_ROOT.resolve("hip_meta").resolve(opName);
        Path sourcePath = kernelDir.resolve(kernelName + ".hip");
        Path hsacoPath = kernelDir.resolve(kernelName + ".hsaco");

        ensureDirectory(kernelDir);
        String source = metadataSourceFor(kernelName, op);
        writeIfChanged(sourcePath, source);
        if (needsCompile(sourcePath, hsacoPath)) {
            compileSource(sourcePath, hsacoPath);
        }
        return new HipKernelSpec(hsacoPath, kernelName);
    }

    private static void ensureDirectory(Path dir) {
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to create HIP kernel directory: " + dir, e);
        }
    }

    private static boolean needsCompile(Path source, Path hsaco) {
        if (!Files.exists(hsaco)) {
            return true;
        }
        try {
            return Files.getLastModifiedTime(source).toMillis()
                    > Files.getLastModifiedTime(hsaco).toMillis();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to stat HIP kernel files", e);
        }
    }

    private static void writeIfChanged(Path path, String content) {
        try {
            if (Files.exists(path)) {
                String existing = Files.readString(path);
                if (existing.equals(content)) {
                    return;
                }
            }
            Files.writeString(path, content);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write HIP kernel source: " + path, e);
        }
    }

    private static void compileSource(Path source, Path hsaco) {
        String hipcc = System.getenv(ENV_HIPCC);
        if (hipcc == null || hipcc.isBlank()) {
            hipcc = "hipcc";
        }
        ProcessBuilder builder =
                new ProcessBuilder(
                        hipcc, "--genco", "-O2", source.toString(), "-o", hsaco.toString());
        builder.inheritIO();
        try {
            Process process = builder.start();
            int code = process.waitFor();
            if (code != 0) {
                throw new IllegalStateException(
                        "HIP kernel compilation failed (exit " + code + ")");
            }
        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("HIP kernel compilation failed", e);
        }
    }

    private static String metadataSourceFor(String kernelName, BinaryOp op) {
        String operator;
        if (op == BinaryOp.ADD) {
            operator = "+";
        } else if (op == BinaryOp.SUBTRACT) {
            operator = "-";
        } else if (op == BinaryOp.MULTIPLY) {
            operator = "*";
        } else if (op == BinaryOp.DIVIDE) {
            operator = "/";
        } else {
            throw new UnsupportedOperationException("Unsupported op for HIP kernel: " + op);
        }
        return "#include <hip/hip_runtime.h>\n"
                + "extern \"C\" __global__ void "
                + kernelName
                + "(const float *a, const float *b, float *out, const long *meta) {\n"
                + "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                + "  long n = meta[0];\n"
                + "  if (idx < n) {\n"
                + "    out[idx] = a[idx] "
                + operator
                + " b[idx];\n"
                + "  }\n"
                + "}\n";
    }
}
