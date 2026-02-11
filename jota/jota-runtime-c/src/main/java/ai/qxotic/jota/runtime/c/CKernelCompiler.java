package ai.qxotic.jota.runtime.c;

import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRTextRenderer;
import ai.qxotic.jota.ir.lir.scratch.ScratchLayout;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelProgram;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.Locale;

final class CKernelCompiler {

    KernelCacheKey cacheKey(LIRGraph graph, ScratchLayout scratchLayout) {
        String hash = hashLirGraph(graph, scratchLayout);
        return KernelCacheKey.of(hash + "-c-lir-v1");
    }

    CKernelSpec compile(KernelProgram program, KernelCacheKey key) {
        if (program.kind() != KernelProgram.Kind.SOURCE) {
            throw new IllegalArgumentException("C compiler expects source program");
        }
        String kernelName = program.entryPoint();
        Path baseDir = Path.of("__kernels", "c", key.value());
        Path sourcePath = baseDir.resolve(kernelName + ".c");
        Path soPath = baseDir.resolve(sharedLibraryName(kernelName));
        try {
            Files.createDirectories(baseDir);
            Files.writeString(sourcePath, program.payload().toString(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write C kernel source", e);
        }
        compileSource(sourcePath, soPath);
        return new CKernelSpec(soPath, kernelName);
    }

    private static void compileSource(Path source, Path soPath) {
        String compiler = compilerCommand();
        List<String> command = new java.util.ArrayList<>();
        command.add(compiler);
        command.addAll(sharedFlags());
        if (openMpEnabled()) {
            command.add("-fopenmp");
        }
        command.add("-O2");
        command.add("-std=gnu17");
        command.add(source.toAbsolutePath().toString());
        command.add("-o");
        command.add(soPath.toAbsolutePath().toString());
        if (!isWindows()) {
            command.add("-lm");
        }
        if (!isMac() && !isWindows()) {
            command.add("-ldl");
        }

        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        try {
            Process process = builder.start();
            String output =
                    new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            int code = process.waitFor();
            if (code != 0) {
                throw new IllegalStateException(
                        "C kernel compilation failed (exit " + code + ")\n" + output);
            }
        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("C kernel compilation failed", e);
        }
    }

    private static String compilerCommand() {
        String override = System.getenv("JOTA_C_CC");
        if (override != null && !override.isBlank()) {
            return override;
        }
        String cc = System.getenv("CC");
        if (cc != null && !cc.isBlank()) {
            return cc;
        }
        return "gcc"; // or clang
    }

    private static List<String> sharedFlags() {
        if (isMac()) {
            return List.of("-dynamiclib", "-fPIC");
        }
        if (isWindows()) {
            return List.of("-shared");
        }
        return List.of("-shared", "-fPIC");
    }

    private static String sharedLibraryName(String base) {
        return System.mapLibraryName(base);
    }

    private static boolean isMac() {
        return System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("mac");
    }

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win");
    }

    private static boolean openMpEnabled() {
        String override = System.getProperty("ai.qxotic.jota.c.openmp");
        if (override != null) {
            return Boolean.parseBoolean(override);
        }
        return !isMac() && !isWindows();
    }

    private static String hashLirGraph(LIRGraph graph, ScratchLayout scratchLayout) {
        LIRTextRenderer renderer = new LIRTextRenderer();
        StringBuilder text = new StringBuilder(renderer.render(graph));
        if (scratchLayout.requiresScratch()) {
            text.append("\n// scratch: ").append(scratchLayout.totalByteSize());
            scratchLayout
                    .offsets()
                    .forEach(
                            (buf, offset) ->
                                    text.append("\n// buf ")
                                            .append(buf.id())
                                            .append(": offset=")
                                            .append(offset));
        }
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = text.toString().getBytes(StandardCharsets.UTF_8);
            byte[] hashed = digest.digest(bytes);
            StringBuilder builder = new StringBuilder(hashed.length * 2);
            for (byte value : hashed) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            return builder.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }
}
