package com.qxotic.jota.runtime.c;

import com.qxotic.jota.Device;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRTextRenderer;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelProgram;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

final class CKernelCompiler {

    private static final ConcurrentHashMap<Path, Object> COMPILE_LOCKS = new ConcurrentHashMap<>();

    private static final boolean KERNEL_LOG = Boolean.getBoolean("jota.kernel.log");
    private static final long COMPILE_TIMEOUT_SECONDS =
            Long.getLong("jota.c.compile.timeout.seconds", 10L);
    private static final String OPT_LEVEL = System.getProperty("jota.c.compile.opt", "2").trim();
    private static final String COMPILER_PROPERTY = "jota.c.compiler";
    private static final String EXTRA_COMPILE_FLAGS_PROPERTY = "jota.c.compile.extraFlags";
    private static final String EXTRA_LINK_FLAGS_PROPERTY = "jota.c.link.extraFlags";

    KernelCacheKey cacheKey(LIRGraph graph, ScratchLayout scratchLayout) {
        String hash = hashLirGraph(graph, scratchLayout);
        return KernelCacheKey.of(hash + "-c-lir-v1");
    }

    Path compile(KernelProgram program, KernelCacheKey key) {
        if (program.kind() != KernelProgram.Kind.SOURCE) {
            throw new IllegalArgumentException("C compiler expects source program");
        }
        String kernelName = program.entryPoint();
        Path baseDir = deviceRoot().resolve(key.value());
        Path sourcePath = baseDir.resolve(kernelName + ".c");
        Path soPath = baseDir.resolve(sharedLibraryName(kernelName));
        log("C kernel compile key=" + key.value() + " entry=" + kernelName);
        try {
            Files.createDirectories(baseDir);
            writeIfChanged(sourcePath, program.payload().toString());
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write C kernel source", e);
        }
        if (needsCompile(sourcePath, soPath)) {
            compileWithLock(sourcePath, soPath);
        } else {
            log("C kernel reuse key=" + key.value() + " entry=" + kernelName);
        }
        return soPath;
    }

    private static void compileWithLock(Path sourcePath, Path soPath) {
        Object lock = COMPILE_LOCKS.computeIfAbsent(soPath, ignored -> new Object());
        synchronized (lock) {
            if (needsCompile(sourcePath, soPath)) {
                compileSource(sourcePath, soPath);
            } else {
                log("C kernel reuse entry=" + soPath.getFileName());
            }
        }
    }

    private static boolean needsCompile(Path source, Path soPath) {
        if (!Files.exists(soPath)) {
            return true;
        }
        try {
            return Files.getLastModifiedTime(source).toMillis()
                    > Files.getLastModifiedTime(soPath).toMillis();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to stat C kernel files", e);
        }
    }

    private static void writeIfChanged(Path path, String content) throws IOException {
        if (Files.exists(path)) {
            String existing = Files.readString(path);
            if (existing.equals(content)) {
                return;
            }
        }
        Files.writeString(path, content, StandardCharsets.UTF_8);
    }

    private static void compileSource(Path source, Path soPath) {
        String compiler = compilerCommand();
        Path tmpSoPath = soPath.resolveSibling(soPath.getFileName() + ".tmp");
        List<String> command = new ArrayList<>();
        command.add(compiler);
        command.addAll(sharedFlags());
        if (openMpEnabled()) {
            command.addAll(openMpCompileFlags());
        }
        command.addAll(extraCompileFlags());
        command.add("-O" + OPT_LEVEL);
        command.add("-std=gnu17");
        command.add(source.toAbsolutePath().toString());
        command.add("-o");
        command.add(tmpSoPath.toAbsolutePath().toString());
        if (!COpenMpConfig.isWindows()) {
            command.add("-lm");
        }
        if (openMpEnabled()) {
            command.addAll(openMpLinkFlags());
        }
        command.addAll(extraLinkFlags());
        if (!COpenMpConfig.isMac() && !COpenMpConfig.isWindows()) {
            command.add("-ldl");
        }

        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        try {
            Process process = builder.start();
            if (!process.waitFor(COMPILE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(1, TimeUnit.SECONDS);
                throw new IllegalStateException(
                        "C kernel compilation timed out after " + COMPILE_TIMEOUT_SECONDS + "s");
            }
            int code = process.exitValue();
            if (code != 0) {
                String output =
                        new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
                throw new IllegalStateException(
                        "C kernel compilation failed (exit "
                                + code
                                + ")\ncommand: "
                                + String.join(" ", command)
                                + (output.isBlank() ? "" : "\ncompiler output:\n" + output));
            }
            moveAtomically(tmpSoPath, soPath);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("C kernel compilation failed", e);
        } catch (IOException e) {
            throw new IllegalStateException("C kernel compilation failed", e);
        } finally {
            try {
                Files.deleteIfExists(tmpSoPath);
            } catch (IOException ignored) {
                // best effort cleanup
            }
        }
    }

    private static void moveAtomically(Path source, Path target) throws IOException {
        try {
            Files.move(
                    source,
                    target,
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static String compilerCommand() {
        String property = System.getProperty(COMPILER_PROPERTY);
        if (property != null && !property.isBlank()) {
            return property.trim();
        }
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
        if (COpenMpConfig.isMac()) {
            return List.of("-dynamiclib", "-fPIC");
        }
        if (COpenMpConfig.isWindows()) {
            return List.of("-shared");
        }
        return List.of("-shared", "-fPIC");
    }

    private static String sharedLibraryName(String base) {
        return System.mapLibraryName(base);
    }

    private static boolean openMpEnabled() {
        return COpenMpConfig.enabled();
    }

    private static List<String> openMpCompileFlags() {
        String override = System.getProperty("jota.c.openmp.compileFlags");
        if (override != null && !override.isBlank()) {
            return splitFlags(override);
        }
        if (COpenMpConfig.isMac()) {
            Path prefix = COpenMpConfig.detectBrewLibOmpPrefix();
            if (prefix == null) {
                return List.of("-Xpreprocessor", "-fopenmp");
            }
            return List.of(
                    "-Xpreprocessor",
                    "-fopenmp",
                    "-I" + prefix.resolve("include").toAbsolutePath());
        }
        return List.of("-fopenmp");
    }

    private static List<String> openMpLinkFlags() {
        String override = System.getProperty("jota.c.openmp.linkFlags");
        if (override != null && !override.isBlank()) {
            return splitFlags(override);
        }
        if (COpenMpConfig.isMac()) {
            Path prefix = COpenMpConfig.detectBrewLibOmpPrefix();
            if (prefix == null) {
                return List.of("-lomp");
            }
            Path libDir = prefix.resolve("lib").toAbsolutePath();
            return List.of("-L" + libDir, "-lomp", "-Wl,-rpath," + libDir);
        }
        return List.of();
    }

    private static List<String> splitFlags(String flags) {
        String[] parts = flags.trim().split("\\s+");
        List<String> result = new ArrayList<>(parts.length);
        for (String part : parts) {
            if (!part.isBlank()) {
                result.add(part);
            }
        }
        return List.copyOf(result);
    }

    private static List<String> extraCompileFlags() {
        String flags = System.getProperty(EXTRA_COMPILE_FLAGS_PROPERTY);
        if (flags == null || flags.isBlank()) {
            return List.of();
        }
        return splitFlags(flags);
    }

    private static List<String> extraLinkFlags() {
        String flags = System.getProperty(EXTRA_LINK_FLAGS_PROPERTY);
        if (flags == null || flags.isBlank()) {
            return List.of();
        }
        return splitFlags(flags);
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
        text.append("\n// openmp: ").append(openMpEnabled());
        text.append("\n// compiler: ").append(compilerCommand());
        text.append("\n// opt-level: ").append(OPT_LEVEL);
        text.append("\n// extra-compile-flags: ").append(String.join(" ", extraCompileFlags()));
        text.append("\n// extra-link-flags: ").append(String.join(" ", extraLinkFlags()));
        if (openMpEnabled()) {
            text.append("\n// openmp-compile-flags: ")
                    .append(String.join(" ", openMpCompileFlags()));
            text.append("\n// openmp-link-flags: ").append(String.join(" ", openMpLinkFlags()));
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

    private static void log(String message) {
        if (KERNEL_LOG) {
            System.out.println("[jota-kernel] " + message);
        }
    }

    private static Path deviceRoot() {
        return KernelCachePaths.deviceRoot(Device.C);
    }
}
