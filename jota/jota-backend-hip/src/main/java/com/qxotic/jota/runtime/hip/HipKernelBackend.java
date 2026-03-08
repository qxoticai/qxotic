package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.Device;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRTextRenderer;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

final class HipKernelBackend implements KernelBackend {

    private static final String ENV_HIPCC = "HIPCC";
    private static final String COMPILER_PROPERTY = "jota.hip.compiler";
    private static final String EXTRA_FLAGS_PROPERTY = "jota.hip.compile.flags";
    private static final Path KERNEL_ROOT = KernelCachePaths.deviceRoot(Device.HIP);
    private static final boolean KERNEL_LOG = Boolean.getBoolean("jota.kernel.log");
    private static final long COMPILE_TIMEOUT_SECONDS =
            Long.getLong("jota.hip.compile.timeout.seconds", 10L);
    private static final String OPT_LEVEL =
            System.getProperty("jota.hip.compile.opt", "2").trim();
    private static final String ARCH = resolveArch();
    private static final boolean TIMING_LOG = Boolean.getBoolean("jota.hip.timing.log");

    private final KernelExecutableCache cache = new InMemoryKernelCache();

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        long t0 = System.nanoTime();
        if (!KernelProgram.HIP.equals(program.language())) {
            throw new UnsupportedOperationException("HIP backend expects HIP programs");
        }
        if (program.kind() != KernelProgram.Kind.SOURCE) {
            throw new UnsupportedOperationException("HIP compile expects source program");
        }
        String kernelName = program.entryPoint();
        Path hsacoPath = compileSourceProgram(program, cacheKey);
        long tCompiled = System.nanoTime();
        byte[] hsaco = loadHsaco(hsacoPath);
        long tLoadedBinary = System.nanoTime();
        HipModule module = HipModule.load(hsaco);
        long tLoadModule = System.nanoTime();
        HipKernelExecutable exec = new HipKernelExecutable(module.function(kernelName));
        long tResolveFunction = System.nanoTime();
        logTiming("compile", cacheKey, t0, tCompiled, tLoadedBinary, tLoadModule, tResolveFunction);
        return new KernelExecutable() {
            @Override
            public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
                exec.launch(config, args, stream);
            }

            @Override
            public void close() {
                module.close();
            }
        };
    }

    @Override
    public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
        long t0 = System.nanoTime();
        if (program.kind() != KernelProgram.Kind.BINARY) {
            throw new UnsupportedOperationException("HIP load expects binary program");
        }
        byte[] hsaco = (byte[]) program.payload();
        long tLoadedBinary = System.nanoTime();
        HipModule module = HipModule.load(hsaco);
        long tLoadModule = System.nanoTime();
        HipKernelExecutable exec = new HipKernelExecutable(module.function(program.entryPoint()));
        long tResolveFunction = System.nanoTime();
        logTiming(
                "load", cacheKey, t0, tLoadedBinary, tLoadedBinary, tLoadModule, tResolveFunction);
        return new KernelExecutable() {
            @Override
            public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
                exec.launch(config, args, stream);
            }

            @Override
            public void close() {
                module.close();
            }
        };
    }

    @Override
    public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey) {
        KernelExecutable exec = cache.get(cacheKey);
        if (exec != null) {
            log("HIP kernel cache hit key=" + cacheKey.value());
            return exec;
        }
        log("HIP kernel cache miss key=" + cacheKey.value());
        KernelExecutable created =
                program.kind() == KernelProgram.Kind.BINARY
                        ? load(program, cacheKey)
                        : compile(program, cacheKey);
        cache.put(cacheKey, created);
        return created;
    }

    KernelCacheKey cacheKey(LIRGraph graph, ScratchLayout scratchLayout) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(new LIRTextRenderer().render(graph).getBytes(StandardCharsets.UTF_8));
            digest.update(
                    Long.toString(scratchLayout.totalByteSize()).getBytes(StandardCharsets.UTF_8));
            scratchLayout.offsets().entrySet().stream()
                    .sorted(Comparator.comparingInt(entry -> entry.getKey().id()))
                    .forEach(
                            entry -> {
                                digest.update(
                                        Integer.toString(entry.getKey().id())
                                                .getBytes(StandardCharsets.UTF_8));
                                digest.update(
                                        Long.toString(entry.getValue())
                                                .getBytes(StandardCharsets.UTF_8));
                            });
            String hipcc = System.getenv(ENV_HIPCC);
            if (hipcc == null || hipcc.isBlank()) {
                hipcc = "hipcc";
            }
            digest.update(resolveHipcc(hipcc).getBytes(StandardCharsets.UTF_8));
            digest.update(OPT_LEVEL.getBytes(StandardCharsets.UTF_8));
            digest.update(String.join(" ", extraCompileFlags()).getBytes(StandardCharsets.UTF_8));
            if (ARCH != null) {
                digest.update(ARCH.getBytes(StandardCharsets.UTF_8));
            }
            byte[] hashed = digest.digest();
            StringBuilder builder = new StringBuilder(hashed.length * 2 + 11);
            for (byte value : hashed) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            builder.append("-hip-lir-v1");
            return KernelCacheKey.of(builder.toString());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    @Override
    public KernelExecutableCache cache() {
        return cache;
    }

    private static final class InMemoryKernelCache implements KernelExecutableCache {
        private final Map<KernelCacheKey, KernelExecutable> map = new ConcurrentHashMap<>();

        @Override
        public KernelExecutable get(KernelCacheKey key) {
            return map.get(key);
        }

        @Override
        public void put(KernelCacheKey key, KernelExecutable exec) {
            map.put(key, exec);
        }
    }

    private Path compileSourceProgram(KernelProgram program, KernelCacheKey key) {
        String kernelName = program.entryPoint();
        Path kernelDir = KERNEL_ROOT.resolve(key.value());
        Path sourcePath = kernelDir.resolve(kernelName + ".hip");
        Path hsacoPath = kernelDir.resolve(kernelName + ".hsaco");
        ensureDirectory(kernelDir);
        String source = requireSource(program.payload());
        long t0 = System.nanoTime();
        writeIfChanged(sourcePath, source);
        long tWrite = System.nanoTime();
        if (needsCompile(sourcePath, hsacoPath)) {
            log("HIP kernel compile key=" + key.value() + " entry=" + kernelName);
            compileSource(sourcePath, hsacoPath);
            long tCompile = System.nanoTime();
            logCompileTiming(key, kernelName, true, t0, tWrite, tCompile);
        } else {
            log("HIP kernel reuse key=" + key.value() + " entry=" + kernelName);
            long tReuse = System.nanoTime();
            logCompileTiming(key, kernelName, false, t0, tWrite, tReuse);
        }
        return hsacoPath;
    }

    private static byte[] loadHsaco(Path hsacoPath) {
        try {
            return Files.readAllBytes(hsacoPath);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read HSACO: " + hsacoPath, e);
        }
    }

    private static void logCompileTiming(
            KernelCacheKey key,
            String kernelName,
            boolean compiled,
            long t0,
            long tWrite,
            long tEnd) {
        if (!TIMING_LOG) {
            return;
        }
        System.out.println(
                "[jota-hip-timing] compileStage key="
                        + key.value()
                        + " entry="
                        + kernelName
                        + " compiled="
                        + compiled
                        + " writeSourceMs="
                        + ms(tWrite - t0)
                        + " compileOrReuseMs="
                        + ms(tEnd - tWrite)
                        + " totalMs="
                        + ms(tEnd - t0));
    }

    private static void logTiming(
            String phase,
            KernelCacheKey key,
            long t0,
            long tCompiled,
            long tLoadedBinary,
            long tLoadModule,
            long tResolveFunction) {
        if (!TIMING_LOG) {
            return;
        }
        System.out.println(
                "[jota-hip-timing] backendPhase="
                        + phase
                        + " key="
                        + key.value()
                        + " compileOrPrepareMs="
                        + ms(tCompiled - t0)
                        + " loadBinaryMs="
                        + ms(tLoadedBinary - tCompiled)
                        + " loadModuleMs="
                        + ms(tLoadModule - tLoadedBinary)
                        + " resolveFunctionMs="
                        + ms(tResolveFunction - tLoadModule)
                        + " totalMs="
                        + ms(tResolveFunction - t0));
    }

    private static String ms(long nanos) {
        return String.format(Locale.ROOT, "%.3f", nanos / 1_000_000.0);
    }

    private static String requireSource(Object payload) {
        if (payload instanceof String source) {
            return source;
        }
        throw new IllegalArgumentException("Expected HIP source payload as String");
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
        List<String> command = new ArrayList<>();
        command.add(resolveHipcc(hipcc));
        command.add("--genco");
        command.add("-O" + OPT_LEVEL);
        if (ARCH != null && !ARCH.isBlank()) {
            command.add("--offload-arch=" + ARCH);
        }
        command.addAll(extraCompileFlags());
        command.add(source.toString());
        command.add("-o");
        command.add(hsaco.toString());
        ProcessBuilder builder = new ProcessBuilder(command);
        builder.inheritIO();
        try {
            Process process = builder.start();
            if (!process.waitFor(COMPILE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(1, TimeUnit.SECONDS);
                throw new IllegalStateException(
                        "HIP kernel compilation timed out after " + COMPILE_TIMEOUT_SECONDS + "s");
            }
            int code = process.exitValue();
            if (code != 0) {
                throw new IllegalStateException(
                        "HIP kernel compilation failed (exit " + code + ")");
            }
        } catch (IOException e) {
            throw new IllegalStateException("HIP kernel compilation failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("HIP kernel compilation interrupted", e);
        }
    }

    private static String resolveArch() {
        String prop = System.getProperty("jota.hip.arch");
        if (prop != null && !prop.isBlank()) {
            return prop.trim();
        }
        String env = System.getenv("JOTA_HIP_ARCH");
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        return null;
    }

    private static String resolveHipcc(String fallbackFromEnv) {
        String prop = System.getProperty(COMPILER_PROPERTY);
        if (prop != null && !prop.isBlank()) {
            return prop.trim();
        }
        return fallbackFromEnv;
    }

    private static List<String> extraCompileFlags() {
        String flags = System.getProperty(EXTRA_FLAGS_PROPERTY);
        if (flags == null || flags.isBlank()) {
            return List.of();
        }
        String[] parts = flags.trim().split("\\s+");
        List<String> result = new ArrayList<>(parts.length);
        for (String part : parts) {
            if (!part.isBlank()) {
                result.add(part);
            }
        }
        return List.copyOf(result);
    }

    private static void log(String message) {
        if (KERNEL_LOG) {
            System.out.println("[jota-kernel] " + message);
        }
    }
}
