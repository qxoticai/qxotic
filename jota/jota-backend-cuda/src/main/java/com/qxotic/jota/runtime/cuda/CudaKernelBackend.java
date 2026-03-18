package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.DeviceType;
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
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class CudaKernelBackend implements KernelBackend {

    static final String NVCC_ENV = "NVCC";
    static final String NVCC_PROPERTY = "jota.cuda.compiler";
    private static final String EXTRA_FLAGS_PROPERTY = "jota.cuda.compile.flags";
    private static final String PCH_ENABLED_PROPERTY = "jota.cuda.pch.enabled";
    private static final String PCH_LOG_PROPERTY = "jota.cuda.pch.log";
    private static final Path KERNEL_ROOT = KernelCachePaths.deviceRoot(DeviceType.CUDA);
    private static final boolean KERNEL_LOG = Boolean.getBoolean("jota.kernel.log");
    private static final long COMPILE_TIMEOUT_SECONDS =
            Long.getLong("jota.cuda.compile.timeout.seconds", 10L);
    private static final String OPT_LEVEL = System.getProperty("jota.cuda.compile.opt", "2").trim();
    private static final boolean TIMING_LOG = Boolean.getBoolean("jota.cuda.timing.log");
    private static final boolean PCH_ENABLED =
            Boolean.parseBoolean(System.getProperty(PCH_ENABLED_PROPERTY, "false"));
    private static final boolean PCH_LOG = Boolean.getBoolean(PCH_LOG_PROPERTY);
    private static final String PCH_PREAMBLE =
            "#include <cuda_runtime.h>\n"
                    + "#include <cuda_fp16.h>\n"
                    + "#include <cuda_bf16.h>\n"
                    + "#include <stdint.h>\n"
                    + "#include <math.h>\n";
    private static final String UNUSED_SCRATCH_WARNING_SUPPRESSION = "-diag-suppress=177";
    private static final Pattern QUOTED_TOKEN = Pattern.compile("\\\"([^\\\"]*)\\\"");
    private static final List<String> EXTRA_FLAGS = parseExtraCompileFlags();
    private static final Map<String, String> NVCC_ID_CACHE = new ConcurrentHashMap<>();
    private static final Map<String, Object> PCH_LOCKS = new ConcurrentHashMap<>();

    private final KernelExecutableCache cache = new InMemoryKernelCache();
    private final Set<String> pchDisabledArchSegments = ConcurrentHashMap.newKeySet();

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        long t0 = System.nanoTime();
        if (!"cuda".equals(program.language())) {
            throw new UnsupportedOperationException("CUDA backend expects CUDA programs");
        }
        if (program.kind() != KernelProgram.Kind.SOURCE) {
            throw new UnsupportedOperationException("CUDA compile expects source program");
        }
        String kernelName = program.entryPoint();
        Path ptxPath = compileSourceProgram(program, cacheKey);
        long tCompiled = System.nanoTime();
        byte[] ptx = loadPtx(ptxPath);
        long tLoadedBinary = System.nanoTime();
        CudaModule module = CudaModule.load(ptx);
        long tLoadModule = System.nanoTime();
        CudaKernelExecutable exec = new CudaKernelExecutable(module.function(kernelName));
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
            throw new UnsupportedOperationException("CUDA load expects binary program");
        }
        byte[] ptx = (byte[]) program.payload();
        long tLoadedBinary = System.nanoTime();
        CudaModule module = CudaModule.load(ptx);
        long tLoadModule = System.nanoTime();
        CudaKernelExecutable exec = new CudaKernelExecutable(module.function(program.entryPoint()));
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
            log("CUDA kernel cache hit key=" + cacheKey.value());
            return exec;
        }
        log("CUDA kernel cache miss key=" + cacheKey.value());
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
            digest.update(resolveNvccExecutable().getBytes(StandardCharsets.UTF_8));
            digest.update(OPT_LEVEL.getBytes(StandardCharsets.UTF_8));
            digest.update(String.join(" ", EXTRA_FLAGS).getBytes(StandardCharsets.UTF_8));
            String arch = resolveArch();
            if (arch != null && !arch.isBlank()) {
                digest.update(arch.getBytes(StandardCharsets.UTF_8));
            }
            byte[] hashed = digest.digest();
            StringBuilder builder = new StringBuilder(hashed.length * 2 + 11);
            for (byte value : hashed) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            builder.append("-cuda-lir-v1");
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
        String arch = resolveArch();
        Path sourceDir = KERNEL_ROOT.resolve("source").resolve(key.value());
        Path binaryDir =
                KERNEL_ROOT.resolve("binary").resolve(cacheArchSegment(arch)).resolve(key.value());
        Path sourcePath = sourceDir.resolve(kernelName + ".cu");
        Path ptxPath = binaryDir.resolve(kernelName + ".ptx");
        ensureDirectory(sourceDir);
        ensureDirectory(binaryDir);
        String source = requireSource(program.payload());
        long t0 = System.nanoTime();
        writeIfChanged(sourcePath, source);
        long tWrite = System.nanoTime();
        if (needsCompile(sourcePath, ptxPath)) {
            log("CUDA kernel compile key=" + key.value() + " entry=" + kernelName);
            compileSource(sourcePath, ptxPath, key, arch);
            long tCompile = System.nanoTime();
            logCompileTiming(key, kernelName, true, t0, tWrite, tCompile);
        } else {
            log("CUDA kernel reuse key=" + key.value() + " entry=" + kernelName);
            long tReuse = System.nanoTime();
            logCompileTiming(key, kernelName, false, t0, tWrite, tReuse);
        }
        return ptxPath;
    }

    private static byte[] loadPtx(Path ptxPath) {
        try {
            return Files.readAllBytes(ptxPath);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read PTX: " + ptxPath, e);
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
                "[jota-cuda-timing] compileStage key="
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
                "[jota-cuda-timing] backendPhase="
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
        throw new IllegalArgumentException("Expected CUDA source payload as String");
    }

    private static void ensureDirectory(Path dir) {
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to create CUDA kernel directory: " + dir, e);
        }
    }

    private static boolean needsCompile(Path source, Path ptx) {
        if (!Files.exists(ptx)) {
            return true;
        }
        try {
            return Files.getLastModifiedTime(source).toMillis()
                    > Files.getLastModifiedTime(ptx).toMillis();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to stat CUDA kernel files", e);
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
            throw new IllegalStateException("Failed to write CUDA kernel source: " + path, e);
        }
    }

    private void compileSource(Path source, Path ptx, KernelCacheKey key, String arch) {
        String compiler = resolveNvccExecutable();
        List<String> command = buildCompileCommand(compiler, source, ptx, null, arch);
        String archSegment = cacheArchSegment(arch);
        Path pchPath = maybeResolvePch(compiler, source, ptx, arch);
        if (pchPath != null) {
            List<String> withPch = buildCompileCommand(compiler, source, ptx, pchPath, arch);
            int pchCode = runProcess(withPch, true);
            if (pchCode == 0) {
                logPch("compile key=" + key.value() + " usedPch=true pch=" + pchPath);
                return;
            }
            pchDisabledArchSegments.add(archSegment);
            logPch(
                    "compile key="
                            + key.value()
                            + " usedPch=true exit="
                            + pchCode
                            + " fallingBack=true");
        }
        int code = runProcess(command, true);
        if (code != 0) {
            throw new IllegalStateException("CUDA kernel compilation failed (exit " + code + ")");
        }
    }

    private static List<String> buildCompileCommand(
            String compiler, Path source, Path ptx, Path pchPathOrNull, String arch) {
        List<String> command = new ArrayList<>();
        command.add(compiler);
        command.add("-ptx");
        command.add("-O" + OPT_LEVEL);
        if (arch != null && !arch.isBlank()) {
            command.add("-arch=" + arch);
        }
        command.addAll(EXTRA_FLAGS);
        if (pchPathOrNull != null) {
            command.add("-include-pch");
            command.add(pchPathOrNull.toString());
        }
        command.add(source.toString());
        command.add("-o");
        command.add(ptx.toString());
        return List.copyOf(command);
    }

    private static int runProcess(List<String> command, boolean inheritIo) {
        ProcessBuilder builder = new ProcessBuilder(command);
        if (inheritIo) {
            builder.inheritIO();
        }
        try {
            Process process = builder.start();
            if (!process.waitFor(COMPILE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(1, TimeUnit.SECONDS);
                throw new IllegalStateException(
                        "CUDA kernel compilation timed out after " + COMPILE_TIMEOUT_SECONDS + "s");
            }
            return process.exitValue();
        } catch (IOException e) {
            throw new IllegalStateException("CUDA kernel compilation failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("CUDA kernel compilation interrupted", e);
        }
    }

    private static ProcessResult runProcessCapture(List<String> command) {
        ProcessBuilder builder = new ProcessBuilder(command);
        builder.redirectErrorStream(true);
        try {
            Process process = builder.start();
            if (!process.waitFor(COMPILE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(1, TimeUnit.SECONDS);
                throw new IllegalStateException(
                        "CUDA process timed out after " + COMPILE_TIMEOUT_SECONDS + "s");
            }
            String output =
                    new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            return new ProcessResult(process.exitValue(), output);
        } catch (IOException e) {
            throw new IllegalStateException("CUDA process execution failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("CUDA process execution interrupted", e);
        }
    }

    private Path maybeResolvePch(String compiler, Path source, Path ptx, String arch) {
        String archSegment = cacheArchSegment(arch);
        if (!PCH_ENABLED || pchDisabledArchSegments.contains(archSegment)) {
            return null;
        }
        String pchKey = buildPchKey(compiler, arch);
        Path pchDir = KERNEL_ROOT.resolve("pch").resolve(archSegment).resolve(pchKey);
        ensureDirectory(pchDir);
        Path header = pchDir.resolve("cuda_preamble.hpp");
        Path pch = pchDir.resolve("cuda_preamble_device.pch");
        Object lock = PCH_LOCKS.computeIfAbsent(pchDir.toString(), ignored -> new Object());
        synchronized (lock) {
            writeIfChanged(header, PCH_PREAMBLE);
            if (Files.exists(pch) && !needsCompile(header, pch)) {
                return pch;
            }
            List<String> probeCommand = buildCompileCommand(compiler, source, ptx, null, arch);
            probeCommand = new ArrayList<>(probeCommand);
            probeCommand.add("-###");
            ProcessResult probe = runProcessCapture(probeCommand);
            if (probe.code != 0) {
                logPch("probe failed exit=" + probe.code + " compiler=" + compiler);
                pchDisabledArchSegments.add(archSegment);
                return null;
            }
            List<String> pchCommand = derivePchCommand(probe.output, header, pch);
            if (pchCommand == null) {
                logPch("probe parse failed; no cc1 line found");
                pchDisabledArchSegments.add(archSegment);
                return null;
            }
            int code = runProcess(pchCommand, false);
            if (code != 0 || !Files.exists(pch)) {
                logPch("build failed exit=" + code + " pch=" + pch);
                pchDisabledArchSegments.add(archSegment);
                return null;
            }
            logPch("build success pch=" + pch);
            return pch;
        }
    }

    private static List<String> derivePchCommand(String probeOutput, Path header, Path pch) {
        String cc1Line = null;
        for (String line : probeOutput.split("\\R")) {
            if (line.contains("\"-cc1\"") && line.contains("\"-x\" \"cuda\"")) {
                cc1Line = line.trim();
                break;
            }
        }
        if (cc1Line == null) {
            return null;
        }
        List<String> args = parseQuotedTokens(cc1Line);
        if (args.isEmpty()) {
            return null;
        }
        List<String> transformed = new ArrayList<>();
        for (int i = 0; i < args.size(); i++) {
            String arg = args.get(i);
            if ("-emit-obj".equals(arg)) {
                transformed.add("-emit-pch");
                continue;
            }
            if (("-dumpdir".equals(arg) || "-main-file-name".equals(arg)) && i + 1 < args.size()) {
                i++;
                continue;
            }
            if ("-o".equals(arg) && i + 1 < args.size()) {
                transformed.add("-o");
                transformed.add(pch.toString());
                i++;
                continue;
            }
            transformed.add(arg);
        }
        if (transformed.isEmpty()) {
            return null;
        }
        transformed.set(transformed.size() - 1, header.toString());
        return List.copyOf(transformed);
    }

    private static List<String> parseQuotedTokens(String line) {
        Matcher matcher = QUOTED_TOKEN.matcher(line);
        List<String> result = new ArrayList<>();
        while (matcher.find()) {
            result.add(matcher.group(1));
        }
        return List.copyOf(result);
    }

    private static String buildPchKey(String compiler, String arch) {
        String identity =
                NVCC_ID_CACHE.computeIfAbsent(compiler, CudaKernelBackend::readCompilerIdentity);
        String input =
                "compiler="
                        + compiler
                        + "\nidentity="
                        + identity
                        + "\narch="
                        + cacheArchSegment(arch)
                        + "\nopt="
                        + OPT_LEVEL
                        + "\nflags="
                        + String.join(" ", EXTRA_FLAGS)
                        + "\npreamble="
                        + PCH_PREAMBLE;
        return sha256Hex(input) + "-cuda-pch-v1";
    }

    private static String readCompilerIdentity(String compiler) {
        ProcessResult version = runProcessCapture(List.of(compiler, "--version"));
        if (version.code != 0) {
            return "unknown";
        }
        return version.output.trim();
    }

    private static String sha256Hex(String text) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashed = digest.digest(text.getBytes(StandardCharsets.UTF_8));
            StringBuilder builder = new StringBuilder(hashed.length * 2);
            for (byte value : hashed) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            return builder.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private static String cacheArchSegment(String arch) {
        if (arch == null || arch.isBlank()) {
            return "auto";
        }
        return arch.trim();
    }

    private static void logPch(String message) {
        if (PCH_LOG) {
            System.out.println("[jota-cuda-pch] " + message);
        }
    }

    private record ProcessResult(int code, String output) {}

    private static String resolveArch() {
        String prop = System.getProperty("jota.cuda.arch");
        if (prop != null && !prop.isBlank()) {
            return prop.trim();
        }
        String env = System.getenv("JOTA_CUDA_ARCH");
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        try {
            int deviceIndex = CudaRuntime.currentDevice();
            String deviceArch = CudaRuntime.deviceArchName(deviceIndex);
            if (deviceArch != null && !deviceArch.isBlank()) {
                return deviceArch.trim();
            }
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            // Fall back to nvcc default arch selection.
        }
        return null;
    }

    private static List<String> parseExtraCompileFlags() {
        List<String> result = new ArrayList<>();
        result.add(UNUSED_SCRATCH_WARNING_SUPPRESSION);
        String flags = System.getProperty(EXTRA_FLAGS_PROPERTY);
        if (flags == null || flags.isBlank()) {
            return List.copyOf(result);
        }
        String[] parts = flags.trim().split("\\s+");
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

    static String resolveNvccExecutable() {
        String fromProperty = System.getProperty(NVCC_PROPERTY);
        if (fromProperty != null && !fromProperty.isBlank()) {
            return fromProperty.trim();
        }
        String fromEnv = System.getenv(NVCC_ENV);
        if (fromEnv != null && !fromEnv.isBlank()) {
            return fromEnv.trim();
        }
        return "nvcc";
    }
}
