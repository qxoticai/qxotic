package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRTextRenderer;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelCachePaths;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.LaunchConfig;
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

final class HipKernelBackend implements KernelBackend {

    private static final String ENV_HIPCC = "HIPCC";
    private static final String COMPILER_PROPERTY = "jota.hip.compiler";
    private static final String EXTRA_FLAGS_PROPERTY = "jota.hip.compile.flags";
    private static final String PCH_ENABLED_PROPERTY = "jota.hip.pch.enabled";
    private static final String PCH_LOG_PROPERTY = "jota.hip.pch.log";
    private static final Path KERNEL_ROOT = KernelCachePaths.deviceRoot(DeviceType.HIP);
    private static final boolean KERNEL_LOG = Boolean.getBoolean("jota.kernel.log");
    private static final long COMPILE_TIMEOUT_SECONDS =
            Long.getLong("jota.hip.compile.timeout.seconds", 10L);
    private static final String OPT_LEVEL = System.getProperty("jota.hip.compile.opt", "2").trim();
    private static final boolean TIMING_LOG = Boolean.getBoolean("jota.hip.timing.log");
    private static final boolean PCH_ENABLED =
            Boolean.parseBoolean(System.getProperty(PCH_ENABLED_PROPERTY, "true"));
    private static final boolean PCH_LOG = Boolean.getBoolean(PCH_LOG_PROPERTY);
    private static final String PCH_PREAMBLE =
            "#include <hip/hip_runtime.h>\n"
                    + "#include <hip/hip_fp16.h>\n"
                    + "#include <hip/hip_bfloat16.h>\n"
                    + "#include <stdint.h>\n"
                    + "#include <math.h>\n";
    private static final Pattern QUOTED_TOKEN = Pattern.compile("\\\"([^\\\"]*)\\\"");
    private static final List<String> EXTRA_FLAGS = parseExtraCompileFlags();
    private static final Map<String, String> HIPCC_ID_CACHE = new ConcurrentHashMap<>();
    private static final Map<String, Object> PCH_LOCKS = new ConcurrentHashMap<>();

    private final KernelExecutableCache cache = new InMemoryKernelCache();
    private final Set<String> pchDisabledArchSegments = ConcurrentHashMap.newKeySet();

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        long t0 = System.nanoTime();
        if (!"hip".equals(program.language())) {
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
        String arch = resolveArch();
        Path sourceDir = KERNEL_ROOT.resolve("source").resolve(key.value());
        Path binaryDir =
                KERNEL_ROOT.resolve("binary").resolve(cacheArchSegment(arch)).resolve(key.value());
        Path sourcePath = sourceDir.resolve(kernelName + ".hip");
        Path hsacoPath = binaryDir.resolve(kernelName + ".hsaco");
        ensureDirectory(sourceDir);
        ensureDirectory(binaryDir);
        String source = requireSource(program.payload());
        long t0 = System.nanoTime();
        writeIfChanged(sourcePath, source);
        long tWrite = System.nanoTime();
        if (needsCompile(sourcePath, hsacoPath)) {
            log("HIP kernel compile key=" + key.value() + " entry=" + kernelName);
            compileSource(sourcePath, hsacoPath, key, arch);
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

    private void compileSource(Path source, Path hsaco, KernelCacheKey key, String arch) {
        String hipcc = System.getenv(ENV_HIPCC);
        if (hipcc == null || hipcc.isBlank()) {
            hipcc = "hipcc";
        }
        String compiler = resolveHipcc(hipcc);
        List<String> command = buildCompileCommand(compiler, source, hsaco, null, arch);
        String archSegment = cacheArchSegment(arch);
        Path pchPath = maybeResolvePch(compiler, source, hsaco, arch);
        if (pchPath != null) {
            List<String> withPch = buildCompileCommand(compiler, source, hsaco, pchPath, arch);
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
            throw new IllegalStateException("HIP kernel compilation failed (exit " + code + ")");
        }
    }

    private static List<String> buildCompileCommand(
            String compiler, Path source, Path hsaco, Path pchPathOrNull, String arch) {
        List<String> command = new ArrayList<>();
        command.add(compiler);
        command.add("--genco");
        command.add("-O" + OPT_LEVEL);
        if (arch != null && !arch.isBlank()) {
            command.add("--offload-arch=" + arch);
        }
        command.addAll(EXTRA_FLAGS);
        if (pchPathOrNull != null) {
            command.add("-include-pch");
            command.add(pchPathOrNull.toString());
        }
        command.add(source.toString());
        command.add("-o");
        command.add(hsaco.toString());
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
                        "HIP kernel compilation timed out after " + COMPILE_TIMEOUT_SECONDS + "s");
            }
            return process.exitValue();
        } catch (IOException e) {
            throw new IllegalStateException("HIP kernel compilation failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("HIP kernel compilation interrupted", e);
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
                        "HIP process timed out after " + COMPILE_TIMEOUT_SECONDS + "s");
            }
            String output =
                    new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            return new ProcessResult(process.exitValue(), output);
        } catch (IOException e) {
            throw new IllegalStateException("HIP process execution failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("HIP process execution interrupted", e);
        }
    }

    private Path maybeResolvePch(String compiler, Path source, Path hsaco, String arch) {
        String archSegment = cacheArchSegment(arch);
        if (!PCH_ENABLED || pchDisabledArchSegments.contains(archSegment)) {
            return null;
        }
        String pchKey = buildPchKey(compiler, arch);
        Path pchDir = KERNEL_ROOT.resolve("pch").resolve(archSegment).resolve(pchKey);
        ensureDirectory(pchDir);
        Path header = pchDir.resolve("hip_preamble.hpp");
        Path pch = pchDir.resolve("hip_preamble_device.pch");
        Object lock = PCH_LOCKS.computeIfAbsent(pchDir.toString(), ignored -> new Object());
        synchronized (lock) {
            writeIfChanged(header, PCH_PREAMBLE);
            if (Files.exists(pch) && !needsCompile(header, pch)) {
                return pch;
            }
            List<String> probeCommand = buildCompileCommand(compiler, source, hsaco, null, arch);
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
            if (line.contains("\"-cc1\"") && line.contains("\"-x\" \"hip\"")) {
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
                HIPCC_ID_CACHE.computeIfAbsent(compiler, HipKernelBackend::readCompilerIdentity);
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
        return sha256Hex(input) + "-hip-pch-v1";
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
            System.out.println("[jota-hip-pch] " + message);
        }
    }

    private record ProcessResult(int code, String output) {}

    private static String resolveArch() {
        String prop = System.getProperty("jota.hip.arch");
        if (prop != null && !prop.isBlank()) {
            return prop.trim();
        }
        String env = System.getenv("JOTA_HIP_ARCH");
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        try {
            int deviceIndex = HipRuntime.currentDevice();
            String deviceArch = HipRuntime.deviceArchName(deviceIndex);
            if (deviceArch != null && !deviceArch.isBlank()) {
                return deviceArch.trim();
            }
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            // Fall back to hipcc default arch selection.
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

    private static List<String> parseExtraCompileFlags() {
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
