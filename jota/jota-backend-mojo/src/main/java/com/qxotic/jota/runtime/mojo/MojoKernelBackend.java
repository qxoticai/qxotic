package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.runtime.KernelBackend;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/** Mojo kernel backend - compiles .mojo source to GPU binary via Mojo toolchain. */
final class MojoKernelBackend implements KernelBackend {

    private static final String TARGET_PROPERTY = "jota.mojo.target";
    static final String COMPILER_PROPERTY = "jota.mojo.compiler";
    static final String COMPILER_ENV = "JOTA_MOJO_COMPILER";
    private static final String HIP_ARCH_PROPERTY = "jota.hip.arch";
    private static final String COMPILE_FLAGS_PROPERTY = "jota.mojo.compile.flags";
    private static final String SUMMARY_PROPERTY = "jota.mojo.kernel.summary";
    private static final String DEFAULT_TARGET = "gfx1103";
    private static final long COMPILE_TIMEOUT_SECONDS =
            Long.getLong("jota.mojo.compile.timeout.seconds", 30L);
    private static final AtomicLong CACHE_HITS = new AtomicLong();
    private static final AtomicLong CACHE_MISSES = new AtomicLong();
    private static final AtomicLong NATIVE_COMPILES = new AtomicLong();
    private static final AtomicLong BINARY_LOADS = new AtomicLong();
    private static final AtomicLong COMPILE_NANOS_TOTAL = new AtomicLong();
    private static final AtomicLong COMPILE_NANOS_MAX = new AtomicLong();
    private static final AtomicLong NEW_KEYS_THIS_RUN = new AtomicLong();
    private static final Set<String> SEEN_KEYS = ConcurrentHashMap.newKeySet();

    static {
        if (Boolean.getBoolean(SUMMARY_PROPERTY)) {
            Runtime.getRuntime()
                    .addShutdownHook(
                            new Thread(
                                    () ->
                                            System.out.println(
                                                    "[jota-mojo-summary] uniqueKeys="
                                                            + SEEN_KEYS.size()
                                                            + " newKeys="
                                                            + NEW_KEYS_THIS_RUN.get()
                                                            + " nativeCompiles="
                                                            + NATIVE_COMPILES.get()
                                                            + " compileMsTotal="
                                                            + (COMPILE_NANOS_TOTAL.get()
                                                                    / 1_000_000)
                                                            + " compileMsMax="
                                                            + (COMPILE_NANOS_MAX.get() / 1_000_000)
                                                            + " binaryLoads="
                                                            + BINARY_LOADS.get()
                                                            + " cacheHits="
                                                            + CACHE_HITS.get()
                                                            + " cacheMisses="
                                                            + CACHE_MISSES.get()),
                                    "jota-mojo-summary"));
        }
    }

    private final KernelBackend delegate;

    MojoKernelBackend(KernelBackend delegate) {
        if (delegate instanceof MojoKernelBackend) {
            throw new IllegalArgumentException(
                    "MojoKernelBackend delegate must be a non-Mojo backend");
        }
        this.delegate = Objects.requireNonNull(delegate, "delegate");
    }

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        return delegate.compile(adaptProgram(program, cacheKey), cacheKey);
    }

    @Override
    public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
        return delegate.load(adaptProgram(program, cacheKey), cacheKey);
    }

    @Override
    public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey) {
        if (SEEN_KEYS.add(cacheKey.value())) {
            NEW_KEYS_THIS_RUN.incrementAndGet();
        }
        KernelExecutable cached = delegate.cache().get(cacheKey);
        if (cached != null) {
            CACHE_HITS.incrementAndGet();
            return cached;
        }
        CACHE_MISSES.incrementAndGet();
        KernelProgram adapted = adaptProgram(program, cacheKey);
        try {
            return delegate.getOrCompile(adapted, cacheKey);
        } catch (RuntimeException ex) {
            if (isMojoSource(program) && adapted.kind() == KernelProgram.Kind.BINARY) {
                clearPersistedBinary(cacheKey);
                KernelProgram rebuilt = compileMojoToGpuBinary(program, cacheKey);
                return delegate.getOrCompile(rebuilt, cacheKey);
            }
            throw ex;
        }
    }

    @Override
    public KernelExecutableCache cache() {
        return delegate.cache();
    }

    private static KernelProgram adaptProgram(KernelProgram program, KernelCacheKey cacheKey) {
        if (!isMojoSource(program)) {
            return program;
        }
        KernelProgram cachedBinary = loadCachedBinaryProgram(program, cacheKey);
        if (cachedBinary != null) {
            BINARY_LOADS.incrementAndGet();
            return cachedBinary;
        }
        return compileMojoToGpuBinary(program, cacheKey);
    }

    private static boolean isMojoSource(KernelProgram program) {
        return program.kind() == KernelProgram.Kind.SOURCE
                && "mojo".equalsIgnoreCase(program.language());
    }

    /** Compiles Mojo source to GPU binary. */
    private static KernelProgram compileMojoToGpuBinary(
            KernelProgram program, KernelCacheKey cacheKey) {
        long compileStart = System.nanoTime();
        String mojoSource = requireSource(program);
        // Use the entryPoint from the program (already sanitized to be a valid Mojo identifier)
        String fnName = program.entryPoint();
        String target = resolveTarget();

        Path sourcePath = MojoCachePaths.lirSourcePath(cacheKey.value());
        Path sourceDir = MojoCachePaths.lirKernelDir(cacheKey.value());
        Path wrapperPath = MojoCachePaths.lirWrapperPath(cacheKey.value());
        Path asmPath = MojoCachePaths.lirAsmPath(cacheKey.value());

        try {
            Files.createDirectories(sourceDir);
            Files.writeString(sourcePath, mojoSource);
            Files.writeString(wrapperPath, wrapper(mojoSource, fnName));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to prepare Mojo source for " + cacheKey, e);
        }

        List<String> command =
                new ArrayList<>(
                        List.of(
                                resolveCompilerExecutable(),
                                "build",
                                wrapperPath.toString(),
                                "--emit",
                                "asm",
                                "--target-accelerator",
                                target,
                                "-o",
                                asmPath.toString()));
        command.addAll(tokenizeCompileFlags(System.getProperty(COMPILE_FLAGS_PROPERTY, "")));
        exec(command, COMPILE_TIMEOUT_SECONDS);

        byte[] elf = extractElfFromAsm(asmPath);
        String entry = readElfSymbol(elf, fnName);
        persistBinary(cacheKey, elf, entry);
        long elapsed = System.nanoTime() - compileStart;
        COMPILE_NANOS_TOTAL.addAndGet(elapsed);
        COMPILE_NANOS_MAX.accumulateAndGet(elapsed, Math::max);
        NATIVE_COMPILES.incrementAndGet();

        return KernelProgram.binary("hip", elf, entry, program.options());
    }

    private static KernelProgram loadCachedBinaryProgram(
            KernelProgram source, KernelCacheKey cacheKey) {
        Path binaryPath = MojoCachePaths.lirBinaryPath(cacheKey.value());
        Path entryPath = MojoCachePaths.lirEntryPath(cacheKey.value());
        if (!Files.isRegularFile(binaryPath) || !Files.isRegularFile(entryPath)) {
            return null;
        }
        try {
            byte[] elf = Files.readAllBytes(binaryPath);
            String entry = Files.readString(entryPath).trim();
            if (elf.length == 0 || entry.isEmpty()) {
                return null;
            }
            return KernelProgram.binary("hip", elf, entry, source.options());
        } catch (IOException e) {
            return null;
        }
    }

    private static void persistBinary(KernelCacheKey cacheKey, byte[] elf, String entry) {
        Path binaryPath = MojoCachePaths.lirBinaryPath(cacheKey.value());
        Path entryPath = MojoCachePaths.lirEntryPath(cacheKey.value());
        try {
            Files.createDirectories(binaryPath.getParent());
            Files.write(binaryPath, elf);
            Files.writeString(entryPath, entry);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to persist Mojo binary for " + cacheKey.value(), e);
        }
    }

    private static void clearPersistedBinary(KernelCacheKey cacheKey) {
        try {
            Files.deleteIfExists(MojoCachePaths.lirBinaryPath(cacheKey.value()));
            Files.deleteIfExists(MojoCachePaths.lirEntryPath(cacheKey.value()));
        } catch (IOException ignored) {
            // Ignore stale-cache cleanup failures; next compile attempt will surface real issues.
        }
    }

    private static String requireSource(KernelProgram program) {
        if (program.payload() instanceof String s) {
            return s;
        }
        throw new IllegalArgumentException("Expected Mojo source payload as String");
    }

    private static String wrapper(String source, String fnName) {
        return source
                + "\n\n"
                + "from std.sys import has_accelerator\n"
                + "from std.gpu.host import DeviceContext\n\n"
                + "def main():\n"
                + "    if not has_accelerator():\n"
                + "        return\n"
                + "    ctx = DeviceContext(api=\"hip\")\n"
                + "    _ = ctx.compile_function["
                + fnName
                + ", "
                + fnName
                + "]()\n";
    }

    private static String resolveTarget() {
        String t = System.getProperty(TARGET_PROPERTY);
        if (t != null && !t.isBlank()) {
            return t.trim();
        }
        t = System.getProperty(HIP_ARCH_PROPERTY);
        if (t != null && !t.isBlank()) {
            return t.trim();
        }
        return DEFAULT_TARGET;
    }

    static String resolveCompilerExecutable() {
        String fromProperty = System.getProperty(COMPILER_PROPERTY);
        if (fromProperty != null && !fromProperty.isBlank()) {
            return fromProperty.trim();
        }
        String fromEnv = System.getenv(COMPILER_ENV);
        if (fromEnv != null && !fromEnv.isBlank()) {
            return fromEnv.trim();
        }
        return "mojo";
    }

    private static List<String> tokenizeCompileFlags(String options) {
        String trimmed = options == null ? "" : options.trim();
        if (trimmed.isEmpty()) {
            return List.of();
        }
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        char quote = 0;
        for (int i = 0; i < trimmed.length(); i++) {
            char c = trimmed.charAt(i);
            if (quote != 0) {
                if (c == quote) {
                    quote = 0;
                } else {
                    current.append(c);
                }
                continue;
            }
            if (c == '\'' || c == '"') {
                quote = c;
                continue;
            }
            if (Character.isWhitespace(c)) {
                if (!current.isEmpty()) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
                continue;
            }
            current.append(c);
        }
        if (!current.isEmpty()) {
            tokens.add(current.toString());
        }
        return tokens;
    }

    private static byte[] extractElfFromAsm(Path asmPath) {
        String text;
        try {
            text = Files.readString(asmPath);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read Mojo asm: " + asmPath, e);
        }

        int marker = text.indexOf("\\177ELF");
        if (marker < 0) {
            throw new IllegalStateException("No embedded ELF found in: " + asmPath);
        }

        int quote = text.lastIndexOf('"', marker);
        if (quote < 0) {
            throw new IllegalStateException("Malformed asm (missing quote) in: " + asmPath);
        }

        return decodeCString(text, quote + 1);
    }

    private static byte[] decodeCString(String s, int start) {
        byte[] buf = new byte[s.length() - start];
        int len = 0;

        for (int i = start; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"') {
                if (len < 4 || buf[0] != 0x7f || buf[1] != 'E' || buf[2] != 'L' || buf[3] != 'F') {
                    throw new IllegalStateException("Decoded payload is not a valid ELF");
                }
                byte[] result = new byte[len];
                System.arraycopy(buf, 0, result, 0, len);
                return result;
            }

            if (c != '\\') {
                buf[len++] = (byte) c;
                continue;
            }

            if (i + 1 >= s.length()) {
                throw new IllegalStateException("Trailing backslash in C-string");
            }

            char esc = s.charAt(++i);
            if (esc >= '0' && esc <= '7') {
                int v = esc - '0';
                int digits = 1;
                while (digits < 3 && i + 1 < s.length()) {
                    char n = s.charAt(i + 1);
                    if (n < '0' || n > '7') break;
                    v = v * 8 + (n - '0');
                    i++;
                    digits++;
                }
                buf[len++] = (byte) v;
            } else {
                buf[len++] =
                        switch (esc) {
                            case 'a' -> 0x07;
                            case 'b' -> '\b';
                            case 'f' -> '\f';
                            case 'n' -> '\n';
                            case 'r' -> '\r';
                            case 't' -> '\t';
                            case 'v' -> 0x0b;
                            case '0' -> 0;
                            default -> (byte) esc;
                        };
            }
        }

        throw new IllegalStateException("Unterminated C-string in asm");
    }

    private static void exec(List<String> cmd, long timeoutSeconds) {
        execCapture(cmd, timeoutSeconds);
    }

    private static String readElfSymbol(byte[] elf, String fallbackFnName) {
        // Mojo docs expose @export for external visibility, but DeviceContext.compile_function
        // does not document a guaranteed symbol-name preservation contract for GPU output.
        // Keep resolving the emitted symbol for reliable hipModuleGetFunction lookup.
        // See: https://docs.modular.com/mojo/manual/decorators/export
        // and: https://docs.modular.com/mojo/std/gpu/host/device_context/DeviceContext
        Path tmp;
        try {
            tmp = Files.createTempFile("mojo-kernel-", ".elf");
            Files.write(tmp, elf);
            tmp.toFile().deleteOnExit();
        } catch (IOException e) {
            return fallbackFnName;
        }

        for (String nm : List.of("llvm-nm", "/opt/rocm/llvm/bin/llvm-nm")) {
            try {
                String out = execCapture(List.of(nm, tmp.toString()), 5);
                for (String line : out.split("\n")) {
                    String[] parts = line.trim().split("\\s+");
                    if (parts.length >= 3 && (parts[1].equals("T") || parts[1].equals("t"))) {
                        String sym = parts[2];
                        if (!sym.isEmpty() && !sym.startsWith(".")) {
                            return sym;
                        }
                    }
                }
            } catch (RuntimeException ignored) {
                // try next llvm-nm candidate
            }
        }
        return fallbackFnName;
    }

    private static String execCapture(List<String> cmd, long timeoutSeconds) {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        try {
            Process p = pb.start();
            boolean ok = p.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            String out = new String(p.getInputStream().readAllBytes());
            if (!ok) {
                p.destroyForcibly();
                throw new IllegalStateException("Command timed out: " + String.join(" ", cmd));
            }
            if (p.exitValue() != 0) {
                throw new IllegalStateException(
                        "Command failed (exit "
                                + p.exitValue()
                                + "): "
                                + String.join(" ", cmd)
                                + "\n"
                                + out);
            }
            return out;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to execute: " + String.join(" ", cmd), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted: " + String.join(" ", cmd), e);
        }
    }
}
