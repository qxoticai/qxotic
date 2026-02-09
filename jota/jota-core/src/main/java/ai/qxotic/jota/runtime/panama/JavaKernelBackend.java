package ai.qxotic.jota.runtime.panama;

import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.tensor.DiskKernelCache;
import ai.qxotic.jota.tensor.ExecutionStream;
import ai.qxotic.jota.tensor.JavaKernel;
import ai.qxotic.jota.tensor.KernelArgs;
import ai.qxotic.jota.tensor.KernelBackend;
import ai.qxotic.jota.tensor.KernelCacheEntry;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelExecutable;
import ai.qxotic.jota.tensor.KernelProgram;
import ai.qxotic.jota.tensor.LaunchConfig;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

public final class JavaKernelBackend implements KernelBackend {

    private final MemoryDomain<MemorySegment> memoryDomain;
    private final DiskKernelCache cache;
    private final KernelExecutableCache executableCache = new InMemoryKernelCache();

    public JavaKernelBackend(MemoryDomain<MemorySegment> memoryDomain, DiskKernelCache cache) {
        this.memoryDomain = Objects.requireNonNull(memoryDomain, "memoryDomain");
        this.cache = Objects.requireNonNull(cache, "cache");
    }

    @Override
    public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
        ensureJavaProgram(program);
        KernelCacheEntry entry = cache.entryFor(cacheKey);
        String className = program.entryPoint();
        try {
            Files.createDirectories(entry.classOutputDir());
            Files.createDirectories(entry.directory());
            if (program.kind() == KernelProgram.Kind.SOURCE) {
                String source = requireSource(program.payload());
                Path sourceFile = entry.directory().resolve(className + ".java");
                Files.writeString(sourceFile, source);
                compileSource(sourceFile, entry.classOutputDir());
            } else {
                byte[] classBytes = requireBinary(program.payload());
                Path classFile =
                        classFileFor(entry.classOutputDir(), entry.packageName(), className);
                Files.createDirectories(classFile.getParent());
                Files.write(classFile, classBytes);
            }
            JavaKernel kernel = loadKernel(entry.classOutputDir(), entry.packageName(), className);
            return buildExecutable(kernel);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to compile Java kernel", e);
        }
    }

    @Override
    public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
        ensureJavaProgram(program);
        KernelCacheEntry entry = cache.entryFor(cacheKey);
        String className = program.entryPoint();
        Path classFile = classFileFor(entry.classOutputDir(), entry.packageName(), className);
        try {
            Files.createDirectories(entry.classOutputDir());
            Files.createDirectories(entry.directory());
            if (program.kind() == KernelProgram.Kind.BINARY) {
                byte[] classBytes = requireBinary(program.payload());
                Files.createDirectories(classFile.getParent());
                Files.write(classFile, classBytes);
            } else if (!Files.exists(classFile)) {
                String source = requireSource(program.payload());
                Path sourceFile = entry.directory().resolve(className + ".java");
                Files.writeString(sourceFile, source);
                compileSource(sourceFile, entry.classOutputDir());
            }
            JavaKernel kernel = loadKernel(entry.classOutputDir(), entry.packageName(), className);
            return buildExecutable(kernel);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load Java kernel", e);
        }
    }

    @Override
    public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey) {
        KernelExecutable existing = executableCache.get(cacheKey);
        if (existing != null) {
            return existing;
        }
        KernelExecutable created =
                program.kind() == KernelProgram.Kind.BINARY
                        ? load(program, cacheKey)
                        : compile(program, cacheKey);
        executableCache.put(cacheKey, created);
        return created;
    }

    @Override
    public KernelExecutableCache cache() {
        return executableCache;
    }

    private KernelExecutable buildExecutable(JavaKernel kernel) {
        return new KernelExecutable() {
            @Override
            public void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream) {
                kernel.execute(memoryDomain, args);
            }

            @Override
            public void close() {
                // no-op
            }
        };
    }

    private void compileSource(Path sourceFile, Path classOutputDir) throws IOException {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            throw new IllegalStateException("JavaCompiler is not available");
        }
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(diagnostics, Locale.ROOT, null)) {
            Iterable<? extends JavaFileObject> units =
                    fileManager.getJavaFileObjects(sourceFile.toFile());
            List<String> options = new ArrayList<>();
            options.add("-classpath");
            options.add(System.getProperty("java.class.path"));
            options.add("-d");
            options.add(classOutputDir.toString());
            if (ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent()) {
                options.add("--add-modules");
                options.add("jdk.incubator.vector");
            }
            JavaCompiler.CompilationTask task =
                    compiler.getTask(null, fileManager, diagnostics, options, null, units);
            Boolean success = task.call();
            if (success == null || !success) {
                throw new IllegalStateException(formatDiagnostics(diagnostics.getDiagnostics()));
            }
        }
    }

    private static Path classFileFor(Path classOutputDir, String packageName, String className) {
        return classOutputDir.resolve(packageName.replace('.', '/')).resolve(className + ".class");
    }

    private JavaKernel loadKernel(Path classOutputDir, String packageName, String className) {
        try {
            URLClassLoader loader = new URLClassLoader(new URL[] {classOutputDir.toUri().toURL()});
            String fqcn = packageName + "." + className;
            Class<?> clazz = Class.forName(fqcn, true, loader);
            Object instance = clazz.getDeclaredConstructor().newInstance();
            return (JavaKernel) instance;
        } catch (ReflectiveOperationException | IOException e) {
            throw new IllegalStateException("Failed to load Java kernel", e);
        }
    }

    private static void ensureJavaProgram(KernelProgram program) {
        if (!KernelProgram.JAVA.equals(program.language())) {
            throw new UnsupportedOperationException("Java backend expects JAVA programs");
        }
    }

    private static String requireSource(Object payload) {
        if (payload instanceof String source) {
            return source;
        }
        throw new IllegalArgumentException("Expected Java source payload as String");
    }

    private static byte[] requireBinary(Object payload) {
        if (payload instanceof byte[] bytes) {
            return bytes;
        }
        throw new IllegalArgumentException("Expected Java class payload as byte[]");
    }

    private static String formatDiagnostics(
            List<Diagnostic<? extends JavaFileObject>> diagnostics) {
        StringBuilder sb = new StringBuilder("Java kernel compilation failed:\n");
        for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics) {
            sb.append("  ")
                    .append(diagnostic.getKind())
                    .append(" at line ")
                    .append(diagnostic.getLineNumber())
                    .append(": ")
                    .append(diagnostic.getMessage(Locale.ROOT))
                    .append('\n');
        }
        return sb.toString();
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
}
