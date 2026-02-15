package ai.qxotic.jota.runtime.panama;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.*;
import ai.qxotic.jota.ir.lir.scratch.ScratchLayout;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.tensor.JavaKernel;
import ai.qxotic.jota.tensor.KernelCache;
import ai.qxotic.jota.tensor.KernelCacheEntry;
import ai.qxotic.jota.tensor.KernelCacheKey;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

final class LIRKernelCompiler {

    private final KernelCache cache;

    LIRKernelCompiler(KernelCache cache) {
        this.cache = Objects.requireNonNull(cache, "cache");
    }

    /** Compiles a kernel without scratch buffer support. */
    JavaKernel compile(LIRGraph graph) {
        return compile(graph, ScratchLayout.EMPTY);
    }

    /** Compiles a kernel with scratch buffer support. */
    JavaKernel compile(LIRGraph graph, ScratchLayout scratchLayout) {
        boolean needsStridedKernel = needsStridedKernel(graph);
        KernelCacheKey key = buildCacheKey(graph, scratchLayout, needsStridedKernel);
        KernelCacheEntry entry = cache.entryFor(key);
        try {
            Files.createDirectories(entry.classOutputDir());
            Files.createDirectories(entry.directory());
            String source =
                    needsStridedKernel
                            ? KernelSourceGenerator.generateStrided(entry, graph, scratchLayout)
                            : KernelSourceGenerator.generate(entry, graph, scratchLayout);
            Files.writeString(entry.sourcePath(), source);
            if (Files.exists(entry.classFilePath())) {
                return load(entry);
            }
            compileSource(entry);
            return load(entry);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to compile LIR kernel", e);
        }
    }

    KernelCacheKey cacheKeyFor(LIRGraph graph, ScratchLayout scratchLayout) {
        boolean needsStridedKernel = needsStridedKernel(graph);
        return buildCacheKey(graph, scratchLayout, needsStridedKernel);
    }

    private void compileSource(KernelCacheEntry entry) throws IOException {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            throw new IllegalStateException("JavaCompiler is not available");
        }
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        try (StandardJavaFileManager fileManager =
                compiler.getStandardFileManager(diagnostics, Locale.ROOT, null)) {
            Iterable<? extends JavaFileObject> units =
                    fileManager.getJavaFileObjects(entry.sourcePath().toFile());
            List<String> options =
                    List.of(
                            "-classpath",
                            compilerClassPath(),
                            "-d",
                            entry.classOutputDir().toString());
            JavaCompiler.CompilationTask task =
                    compiler.getTask(null, fileManager, diagnostics, options, null, units);
            Boolean success = task.call();
            if (success == null || !success) {
                throw new IllegalStateException(formatDiagnostics(diagnostics.getDiagnostics()));
            }
        }
    }

    private JavaKernel load(KernelCacheEntry entry) {
        try {
            URLClassLoader loader =
                    new URLClassLoader(
                            new URL[] {entry.classOutputDir().toUri().toURL()},
                            JavaKernel.class.getClassLoader());
            Class<?> clazz =
                    Class.forName(entry.packageName() + "." + entry.className(), true, loader);
            Object instance = clazz.getDeclaredConstructor().newInstance();
            return (JavaKernel) instance;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load kernel " + entry.className(), e);
        }
    }

    private static String compilerClassPath() {
        LinkedHashSet<String> entries = new LinkedHashSet<>();
        addSplitClassPath(entries, System.getProperty("java.class.path"));
        addSplitClassPath(entries, System.getProperty("surefire.test.class.path"));
        collectClassLoaderPaths(Thread.currentThread().getContextClassLoader(), entries);
        collectClassLoaderPaths(LIRKernelCompiler.class.getClassLoader(), entries);
        return String.join(File.pathSeparator, entries);
    }

    private static void addSplitClassPath(LinkedHashSet<String> entries, String classPath) {
        if (classPath == null || classPath.isBlank()) {
            return;
        }
        for (String part : classPath.split(java.util.regex.Pattern.quote(File.pathSeparator))) {
            if (part != null && !part.isBlank()) {
                entries.add(part);
            }
        }
    }

    private static void collectClassLoaderPaths(ClassLoader loader, LinkedHashSet<String> entries) {
        ClassLoader cursor = loader;
        while (cursor != null) {
            if (cursor instanceof URLClassLoader urlClassLoader) {
                for (URL url : urlClassLoader.getURLs()) {
                    if (!"file".equalsIgnoreCase(url.getProtocol())) {
                        continue;
                    }
                    try {
                        Path path = Paths.get(url.toURI());
                        String pathString = path.toString();
                        if (!pathString.isBlank()) {
                            entries.add(pathString);
                        }
                    } catch (Exception ignored) {
                        // best-effort classpath assembly
                    }
                }
            }
            cursor = cursor.getParent();
        }
    }

    private KernelCacheKey buildCacheKey(
            LIRGraph graph, ScratchLayout scratchLayout, boolean needsStridedKernel) {
        String hash = hashLirGraph(graph, scratchLayout, needsStridedKernel);
        return KernelCacheKey.of(hash + "-lir-v7");
    }

    private String hashLirGraph(
            LIRGraph graph, ScratchLayout scratchLayout, boolean needsStridedKernel) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            updateGraphHash(digest, graph, scratchLayout, needsStridedKernel);
            byte[] hashed = digest.digest();
            StringBuilder builder = new StringBuilder(hashed.length * 2);
            for (byte value : hashed) {
                builder.append(String.format("%02x", value));
            }
            return builder.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private void updateGraphHash(
            MessageDigest digest,
            LIRGraph graph,
            ScratchLayout scratchLayout,
            boolean needsStridedKernel) {
        updateInt(digest, graph.inputs().size());
        for (LIRInput input : graph.inputs()) {
            if (input instanceof BufferRef buf) {
                updateInt(digest, 1);
                updateBufferRef(digest, buf);
            } else if (input instanceof ScalarInput scalar) {
                updateInt(digest, 2);
                updateInt(digest, scalar.id());
                updateLong(digest, hashDataType(scalar.dataType()));
            } else {
                updateInt(digest, 0);
            }
        }

        updateInt(digest, graph.outputs().size());
        for (BufferRef output : graph.outputs()) {
            updateBufferRef(digest, output);
        }

        Map<LIRExprNode, Long> memo = new IdentityHashMap<>();
        long bodyHash = hashNode(graph.exprGraph(), graph.body(), memo);
        updateLong(digest, bodyHash);

        updateBoolean(digest, needsStridedKernel);
        if (scratchLayout.requiresScratch()) {
            updateLong(digest, scratchLayout.totalByteSize());
            List<Map.Entry<BufferRef, Long>> offsets =
                    new ArrayList<>(scratchLayout.offsets().entrySet());
            offsets.sort(java.util.Comparator.comparingInt(e -> e.getKey().id()));
            updateInt(digest, offsets.size());
            for (Map.Entry<BufferRef, Long> entry : offsets) {
                updateBufferRef(digest, entry.getKey());
                updateLong(digest, entry.getValue());
            }
        } else {
            updateInt(digest, 0);
        }
    }

    private long hashNode(LIRExprGraph exprGraph, LIRExprNode node, Map<LIRExprNode, Long> memo) {
        LIRExprNode resolved = exprGraph.resolve(node);
        Long cached = memo.get(resolved);
        if (cached != null) {
            return cached;
        }
        long hash = 0x9e3779b97f4a7c15L;
        hash = mix(hash, resolved.kind().ordinal());
        hash = mix(hash, resolved.dataType() == null ? -1 : hashDataType(resolved.dataType()));
        hash = mix(hash, resolved.useCount());
        switch (resolved) {
            case SConst sc -> hash = mix(hash, sc.rawBits());
            case SInput si -> hash = mix(hash, si.inputId());
            case SUnary su -> hash = mix(hash, su.op().ordinal());
            case SBinary sb -> hash = mix(hash, sb.op().ordinal());
            case STernary __ -> hash = mix(hash, 3);
            case SCast sc -> hash = mix(hash, hashDataType(sc.targetType()));
            case SLoad sl -> hash = mix(hash, hashBufferRef(sl.buffer()));
            case SFromIndex __ -> hash = mix(hash, 5);
            case SRef sr -> hash = mix(hash, sr.name().hashCode());
            case IConst ic -> hash = mix(hash, ic.value());
            case IVar iv -> hash = mix(hash, iv.name().hashCode());
            case IBinary ib -> hash = mix(hash, ib.op().ordinal());
            case IFromScalar __ -> hash = mix(hash, 7);
            case Store store -> hash = mix(hash, hashBufferRef(store.buffer()));
            case StructuredFor loop -> {
                hash = mix(hash, loop.indexName().hashCode());
                hash = mix(hash, loop.iterArgs().size());
                for (LoopIterArg arg : loop.iterArgs()) {
                    hash = mix(hash, arg.name().hashCode());
                    hash = mix(hash, hashDataType(arg.dataType()));
                    hash = mix(hash, hashNode(exprGraph, arg.init(), memo));
                }
            }
            case Block block -> hash = mix(hash, block.statements().size());
            case Yield yield -> hash = mix(hash, yield.values().size());
            default -> {}
        }
        for (LIRExprNode input : resolved.inputs()) {
            hash = mix(hash, hashNode(exprGraph, input, memo));
        }
        memo.put(resolved, hash);
        return hash;
    }

    private long hashBufferRef(BufferRef buffer) {
        long hash = 0x9e3779b97f4a7c15L;
        hash = mix(hash, buffer.id());
        hash = mix(hash, hashDataType(buffer.dataType()));
        int rank = (int) buffer.layout().shape().flatRank();
        hash = mix(hash, rank);
        for (int i = 0; i < rank; i++) {
            hash = mix(hash, buffer.layout().shape().flatAt(i));
            hash = mix(hash, buffer.layout().stride().flatAt(i));
        }
        return hash;
    }

    private void updateBufferRef(MessageDigest digest, BufferRef buffer) {
        updateLong(digest, hashBufferRef(buffer));
    }

    private void updateBoolean(MessageDigest digest, boolean value) {
        digest.update((byte) (value ? 1 : 0));
    }

    private void updateInt(MessageDigest digest, int value) {
        updateLong(digest, value);
    }

    private void updateLong(MessageDigest digest, long value) {
        digest.update((byte) (value));
        digest.update((byte) (value >>> 8));
        digest.update((byte) (value >>> 16));
        digest.update((byte) (value >>> 24));
        digest.update((byte) (value >>> 32));
        digest.update((byte) (value >>> 40));
        digest.update((byte) (value >>> 48));
        digest.update((byte) (value >>> 56));
    }

    private long hashDataType(DataType dataType) {
        return dataType.name().hashCode();
    }

    private long mix(long hash, long value) {
        long h = hash;
        h ^= value + 0x9e3779b97f4a7c15L + (h << 6) + (h >>> 2);
        return h;
    }

    private boolean needsStridedKernel(LIRGraph graph) {
        // Check if any input or output buffer has non-unit strides
        for (LIRInput input : graph.inputs()) {
            if (input instanceof BufferRef buffer) {
                if (!isContiguousStride(buffer.byteStrides(), buffer.dataType())) {
                    return true;
                }
            }
        }
        for (BufferRef buffer : graph.outputs()) {
            if (!isContiguousStride(buffer.byteStrides(), buffer.dataType())) {
                return true;
            }
        }
        return false;
    }

    private boolean isContiguousStride(long[] strides, DataType dataType) {
        long elementSize = dataType.byteSize();
        long expectedStride = 1;
        // Check if strides match contiguous layout (row-major with unit element strides)
        for (int i = strides.length - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= dataType.byteSize(); // This is wrong for multi-dim
        }
        // For simplicity, check if all strides are non-zero and not broadcasted
        // A proper implementation would check if strides match contiguous layout
        for (long stride : strides) {
            if (stride == 0) { // broadcasted
                return false;
            }
        }
        // Check if the last stride equals element size (contiguous along last dimension)
        if (strides.length > 0 && strides[strides.length - 1] != elementSize) {
            return false;
        }
        return true;
    }

    private String hashLirGraph(LIRGraph graph, ScratchLayout scratchLayout) {
        LIRTextRenderer renderer = new LIRTextRenderer();
        StringBuilder text = new StringBuilder(renderer.render(graph));
        // Include scratch layout in hash to ensure cache invalidation
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
            byte[] bytes = text.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
            byte[] hashed = digest.digest(bytes);
            StringBuilder builder = new StringBuilder(hashed.length * 2);
            for (byte value : hashed) {
                builder.append(String.format("%02x", value));
            }
            return builder.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private String formatDiagnostics(List<Diagnostic<? extends JavaFileObject>> diagnostics) {
        StringBuilder builder = new StringBuilder("Kernel compilation failed:\n");
        for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics) {
            builder.append(diagnostic.getKind())
                    .append(": ")
                    .append(diagnostic.getMessage(Locale.ROOT))
                    .append(" at ")
                    .append(diagnostic.getSource() == null ? "<unknown>" : diagnostic.getSource())
                    .append(":")
                    .append(diagnostic.getLineNumber())
                    .append("\n");
        }
        return builder.toString();
    }

    private static final class KernelSourceGenerator {

        private final KernelCacheEntry entry;
        private final LIRGraph graph;
        private final ScratchLayout scratchLayout;
        private final Map<ScalarInput, String> scalarInputs = new IdentityHashMap<>();
        private final Map<Integer, String> scalarInputNames = new HashMap<>();
        private final Map<LIRExprNode, String> tempNames = new IdentityHashMap<>();
        private final Map<BufferRef, BufferVar> buffers = new IdentityHashMap<>();
        private final Map<BufferRef, ScratchBufferVar> scratchBuffers = new IdentityHashMap<>();
        private final Map<String, String> scalarNames = new IdentityHashMap<>();
        private final List<String> lines = new ArrayList<>();
        private final Set<DataType> usedTypes = new HashSet<>();
        private final Map<BufferRef, String[]> bufferStrideVars = new IdentityHashMap<>();
        private final boolean generateStridedKernel;
        private int tempId;
        private int indentLevel;
        private final LIRExprGraph exprGraph;

        private KernelSourceGenerator(
                KernelCacheEntry entry,
                LIRGraph graph,
                ScratchLayout scratchLayout,
                boolean generateStridedKernel) {
            this.entry = entry;
            this.graph = graph;
            this.scratchLayout = scratchLayout;
            this.generateStridedKernel = generateStridedKernel;
            this.exprGraph = graph.exprGraph();
        }

        static String generate(
                KernelCacheEntry entry, LIRGraph graph, ScratchLayout scratchLayout) {
            KernelSourceGenerator generator =
                    new KernelSourceGenerator(entry, graph, scratchLayout, false);
            return generator.generate();
        }

        static String generateStrided(
                KernelCacheEntry entry, LIRGraph graph, ScratchLayout scratchLayout) {
            KernelSourceGenerator generator =
                    new KernelSourceGenerator(entry, graph, scratchLayout, true);
            return generator.generate();
        }

        private String generate() {
            StringBuilder source = new StringBuilder();
            source.append("package ").append(entry.packageName()).append(";\n\n");
            source.append("import ai.qxotic.jota.BFloat16;\n");
            source.append("import ai.qxotic.jota.memory.Memory;\n");
            source.append("import ai.qxotic.jota.memory.MemoryAccess;\n");
            source.append("import ai.qxotic.jota.memory.MemoryDomain;\n");
            source.append("import ai.qxotic.jota.memory.MemoryView;\n");
            source.append("import ai.qxotic.jota.tensor.KernelArgs;\n");
            source.append("import ai.qxotic.jota.tensor.JavaKernel;\n");
            source.append("import java.lang.foreign.MemorySegment;\n");
            source.append("import java.lang.foreign.ValueLayout;\n");
            source.append("\n");
            source.append("public final class ")
                    .append(entry.className())
                    .append(" implements JavaKernel {\n");

            if (scratchLayout.requiresScratch()) {
                // Kernel that needs scratch - implement execute with scratch
                emitExecuteWithScratch(source);
                emitExecuteWithoutScratchStub(source);
                emitScratchByteSize(source);
            } else {
                // Kernel without scratch - implement execute without scratch
                emitExecuteWithoutScratch(source);
            }

            emitHelpers(source);
            source.append("}\n");
            return source.toString();
        }

        private void emitExecuteWithoutScratch(StringBuilder source) {
            source.append("  @Override\n");
            source.append("  @SuppressWarnings(\"unchecked\")\n");
            source.append(
                    "  public void execute(MemoryDomain<MemorySegment> memoryDomain, KernelArgs args) {\n");
            source.append(
                    "    MemoryAccess<MemorySegment> access = (MemoryAccess<MemorySegment>) memoryDomain.directAccess();\n");
            emitInputs(source);
            emitOutputs(source);

            lines.clear();
            emitNode(graph.body());
            for (String line : lines) {
                source.append("    ").append(line).append("\n");
            }
            source.append("  }\n");
        }

        private void emitExecuteWithScratch(StringBuilder source) {
            source.append("  @Override\n");
            source.append("  @SuppressWarnings(\"unchecked\")\n");
            source.append(
                    "  public void execute(MemoryDomain<MemorySegment> memoryDomain, KernelArgs args, "
                            + "Memory<MemorySegment> scratch) {\n");
            source.append(
                    "    MemoryAccess<MemorySegment> access = (MemoryAccess<MemorySegment>) memoryDomain.directAccess();\n");
            source.append("    MemorySegment scratchBase = scratch.base();\n");
            emitInputs(source);
            emitOutputs(source);
            emitScratchSlices(source);

            lines.clear();
            emitNode(graph.body());
            for (String line : lines) {
                source.append("    ").append(line).append("\n");
            }
            source.append("  }\n");
        }

        private void emitExecuteWithoutScratchStub(StringBuilder source) {
            source.append("  @Override\n");
            source.append(
                    "  public void execute(MemoryDomain<MemorySegment> memoryDomain, KernelArgs args) {\n");
            source.append(
                    "    throw new UnsupportedOperationException(\"This kernel requires scratch memory\");\n");
            source.append("  }\n");
        }

        private void emitScratchByteSize(StringBuilder source) {
            source.append("  @Override\n");
            source.append("  public long scratchByteSize() {\n");
            source.append("    return ")
                    .append(scratchLayout.alignedTotalByteSize())
                    .append("L;\n");
            source.append("  }\n");
        }

        private void emitScratchSlices(StringBuilder source) {
            int slotId = 0;
            for (var entry : scratchLayout.offsets().entrySet()) {
                BufferRef buf = entry.getKey();
                long offset = entry.getValue();
                long size = buf.dataType().byteSizeFor(buf.shape());
                String name = "scratch" + slotId++;
                scratchBuffers.put(buf, new ScratchBufferVar(name, buf.dataType(), offset));
                usedTypes.add(buf.dataType());
                source.append("    MemorySegment ")
                        .append(name)
                        .append(" = scratchBase.asSlice(")
                        .append(offset)
                        .append("L, ")
                        .append(size)
                        .append("L);\n");
            }
        }

        private void emitInputs(StringBuilder source) {
            int argIndex = 0;
            for (LIRInput input : graph.inputs()) {
                if (input instanceof BufferRef buffer) {
                    String viewName = "inputView" + argIndex;
                    String memName = "input" + argIndex;
                    String offsetName = memName + "ByteOffset";
                    long[] byteStrides = buffer.byteStrides();
                    int rank = buffer.flatRank();
                    buffers.put(
                            buffer,
                            new RegularBufferVar(memName, buffer.dataType(), byteStrides, rank));
                    usedTypes.add(buffer.dataType());
                    // Get MemoryView from args
                    source.append(
                            "    MemoryView<MemorySegment> "
                                    + viewName
                                    + " = (MemoryView<MemorySegment>) args.getBuffer("
                                    + argIndex
                                    + ");\n");
                    // Extract Memory and offset from MemoryView
                    source.append(
                            "    Memory<MemorySegment> "
                                    + memName
                                    + " = "
                                    + viewName
                                    + ".memory();\n");
                    source.append("    long " + offsetName + " = " + viewName + ".byteOffset();\n");
                    // Extract byte strides for strided access
                    String[] strideVars = new String[rank];
                    if (rank > 0) {
                        if (generateStridedKernel) {
                            // Read strides from MemoryView at runtime for strided kernels
                            source.append(
                                    "    long[] "
                                            + memName
                                            + "Strides = "
                                            + viewName
                                            + ".byteStride().toArray();\n");
                            for (int i = 0; i < rank; i++) {
                                strideVars[i] = memName + "Strides[" + i + "]";
                            }
                        } else {
                            // Use compile-time constant strides for contiguous kernels
                            for (int i = 0; i < rank; i++) {
                                strideVars[i] = memName + "Stride" + i;
                                source.append(
                                        "    long "
                                                + strideVars[i]
                                                + " = "
                                                + byteStrides[i]
                                                + "L;\n");
                            }
                        }
                    }
                    bufferStrideVars.put(buffer, strideVars);
                } else if (input instanceof ScalarInput scalar) {
                    String varName = "scalar" + argIndex;
                    scalarInputs.put(scalar, varName);
                    scalarInputNames.put(scalar.id(), varName);
                    usedTypes.add(scalar.dataType());
                    source.append(
                            "    "
                                    + javaType(scalar.dataType())
                                    + " "
                                    + varName
                                    + " = "
                                    + scalarInputRead(scalar.dataType(), argIndex)
                                    + ";\n");
                }
                argIndex++;
            }
        }

        private void emitOutputs(StringBuilder source) {
            int argIndex = graph.inputs().size();
            for (int i = 0; i < graph.outputs().size(); i++) {
                BufferRef buffer = graph.outputs().get(i);
                String viewName = "outputView" + i;
                String memName = "output" + i;
                String offsetName = memName + "ByteOffset";
                long[] byteStrides = buffer.byteStrides();
                int rank = buffer.flatRank();
                buffers.put(
                        buffer,
                        new RegularBufferVar(memName, buffer.dataType(), byteStrides, rank));
                usedTypes.add(buffer.dataType());
                // Get MemoryView from args
                source.append(
                        "    MemoryView<MemorySegment> "
                                + viewName
                                + " = (MemoryView<MemorySegment>) args.getBuffer("
                                + argIndex
                                + ");\n");
                // Extract Memory and offset from MemoryView
                source.append(
                        "    Memory<MemorySegment> " + memName + " = " + viewName + ".memory();\n");
                source.append("    long " + offsetName + " = " + viewName + ".byteOffset();\n");
                // Extract byte strides for strided access
                String[] strideVars = new String[rank];
                if (rank > 0) {
                    if (generateStridedKernel) {
                        // Read strides from MemoryView at runtime for strided kernels
                        source.append(
                                "    long[] "
                                        + memName
                                        + "Strides = "
                                        + viewName
                                        + ".byteStride().toArray();\n");
                        for (int j = 0; j < rank; j++) {
                            strideVars[j] = memName + "Strides[" + j + "]";
                        }
                    } else {
                        // Use compile-time constant strides for contiguous kernels
                        for (int j = 0; j < rank; j++) {
                            strideVars[j] = memName + "Stride" + j;
                            source.append(
                                    "    long " + strideVars[j] + " = " + byteStrides[j] + "L;\n");
                        }
                    }
                }
                bufferStrideVars.put(buffer, strideVars);
                argIndex++;
            }
        }

        private void emitNode(LIRExprNode node) {
            switch (node) {
                case Block block -> {
                    for (LIRExprNode stmt : block.statements()) {
                        emitNode(stmt);
                    }
                }
                case Store store -> emitStore(store);
                case StructuredFor loop -> emitStructuredFor(loop);
                case Yield yield -> emitYield(yield);
                default -> {}
            }
        }

        private void emitStructuredFor(StructuredFor loop) {
            String idx = loop.indexName();
            String lb = emitIndexExpr(loop.lowerBound());
            String ub = emitIndexExpr(loop.upperBound());
            String step = emitIndexExpr(loop.step());

            // Check if this is a simple forward loop with step = 1
            boolean isSimpleForwardLoop =
                    step.equals("1") || (loop.step() instanceof IConst ic && ic.value() == 1);

            if (isSimpleForwardLoop) {
                // Simplified loop without extra variables
                for (LoopIterArg arg : loop.iterArgs()) {
                    String initExpr = emitScalarExpr(arg.init());
                    addLine(javaType(arg.dataType()) + " " + arg.name() + " = " + initExpr + ";");
                }

                addLine(
                        "for (long "
                                + idx
                                + " = "
                                + lb
                                + "; "
                                + idx
                                + " < "
                                + ub
                                + "; "
                                + idx
                                + "++) {");
                indentLevel++;
                emitStructuredBody(loop);
                indentLevel--;
                addLine("}");
            } else {
                // Full loop with step variable for non-trivial cases
                String lbVar = idx + "_lb";
                String ubVar = idx + "_ub";
                String stepVar = idx + "_step";
                addLine("long " + lbVar + " = " + lb + ";");
                addLine("long " + ubVar + " = " + ub + ";");
                addLine("long " + stepVar + " = " + step + ";");

                for (LoopIterArg arg : loop.iterArgs()) {
                    String initExpr = emitScalarExpr(arg.init());
                    addLine(javaType(arg.dataType()) + " " + arg.name() + " = " + initExpr + ";");
                }

                addLine(
                        "for (long "
                                + idx
                                + " = "
                                + lbVar
                                + "; "
                                + stepVar
                                + " > 0 ? "
                                + idx
                                + " < "
                                + ubVar
                                + " : "
                                + idx
                                + " > "
                                + ubVar
                                + "; "
                                + idx
                                + " += "
                                + stepVar
                                + ") {");
                indentLevel++;
                emitStructuredBody(loop);
                indentLevel--;
                addLine("}");
            }
        }

        private void emitStructuredBody(StructuredFor loop) {
            Block body = loop.body();
            Yield yield = extractYield(body);
            for (int i = 0; i < body.statements().size() - 1; i++) {
                emitNode(body.statements().get(i));
            }

            List<String> nextNames = new ArrayList<>(loop.iterArgs().size());
            for (int i = 0; i < loop.iterArgs().size(); i++) {
                LoopIterArg arg = loop.iterArgs().get(i);
                String nextName = arg.name() + "_next" + tempId++;
                String expr = emitScalarExpr(yield.values().get(i));
                addLine(javaType(arg.dataType()) + " " + nextName + " = " + expr + ";");
                nextNames.add(nextName);
            }
            for (int i = 0; i < loop.iterArgs().size(); i++) {
                LoopIterArg arg = loop.iterArgs().get(i);
                addLine(arg.name() + " = " + nextNames.get(i) + ";");
            }
        }

        private void emitYield(Yield yield) {
            if (!yield.values().isEmpty()) {
                throw new IllegalStateException("Yield should be handled by StructuredFor");
            }
        }

        private Yield extractYield(Block body) {
            if (!body.statements().isEmpty()) {
                LIRExprNode last = body.statements().getLast();
                if (last instanceof Yield yield) {
                    return yield;
                }
            }
            throw new IllegalStateException("Structured loop body must end with Yield");
        }

        private void emitStore(Store store) {
            BufferVar buffer = requireBuffer(store.buffer());
            String value = emitScalarExpr(store.value());
            DataType bufferType = store.buffer().dataType();
            DataType valueType = store.value().dataType();
            String offset = emitIndexWithStridesExpr(store.buffer(), store.offset());
            addLine(writeValue(buffer, bufferType, offset, value, valueType));
        }

        private String emitScalarExpr(LIRExprNode node) {
            LIRExprNode resolved = exprGraph.resolve(node);
            usedTypes.add(resolved.dataType());
            if (resolved.kind() == LIRExprKind.S_REF
                    || resolved.kind() == LIRExprKind.S_INPUT
                    || resolved.kind() == LIRExprKind.S_LOAD
                    || resolved.kind() == LIRExprKind.S_FROM_INDEX) {
                return emitScalarExprInline(resolved);
            }
            if ((resolved.kind() == LIRExprKind.S_TERNARY || resolved.useCount() > 1)
                    && (resolved.kind() == LIRExprKind.S_UNARY
                            || resolved.kind() == LIRExprKind.S_BINARY
                            || resolved.kind() == LIRExprKind.S_TERNARY
                            || resolved.kind() == LIRExprKind.S_CAST)) {
                String cached = tempNames.get(resolved);
                if (cached != null) {
                    return cached;
                }
                String expr = emitScalarExprInline(resolved);
                if (!shouldMaterializeTemp(resolved, expr)) {
                    return expr;
                }
                String var = nextTempName();
                if (var.equals(expr)) {
                    return expr;
                }
                addLine(javaType(resolved.dataType()) + " " + var + " = " + expr + ";");
                tempNames.put(resolved, var);
                return var;
            }
            return emitScalarExprInline(resolved);
        }

        private String emitScalarExprInline(LIRExprNode resolved) {
            return switch (resolved.kind()) {
                case S_CONST -> scalarLiteral(((SConst) resolved).rawBits(), resolved.dataType());
                case S_INPUT ->
                        requireScalarInputById(((SInput) resolved).inputId(), resolved.dataType());
                case S_REF -> {
                    String refName = ((SRef) resolved).name();
                    String mapped = scalarNames.get(refName);
                    yield mapped != null ? mapped : refName;
                }
                case S_FROM_INDEX -> {
                    String indexExpr = emitIndexExpr(((SFromIndex) resolved).indexExpr());
                    if (resolved.dataType() == DataType.I64) {
                        yield indexExpr;
                    }
                    yield castExpr(DataType.I64, resolved.dataType(), indexExpr);
                }
                case S_LOAD -> emitScalarLoadExpr((SLoad) resolved);
                case S_UNARY -> emitUnaryExpr((SUnary) resolved);
                case S_BINARY -> emitBinaryExpr((SBinary) resolved);
                case S_TERNARY -> emitTernaryExpr((STernary) resolved);
                case S_CAST -> emitCastExpr((SCast) resolved);
                default ->
                        throw new IllegalStateException(
                                "Expected scalar node, got " + resolved.kind());
            };
        }

        private boolean shouldMaterializeTemp(LIRExprNode node, String expr) {
            if (node.kind() == LIRExprKind.S_TERNARY) {
                return true;
            }
            if (node.useCount() > 2) {
                return true;
            }
            if (expr.length() <= 80) {
                return false;
            }
            return switch (node.kind()) {
                case S_UNARY, S_BINARY, S_CAST -> true;
                default -> false;
            };
        }

        private String emitScalarLoadExpr(SLoad load) {
            BufferVar buffer = requireBuffer(load.buffer());
            String offset = emitIndexWithStridesExpr(load.buffer(), load.offset());
            return readValue(buffer, load.buffer().dataType(), offset);
        }

        private String emitUnaryExpr(SUnary unary) {
            String input = emitScalarExpr(unary.input());
            DataType type = unary.dataType();
            return switch (unary.op()) {
                case NEGATE -> "-" + input;
                case ABS -> "Math.abs(" + input + ")";
                case EXP -> castFloating(type, "Math.exp(" + input + ")");
                case LOG -> castFloating(type, "Math.log(" + input + ")");
                case SQRT -> castFloating(type, "Math.sqrt(" + input + ")");
                case SQUARE -> input + " * " + input;
                case SIN -> castFloating(type, "Math.sin(" + input + ")");
                case COS -> castFloating(type, "Math.cos(" + input + ")");
                case TAN -> castFloating(type, "Math.tan(" + input + ")");
                case TANH -> castFloating(type, "Math.tanh(" + input + ")");
                case RECIPROCAL -> reciprocalFor(type, input);
                case LOGICAL_NOT ->
                        unary.input().dataType() == DataType.BOOL
                                ? "!(" + input + ")"
                                : "!(" + input + " != 0)";
                case BITWISE_NOT -> bitwiseNotFor(type, input);
            };
        }

        private String emitBinaryExpr(SBinary binary) {
            String left = emitScalarExpr(binary.left());
            String right = emitScalarExpr(binary.right());
            DataType type = binary.dataType();
            if (binary.op() == BinaryOperator.ADD) {
                String l = maybeConvertForBinaryExpr(binary.left(), left, type);
                String r = maybeConvertForBinaryExpr(binary.right(), right, type);
                return "(" + l + " + " + r + ")";
            }
            if (binary.op() == BinaryOperator.SUBTRACT) {
                return "(" + left + " - " + right + ")";
            }
            if (binary.op() == BinaryOperator.MULTIPLY) {
                return "(" + left + " * " + right + ")";
            }
            if (binary.op() == BinaryOperator.DIVIDE) {
                return "(" + left + " / " + right + ")";
            }
            if (binary.op() == BinaryOperator.MIN) {
                return minExpr(type, left, right);
            }
            if (binary.op() == BinaryOperator.MAX) {
                return maxExpr(type, left, right);
            }
            if (binary.op() == BinaryOperator.POW) {
                return powExpr(type, left, right);
            }
            if (binary.op() == BinaryOperator.LOGICAL_AND) {
                String l = binary.left().dataType() != DataType.BOOL ? "(" + left + " != 0)" : left;
                String r =
                        binary.right().dataType() != DataType.BOOL ? "(" + right + " != 0)" : right;
                String result = "(" + l + " && " + r + ")";
                return type == DataType.BOOL ? result : "(" + result + " ? 1 : 0)";
            }
            if (binary.op() == BinaryOperator.LOGICAL_OR) {
                String l = binary.left().dataType() != DataType.BOOL ? "(" + left + " != 0)" : left;
                String r =
                        binary.right().dataType() != DataType.BOOL ? "(" + right + " != 0)" : right;
                String result = "(" + l + " || " + r + ")";
                return type == DataType.BOOL ? result : "(" + result + " ? 1 : 0)";
            }
            if (binary.op() == BinaryOperator.LOGICAL_XOR) {
                String l = binary.left().dataType() != DataType.BOOL ? "(" + left + " != 0)" : left;
                String r =
                        binary.right().dataType() != DataType.BOOL ? "(" + right + " != 0)" : right;
                String result = "(" + l + " ^ " + r + ")";
                return type == DataType.BOOL ? result : "(" + result + " ? 1 : 0)";
            }
            if (binary.op() == BinaryOperator.BITWISE_AND) {
                return castFor(type, "(" + left + " & " + right + ")");
            }
            if (binary.op() == BinaryOperator.BITWISE_OR) {
                return castFor(type, "(" + left + " | " + right + ")");
            }
            if (binary.op() == BinaryOperator.BITWISE_XOR) {
                return castFor(type, "(" + left + " ^ " + right + ")");
            }
            if (binary.op() == BinaryOperator.EQUAL) {
                return compareExpr("==", binary.left(), binary.right(), type);
            }
            if (binary.op() == BinaryOperator.LESS_THAN) {
                return compareExpr("<", binary.left(), binary.right(), type);
            }
            throw new UnsupportedOperationException("Unsupported binary operator: " + binary.op());
        }

        private String maybeConvertForBinaryExpr(
                LIRExprNode node, String exprStr, DataType targetType) {
            if (node.dataType() == DataType.BOOL && targetType != DataType.BOOL) {
                return "(" + exprStr + " ? 1 : 0)";
            }
            return exprStr;
        }

        private String emitTernaryExpr(STernary ternary) {
            String cond = emitScalarExpr(ternary.condition());
            if (ternary.condition().dataType() != DataType.BOOL) {
                cond = cond + " != 0";
            }
            String tVal = emitScalarExpr(ternary.trueValue());
            String fVal = emitScalarExpr(ternary.falseValue());
            return "(" + cond + " ? " + tVal + " : " + fVal + ")";
        }

        private String emitCastExpr(SCast cast) {
            String input = emitScalarExpr(cast.input());
            return castExpr(cast.input().dataType(), cast.targetType(), input);
        }

        private String emitIndexExpr(LIRExprNode node) {
            LIRExprNode resolved = exprGraph.resolve(node);
            return switch (resolved.kind()) {
                case I_CONST -> Long.toString(((IConst) resolved).value());
                case I_VAR -> ((IVar) resolved).name();
                case I_BINARY ->
                        "("
                                + emitIndexExpr(((IBinary) resolved).left())
                                + " "
                                + indexOp(((IBinary) resolved).op())
                                + " "
                                + emitIndexExpr(((IBinary) resolved).right())
                                + ")";
                case I_FROM_SCALAR -> emitIndexFromScalarExpr((IFromScalar) resolved);
                default ->
                        throw new IllegalStateException(
                                "Expected index node, got " + resolved.kind());
            };
        }

        private String emitIndexWithStridesExpr(BufferRef buffer, LIRExprNode node) {
            String[] strideVars = bufferStrideVars.get(buffer);
            long[] byteStrides = buffer.byteStrides();
            return emitIndexWithStrideReplacementExpr(node, byteStrides, strideVars);
        }

        private String emitIndexWithStrideReplacementExpr(
                LIRExprNode node, long[] strides, String[] strideVars) {
            LIRExprNode resolved = exprGraph.resolve(node);
            return switch (resolved.kind()) {
                case I_CONST -> {
                    long value = ((IConst) resolved).value();
                    String replacement = findStrideVariable(value, strides, strideVars);
                    yield replacement != null ? replacement : Long.toString(value);
                }
                case I_VAR -> ((IVar) resolved).name();
                case I_BINARY -> {
                    String left =
                            emitIndexWithStrideReplacementExpr(
                                    ((IBinary) resolved).left(), strides, strideVars);
                    String right =
                            emitIndexWithStrideReplacementExpr(
                                    ((IBinary) resolved).right(), strides, strideVars);
                    yield "(" + left + " " + indexOp(((IBinary) resolved).op()) + " " + right + ")";
                }
                case I_FROM_SCALAR -> emitIndexFromScalarExpr((IFromScalar) resolved);
                default ->
                        throw new IllegalStateException(
                                "Expected index node, got " + resolved.kind());
            };
        }

        private String emitIndexFromScalarExpr(IFromScalar fromScalar) {
            String scalarExpr = emitScalarExpr(fromScalar.scalarExpr());
            DataType sourceType = fromScalar.scalarExpr().dataType();
            if (sourceType == DataType.BOOL) {
                return "(" + scalarExpr + " ? 1L : 0L)";
            }
            if (sourceType == DataType.I64) {
                return scalarExpr;
            }
            return "(long) (" + scalarExpr + ")";
        }

        private String compareExpr(
                String op, LIRExprNode leftExpr, LIRExprNode rightExpr, DataType resultType) {
            DataType leftType = leftExpr.dataType();
            DataType rightType = rightExpr.dataType();
            String left = emitScalarExpr(leftExpr);
            String right = emitScalarExpr(rightExpr);

            if (op.equals("<") && (leftType == DataType.BOOL || rightType == DataType.BOOL)) {
                left = castExpr(leftType, DataType.I32, left);
                right = castExpr(rightType, DataType.I32, right);
                leftType = DataType.I32;
                rightType = DataType.I32;
            }

            if (leftType == DataType.BOOL && rightType != DataType.BOOL) {
                left = castExpr(leftType, rightType, left);
                leftType = rightType;
            } else if (rightType == DataType.BOOL && leftType != DataType.BOOL) {
                right = castExpr(rightType, leftType, right);
                rightType = leftType;
            }

            String expr = "(" + left + " " + op + " " + right + ")";
            if (resultType == DataType.BOOL) {
                return expr;
            }
            return "(" + expr + " ? 1 : 0)";
        }

        private String indexOp(IndexBinaryOp op) {
            return switch (op) {
                case ADD -> "+";
                case SUBTRACT -> "-";
                case MULTIPLY -> "*";
                case DIVIDE -> "/";
                case MODULO -> "%";
                case BITWISE_AND -> "&";
                case SHIFT_LEFT -> "<<";
                case SHIFT_RIGHT -> ">>";
            };
        }

        /**
         * Finds if a constant value matches a stride and returns the corresponding variable name.
         */
        private String findStrideVariable(long value, long[] strides, String[] strideVars) {
            if (strides == null || strideVars == null) {
                return null;
            }
            for (int i = 0; i < strides.length; i++) {
                if (strides[i] == value && strideVars[i] != null) {
                    return strideVars[i];
                }
            }
            return null;
        }

        private BufferVar requireBuffer(BufferRef ref) {
            // Check scratch buffers first
            ScratchBufferVar scratchBuffer = scratchBuffers.get(ref);
            if (scratchBuffer != null) {
                return scratchBuffer;
            }
            // Then check regular buffers
            BufferVar buffer = buffers.get(ref);
            if (buffer == null) {
                throw new IllegalStateException("Unknown buffer: " + ref);
            }
            return buffer;
        }

        private String requireScalarInput(ScalarInput input) {
            String name = scalarInputs.get(input);
            if (name == null) {
                throw new IllegalStateException("Unknown scalar input: " + input.id());
            }
            return name;
        }

        private String requireScalarInputById(int id, DataType dataType) {
            String name = scalarInputNames.get(id);
            if (name == null) {
                throw new IllegalStateException("Unknown scalar input id: " + id);
            }
            usedTypes.add(dataType);
            return name;
        }

        private String readValue(BufferVar buffer, DataType type, String offset) {
            if (buffer.isScratch()) {
                // Scratch buffers use direct MemorySegment access
                return readValueFromSegment(buffer.name(), type, offset);
            }
            // Regular buffers use MemoryAccess
            String memName = buffer.name();
            String address = memName + "ByteOffset + " + offset;
            if (type == DataType.FP32) {
                return "access.readFloat(" + memName + ", " + address + ")";
            }
            if (type == DataType.FP64) {
                return "access.readDouble(" + memName + ", " + address + ")";
            }
            if (type == DataType.FP16) {
                return "Float.float16ToFloat(access.readShort(" + memName + ", " + address + "))";
            }
            if (type == DataType.BF16) {
                return "BFloat16.toFloat(access.readShort(" + memName + ", " + address + "))";
            }
            if (type == DataType.I8) {
                return "access.readByte(" + memName + ", " + address + ")";
            }
            if (type == DataType.I16) {
                return "access.readShort(" + memName + ", " + address + ")";
            }
            if (type == DataType.I32) {
                return "access.readInt(" + memName + ", " + address + ")";
            }
            if (type == DataType.I64) {
                return "access.readLong(" + memName + ", " + address + ")";
            }
            if (type == DataType.BOOL) {
                // For BOOL, return boolean expression for logical ops, convert to 0/1 for
                // arithmetic
                // Use nested ternary to avoid int+boolean errors
                return "(access.readByte(" + memName + ", " + address + ") != 0)";
            }
            throw new UnsupportedOperationException("Unsupported dtype: " + type);
        }

        private String readValueFromSegment(String segName, DataType type, String offset) {
            if (type == DataType.FP32) {
                return segName + ".get(ValueLayout.JAVA_FLOAT_UNALIGNED, " + offset + ")";
            }
            if (type == DataType.FP64) {
                return segName + ".get(ValueLayout.JAVA_DOUBLE_UNALIGNED, " + offset + ")";
            }
            if (type == DataType.FP16) {
                return "Float.float16ToFloat("
                        + segName
                        + ".get(ValueLayout.JAVA_SHORT_UNALIGNED, "
                        + offset
                        + "))";
            }
            if (type == DataType.BF16) {
                return "BFloat16.toFloat("
                        + segName
                        + ".get(ValueLayout.JAVA_SHORT_UNALIGNED, "
                        + offset
                        + "))";
            }
            if (type == DataType.I8) {
                return segName + ".get(ValueLayout.JAVA_BYTE, " + offset + ")";
            }
            if (type == DataType.I16) {
                return segName + ".get(ValueLayout.JAVA_SHORT_UNALIGNED, " + offset + ")";
            }
            if (type == DataType.I32) {
                return segName + ".get(ValueLayout.JAVA_INT_UNALIGNED, " + offset + ")";
            }
            if (type == DataType.I64) {
                return segName + ".get(ValueLayout.JAVA_LONG_UNALIGNED, " + offset + ")";
            }
            if (type == DataType.BOOL) {
                // For BOOL, return boolean expression for logical ops
                return "(" + segName + ".get(ValueLayout.JAVA_BYTE, " + offset + ") != 0)";
            }
            throw new UnsupportedOperationException("Unsupported dtype: " + type);
        }

        private String writeValue(
                BufferVar buffer, DataType type, String offset, String value, DataType valueType) {
            if (buffer.isScratch()) {
                // Scratch buffers use direct MemorySegment access
                return writeValueToSegment(buffer.name(), type, offset, value, valueType);
            }
            // Regular buffers use MemoryAccess
            String memName = buffer.name();
            String address = memName + "ByteOffset + " + offset;
            if (type == DataType.FP32) {
                return "access.writeFloat(" + memName + ", " + address + ", " + value + ");";
            }
            if (type == DataType.FP64) {
                return "access.writeDouble(" + memName + ", " + address + ", " + value + ");";
            }
            if (type == DataType.FP16) {
                return "access.writeShort("
                        + memName
                        + ", "
                        + address
                        + ", Float.floatToFloat16("
                        + value
                        + "));";
            }
            if (type == DataType.BF16) {
                return "access.writeShort("
                        + memName
                        + ", "
                        + address
                        + ", BFloat16.fromFloat("
                        + value
                        + "));";
            }
            if (type == DataType.I8) {
                return "access.writeByte("
                        + memName
                        + ", "
                        + address
                        + ", (byte) ("
                        + value
                        + "));";
            }
            if (type == DataType.I16) {
                return "access.writeShort("
                        + memName
                        + ", "
                        + address
                        + ", (short) ("
                        + value
                        + "));";
            }
            if (type == DataType.I32) {
                return "access.writeInt(" + memName + ", " + address + ", (int) (" + value + "));";
            }
            if (type == DataType.I64) {
                return "access.writeLong("
                        + memName
                        + ", "
                        + address
                        + ", (long) ("
                        + value
                        + "));";
            }
            if (type == DataType.BOOL) {
                // Convert value to boolean - only add != 0 if value is not already boolean
                if (valueType == DataType.BOOL) {
                    return "access.writeByte("
                            + memName
                            + ", "
                            + address
                            + ", (byte) ("
                            + value
                            + " ? 1 : 0));";
                } else {
                    return "access.writeByte("
                            + memName
                            + ", "
                            + address
                            + ", (byte) ("
                            + value
                            + " != 0 ? 1 : 0));";
                }
            }
            throw new UnsupportedOperationException("Unsupported dtype: " + type);
        }

        private String writeValueToSegment(
                String segName, DataType type, String offset, String value, DataType valueType) {
            if (type == DataType.FP32) {
                return segName
                        + ".set(ValueLayout.JAVA_FLOAT_UNALIGNED, "
                        + offset
                        + ", "
                        + value
                        + ");";
            }
            if (type == DataType.FP64) {
                return segName
                        + ".set(ValueLayout.JAVA_DOUBLE_UNALIGNED, "
                        + offset
                        + ", "
                        + value
                        + ");";
            }
            if (type == DataType.FP16) {
                return segName
                        + ".set(ValueLayout.JAVA_SHORT_UNALIGNED, "
                        + offset
                        + ", Float.floatToFloat16("
                        + value
                        + "));";
            }
            if (type == DataType.BF16) {
                return segName
                        + ".set(ValueLayout.JAVA_SHORT_UNALIGNED, "
                        + offset
                        + ", BFloat16.fromFloat("
                        + value
                        + "));";
            }
            if (type == DataType.I8) {
                return segName
                        + ".set(ValueLayout.JAVA_BYTE, "
                        + offset
                        + ", (byte) ("
                        + value
                        + "));";
            }
            if (type == DataType.I16) {
                return segName
                        + ".set(ValueLayout.JAVA_SHORT_UNALIGNED, "
                        + offset
                        + ", (short) ("
                        + value
                        + "));";
            }
            if (type == DataType.I32) {
                return segName
                        + ".set(ValueLayout.JAVA_INT_UNALIGNED, "
                        + offset
                        + ", (int) ("
                        + value
                        + "));";
            }
            if (type == DataType.I64) {
                return segName
                        + ".set(ValueLayout.JAVA_LONG_UNALIGNED, "
                        + offset
                        + ", (long) ("
                        + value
                        + "));";
            }
            if (type == DataType.BOOL) {
                // Convert value to boolean - only add != 0 if value is not already boolean
                if (valueType == DataType.BOOL) {
                    return segName
                            + ".set(ValueLayout.JAVA_BYTE, "
                            + offset
                            + ", (byte) ("
                            + value
                            + " ? 1 : 0));";
                } else {
                    return segName
                            + ".set(ValueLayout.JAVA_BYTE, "
                            + offset
                            + ", (byte) ("
                            + value
                            + " != 0 ? 1 : 0));";
                }
            }
            throw new UnsupportedOperationException("Unsupported dtype: " + type);
        }

        private String scalarInputRead(DataType type, int index) {
            if (type == DataType.FP32 || type == DataType.FP16 || type == DataType.BF16) {
                return "args.getFloat(" + index + ")";
            }
            if (type == DataType.FP64) {
                return "args.getDouble(" + index + ")";
            }
            if (type == DataType.I8) {
                return "args.getByte(" + index + ")";
            }
            if (type == DataType.I16) {
                return "args.getShort(" + index + ")";
            }
            if (type == DataType.I32) {
                return "args.getInt(" + index + ")";
            }
            if (type == DataType.I64) {
                return "args.getLong(" + index + ")";
            }
            if (type == DataType.BOOL) {
                return "args.getBoolean(" + index + ")";
            }
            throw new UnsupportedOperationException("Unsupported dtype: " + type);
        }

        private String nextTempName() {
            String name;
            do {
                name = "t" + tempId++;
            } while (scalarNames.containsKey(name) || scalarInputNames.containsValue(name));
            return name;
        }

        private String scalarLiteral(long bits, DataType type) {
            if (type == DataType.FP32) {
                float value = Float.intBitsToFloat((int) bits);
                return formatFloatLiteral(value);
            }
            if (type == DataType.FP64) {
                double value = Double.longBitsToDouble(bits);
                return formatDoubleLiteral(value);
            }
            if (type == DataType.FP16) {
                return "Float.float16ToFloat((short) " + bits + "L)";
            }
            if (type == DataType.BF16) {
                return "BFloat16.toFloat((short) " + bits + "L)";
            }
            if (type == DataType.I8) {
                return "(byte) " + bits + "L";
            }
            if (type == DataType.I16) {
                return "(short) " + bits + "L";
            }
            if (type == DataType.I32) {
                return "(int) " + bits + "L";
            }
            if (type == DataType.I64) {
                return bits + "L";
            }
            if (type == DataType.BOOL) {
                return bits != 0 ? "true" : "false";
            }
            throw new UnsupportedOperationException("Unsupported literal dtype: " + type);
        }

        private String formatFloatLiteral(float value) {
            if (Float.isNaN(value)) {
                return "Float.NaN";
            }
            if (Float.isInfinite(value)) {
                return value > 0 ? "Float.POSITIVE_INFINITY" : "Float.NEGATIVE_INFINITY";
            }
            // Use toString for normal values, append 'f' suffix
            String str = Float.toString(value);
            if (!str.contains("E") && !str.contains("e")) {
                // For non-scientific notation, ensure it has a decimal point or 'f' suffix
                if (!str.contains(".")) {
                    str = str + ".0f";
                } else {
                    str = str + "f";
                }
            } else {
                str = str + "f";
            }
            return str;
        }

        private String formatDoubleLiteral(double value) {
            if (Double.isNaN(value)) {
                return "Double.NaN";
            }
            if (Double.isInfinite(value)) {
                return value > 0 ? "Double.POSITIVE_INFINITY" : "Double.NEGATIVE_INFINITY";
            }
            // Use toString for normal values
            String str = Double.toString(value);
            if (!str.contains("E") && !str.contains("e") && !str.contains(".")) {
                // For non-scientific notation without decimal, add .0
                str = str + ".0";
            }
            return str;
        }

        private String javaType(DataType type) {
            if (type == DataType.BOOL) {
                return "boolean";
            }
            if (type == DataType.I8) {
                return "byte";
            }
            if (type == DataType.I16) {
                return "short";
            }
            if (type == DataType.I32) {
                return "int";
            }
            if (type == DataType.I64) {
                return "long";
            }
            if (type == DataType.FP64) {
                return "double";
            }
            if (type == DataType.FP16 || type == DataType.BF16 || type == DataType.FP32) {
                return "float";
            }
            throw new UnsupportedOperationException("Unsupported dtype: " + type);
        }

        private String reciprocalFor(DataType type, String input) {
            if (type == DataType.FP64) {
                return "(1.0d / " + input + ")";
            }
            return "(1.0f / " + input + ")";
        }

        private String bitwiseNotFor(DataType type, String input) {
            if (type == DataType.I8) {
                return "(byte) (~" + input + ")";
            }
            if (type == DataType.I16) {
                return "(short) (~" + input + ")";
            }
            return "~" + input;
        }

        private String minExpr(DataType type, String left, String right) {
            if (type == DataType.BOOL) {
                // For booleans, MIN is logical AND (both must be true)
                return "(" + left + " && " + right + ")";
            }
            if (type == DataType.FP64) {
                return "Math.min(" + left + ", " + right + ")";
            }
            if (type == DataType.FP32 || type == DataType.FP16 || type == DataType.BF16) {
                return "Math.min(" + left + ", " + right + ")";
            }
            return "(" + left + " < " + right + " ? " + left + " : " + right + ")";
        }

        private String maxExpr(DataType type, String left, String right) {
            if (type == DataType.BOOL) {
                // For booleans, MAX is logical OR (either can be true)
                return "(" + left + " || " + right + ")";
            }
            if (type == DataType.FP64) {
                return "Math.max(" + left + ", " + right + ")";
            }
            if (type == DataType.FP32 || type == DataType.FP16 || type == DataType.BF16) {
                return "Math.max(" + left + ", " + right + ")";
            }
            return "(" + left + " > " + right + " ? " + left + " : " + right + ")";
        }

        private String powExpr(DataType type, String left, String right) {
            if (type == DataType.FP64) {
                return "Math.pow(" + left + ", " + right + ")";
            }
            return "(float) Math.pow(" + left + ", " + right + ")";
        }

        private String castExpr(DataType src, DataType dst, String input) {
            input = stripRedundantCast(src, input);
            if (dst == src) {
                return input;
            }
            if (dst == DataType.BOOL) {
                return "(" + input + " != 0)";
            }
            if (src == DataType.BOOL) {
                // Input is boolean (e.g., from comparison or ternary), convert to 0/1
                return "(" + input + " ? 1 : 0)";
            }
            if (dst == DataType.FP64) {
                return "(double) (" + input + ")";
            }
            if (dst == DataType.FP32 || dst == DataType.FP16 || dst == DataType.BF16) {
                return "(float) (" + input + ")";
            }
            if (dst == DataType.I64) {
                return "(long) (" + input + ")";
            }
            if (dst == DataType.I32) {
                return "(int) (" + input + ")";
            }
            if (dst == DataType.I16) {
                return "(short) (" + input + ")";
            }
            if (dst == DataType.I8) {
                return "(byte) (" + input + ")";
            }
            throw new UnsupportedOperationException("Unsupported cast: " + src + " -> " + dst);
        }

        private String castFor(DataType type, String expr) {
            if (type == DataType.I8) {
                return "(byte) " + expr;
            }
            if (type == DataType.I16) {
                return "(short) " + expr;
            }
            if (type == DataType.I32) {
                return "(int) " + expr;
            }
            if (type == DataType.I64) {
                return "(long) " + expr;
            }
            return expr;
        }

        private String castFloating(DataType type, String expr) {
            if (type == DataType.FP64) {
                return expr;
            }
            return "(float) (" + expr + ")";
        }

        private String stripRedundantCast(DataType src, String input) {
            String prefix = null;
            if (src == DataType.I64) {
                prefix = "(long) (";
            } else if (src == DataType.I32) {
                prefix = "(int) (";
            } else if (src == DataType.I16) {
                prefix = "(short) (";
            } else if (src == DataType.I8) {
                prefix = "(byte) (";
            } else if (src == DataType.FP64) {
                prefix = "(double) (";
            } else if (src == DataType.FP32 || src == DataType.FP16 || src == DataType.BF16) {
                prefix = "(float) (";
            }
            if (prefix == null) {
                return input;
            }
            if (input.startsWith(prefix) && input.endsWith(")")) {
                return input.substring(prefix.length(), input.length() - 1);
            }
            return input;
        }

        private void emitHelpers(StringBuilder source) {
            // Only emit toFloat helper if FP16 or BF16 types are used
            if (usedTypes.contains(DataType.FP16) || usedTypes.contains(DataType.BF16)) {
                source.append("  private static float toFloat(short bits) {\n");
                source.append("    return Float.float16ToFloat(bits);\n");
                source.append("  }\n");
            }
        }

        private void addLine(String line) {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < indentLevel; i++) {
                builder.append("  ");
            }
            builder.append(line);
            lines.add(builder.toString());
        }

        /** Base interface for buffer variable references in generated code. */
        private sealed interface BufferVar permits RegularBufferVar, ScratchBufferVar {
            String name();

            DataType dataType();

            /** Returns true if this is a scratch buffer (direct segment access). */
            default boolean isScratch() {
                return false;
            }
        }

        /** Regular buffer backed by Memory (extracted from MemoryView). */
        private record RegularBufferVar(
                String name, DataType dataType, long[] byteStrides, int rank) implements BufferVar {
            RegularBufferVar(String name, DataType dataType) {
                this(name, dataType, null, 0);
            }

            RegularBufferVar(String name, DataType dataType, long[] byteStrides) {
                this(name, dataType, byteStrides, byteStrides != null ? byteStrides.length : 0);
            }

            boolean isStrided() {
                return byteStrides != null && byteStrides.length > 0;
            }
        }

        /** Scratch buffer backed by Memory wrapper around MemorySegment slice. */
        private record ScratchBufferVar(String name, DataType dataType, long byteOffset)
                implements BufferVar {
            @Override
            public boolean isScratch() {
                return true;
            }
        }
    }
}
