package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.IdentityHashMap;
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

final class JavaKernelCompiler {

    private final KernelCache cache;

    JavaKernelCompiler(KernelCache cache) {
        this.cache = Objects.requireNonNull(cache, "cache");
    }

    JitKernel compile(ExpressionGraph graph, KernelStyle style) {
        KernelCacheKey key = buildCacheKey(graph, style);
        KernelCacheEntry entry = cache.entryFor(key);
        try {
            Files.createDirectories(entry.classOutputDir());
            Files.createDirectories(entry.directory());
            if (Files.notExists(entry.sourcePath())) {
                String source = KernelSourceGenerator.generate(entry, graph, style);
                Files.writeString(entry.sourcePath(), source);
            }
            if (Files.exists(entry.classFilePath())) {
                return load(entry);
            }
            compileSource(entry);
            return load(entry);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to compile kernel", e);
        }
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
                            System.getProperty("java.class.path"),
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

    private JitKernel load(KernelCacheEntry entry) {
        try {
            URLClassLoader loader =
                    new URLClassLoader(
                            new URL[] {entry.classOutputDir().toUri().toURL()},
                            JitKernel.class.getClassLoader());
            Class<?> clazz =
                    Class.forName(entry.packageName() + "." + entry.className(), true, loader);
            Object instance = clazz.getDeclaredConstructor().newInstance();
            return (JitKernel) instance;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load kernel " + entry.className(), e);
        }
    }

    private KernelCacheKey buildCacheKey(ExpressionGraph graph, KernelStyle style) {
        KernelCacheKey baseKey = GraphHasher.hash(graph);
        String suffix = style == null ? "unknown" : style.name().toLowerCase(Locale.ROOT);
            return KernelCacheKey.of(baseKey.value() + "-" + suffix + "-v5");

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
        private final ExpressionGraph graph;
        private final KernelStyle style;
        private final Map<ExprNode, String> names = new IdentityHashMap<>();
        private final List<String> lines = new ArrayList<>();
        private final Set<HelperMethod> helpers = new HashSet<>();
        private int counter;

        private KernelSourceGenerator(
                KernelCacheEntry entry, ExpressionGraph graph, KernelStyle style) {
            this.entry = entry;
            this.graph = graph;
            this.style = style;
        }

        static String generate(KernelCacheEntry entry, ExpressionGraph graph, KernelStyle style) {
            KernelSourceGenerator generator = new KernelSourceGenerator(entry, graph, style);
            return generator.generate();
        }

        private String generate() {
            StringBuilder source = new StringBuilder();
            source.append("package ").append(entry.packageName()).append(";\n\n");
            source.append("import ai.qxotic.jota.BFloat16;\n");
            source.append("import ai.qxotic.jota.Indexing;\n");
            source.append("import ai.qxotic.jota.Shape;\n");
            source.append("import ai.qxotic.jota.memory.MemoryContext;\n");
            source.append("import ai.qxotic.jota.memory.MemoryView;\n");
            source.append("import ai.qxotic.jota.tensor.JitKernel;\n");
            source.append("import java.lang.foreign.MemorySegment;\n");
            source.append("import java.lang.foreign.ValueLayout;\n");
            source.append("\n");
            source.append("public final class ")
                    .append(entry.className())
                    .append(" implements JitKernel {\n");
            source.append("  @Override\n");
            source.append(
                    "  public void execute(MemoryContext<MemorySegment> context, MemoryView<MemorySegment>[] inputs, MemoryView<MemorySegment> output) {\n");

            ReductionNode reductionRoot = findReductionRoot(graph.root());
            if (reductionRoot != null) {
                appendReductionBody(source, graph.root(), reductionRoot);
                source.append("  }\n");
                appendHelpers(source);
                source.append("}\n");
                return source.toString();
            }

            String outputVar = emitNode(graph.root());
            if (style == KernelStyle.STRIDED) {
                source.append("    long[] shape = output.shape().toArray();\n");
                source.append("    long[] outStride = output.byteStride().toArray();\n");
                source.append("    long outBaseOffset = output.byteOffset();\n");
                source.append(
                        "    MemorySegment outBase = (MemorySegment) output.memory().base();\n");
                for (int i = 0; i < graph.inputs().size(); i++) {
                    source.append(
                            "    MemorySegment in"
                                    + i
                                    + " = (MemorySegment) inputs["
                                    + i
                                    + "].memory().base();\n");
                    source.append(
                            "    long in" + i + "BaseOffset = inputs[" + i + "].byteOffset();\n");
                    source.append(
                            "    long[] in"
                                    + i
                                    + "Stride = inputs["
                                    + i
                                    + "].byteStride().toArray();\n");
                }
            }
            source.append("    long size = output.shape().size();\n");
            source.append("    for (long i = 0; i < size; i++) {\n");
            for (String line : lines) {
                source.append("      ").append(line).append("\n");
            }
            source.append("      ").append(writeOutput(graph.root(), outputVar)).append("\n");
            source.append("    }\n");
            source.append("  }\n");
            appendHelpers(source);
            source.append("}\n");
            return source.toString();
        }

        private ReductionNode findReductionRoot(ExprNode root) {
            List<ReductionNode> reductions = new ArrayList<>();
            Deque<ExprNode> stack = new ArrayDeque<>();
            Set<ExprNode> visited = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
            stack.push(root);
            while (!stack.isEmpty()) {
                ExprNode node = stack.pop();
                if (!visited.add(node)) {
                    continue;
                }
                if (node instanceof ReductionNode reduction) {
                    reductions.add(reduction);
                    stack.push(reduction.input());
                    continue;
                }
                if (node instanceof UnaryNode unary) {
                    stack.push(unary.input());
                } else if (node instanceof BinaryNode binary) {
                    stack.push(binary.left());
                    stack.push(binary.right());
                } else if (node instanceof TernaryNode ternary) {
                    stack.push(ternary.condition());
                    stack.push(ternary.trueValue());
                    stack.push(ternary.falseValue());
                } else if (node instanceof CastNode cast) {
                    stack.push(cast.input());
                } else if (node instanceof InputNode || node instanceof ScalarNode) {
                    continue;
                } else {
                    throw new IllegalStateException("Unsupported node: " + node);
                }
            }
            if (reductions.isEmpty()) {
                return null;
            }
            ReductionNode rootReduction = reductions.get(0);
            Set<ReductionNode> chain = new HashSet<>();
            ReductionOp op = rootReduction.op();
            boolean keepDims = rootReduction.keepDims();
            ReductionNode current = rootReduction;
            while (true) {
                chain.add(current);
                ExprNode input = current.input();
                if (input instanceof ReductionNode next
                        && next.op() == op
                        && next.keepDims() == keepDims) {
                    current = next;
                } else {
                    break;
                }
            }
            for (ReductionNode reduction : reductions) {
                if (!chain.contains(reduction)) {
                    throw new IllegalStateException(
                            "Multiple reductions in a single kernel are not supported");
                }
            }
            return rootReduction;
        }

        private void appendReductionBody(
                StringBuilder source, ExprNode root, ReductionNode reduction) {
            ReductionInfo info = collectReductionInfo(reduction);
            if (info.op() != ReductionOp.SUM
                    && info.op() != ReductionOp.PROD
                    && info.op() != ReductionOp.MIN
                    && info.op() != ReductionOp.MAX) {
                throw new IllegalStateException("Unsupported reduction op: " + info.op());
            }
            appendFoldReductionBody(source, root, reduction, info);
        }

        private void appendStridedInputPreamble(StringBuilder source) {
            source.append("    long[] shape = output.shape().toArray();\n");
            for (int i = 0; i < graph.inputs().size(); i++) {
                source.append(
                        "    MemorySegment in"
                                + i
                                + " = (MemorySegment) inputs["
                                + i
                                + "].memory().base();\n");
                source.append("    long in" + i + "BaseOffset = inputs[" + i + "].byteOffset();\n");
                source.append(
                        "    long[] in"
                                + i
                                + "Stride = inputs["
                                + i
                                + "].byteStride().toArray();\n");
            }
        }

        private void appendPostReductionOutput(
                StringBuilder source, ExprNode root, ReductionNode reduction) {
            Map<ExprNode, String> overrides = new IdentityHashMap<>();
            overrides.put(reduction, reductionResultExpression(reduction.dataType(), "acc"));
            String outputExpr = expressionForNode(root, "i", overrides);
            source.append("      ")
                    .append(typeFor(root.dataType()))
                    .append(" outValue = ")
                    .append(outputExpr)
                    .append(";\n");
            source.append("      long outOffset = Indexing.linearToOffset(output, i);\n");
            source.append("      ")
                    .append(writeReductionOutput(root.dataType(), "outValue"))
                    .append("\n");
        }

        private void appendFoldReductionBody(
                StringBuilder source, ExprNode root, ReductionNode reduction, ReductionInfo info) {
            boolean expressionInput = !(info.input() instanceof InputNode);
            DataType inputType = info.input().dataType();
            DataType accumulatorType = reductionAccumulatorType(info.op(), info.dataType());
            String accumulatorJavaType = typeFor(accumulatorType);
            String seedExpr = seedFor(info.op(), accumulatorType);
            String combineExpr = combineExpression(info.op(), accumulatorType);
            source.append("    MemoryView<MemorySegment> input = inputs[0];\n");
            source.append("    long[] inShape = input.shape().toArray();\n");
            source.append("    int[] axes = new int[] {");
            for (int i = 0; i < info.axes().length; i++) {
                if (i > 0) {
                    source.append(", ");
                }
                source.append(info.axes()[i]);
            }
            source.append("};\n");
            source.append("    long[] reduceDims = new long[axes.length];\n");
            source.append("    for (int i = 0; i < axes.length; i++) {\n");
            source.append("      reduceDims[i] = inShape[axes[i]];\n");
            source.append("    }\n");
            source.append("    Shape reduceShape = Shape.flat(reduceDims);\n");
            source.append("    long reduceSize = reduceShape.size();\n");
            source.append("    MemorySegment outBase = (MemorySegment) output.memory().base();\n");
            if (style == KernelStyle.STRIDED) {
                appendStridedInputPreamble(source);
            }
            if (expressionInput && inputType != DataType.FP32 && inputType != DataType.I32) {
                source.append(
                        "    throw new IllegalStateException(\"Reduction expressions are only supported for FP32/I32\");\n");
                return;
            }
            source.append("    long outSize = output.shape().size();\n");
            source.append("    for (long i = 0; i < outSize; i++) {\n");
            source.append("      long[] outCoord = Indexing.linearToCoord(output.shape(), i);\n");
            source.append("      long[] inCoord = new long[inShape.length];\n");
            if (info.keepDims()) {
                source.append("      for (int dim = 0; dim < inShape.length; dim++) {\n");
                source.append("        inCoord[dim] = outCoord[dim];\n");
                source.append("      }\n");
            } else {
                source.append("      int outDim = 0;\n");
                source.append("      for (int dim = 0; dim < inShape.length; dim++) {\n");
                source.append("        boolean reduced = false;\n");
                source.append("        for (int ax : axes) {\n");
                source.append("          if (ax == dim) { reduced = true; break; }\n");
                source.append("        }\n");
                source.append("        if (reduced) {\n");
                source.append("          inCoord[dim] = 0;\n");
                source.append("        } else {\n");
                source.append("          inCoord[dim] = outCoord[outDim++];\n");
                source.append("        }\n");
                source.append("      }\n");
            }
            source.append("      " + accumulatorJavaType + " acc = " + seedExpr + ";\n");
            source.append("      for (long r = 0; r < reduceSize; r++) {\n");
            source.append("        long[] reduceCoord = Indexing.linearToCoord(reduceShape, r);\n");
            source.append("        for (int j = 0; j < axes.length; j++) {\n");
            source.append("          inCoord[axes[j]] = reduceCoord[j];\n");
            source.append("        }\n");
            source.append(
                    "        long inputIndex = Indexing.coordToLinear(input.shape(), inCoord);\n");
            if (!expressionInput) {
                source.append(
                        "        long inputOffset = Indexing.linearToOffset(input, inputIndex);\n");
            }
            appendFoldValueRead(
                    source,
                    info.input(),
                    expressionInput,
                    inputType,
                    accumulatorType,
                    accumulatorJavaType);
            source.append("        ").append(combineExpr).append(";\n");
            source.append("      }\n");
            appendPostReductionOutput(source, root, reduction);
            source.append("    }\n");
        }

        private void appendFoldValueRead(
                StringBuilder source,
                ExprNode input,
                boolean expressionInput,
                DataType inputType,
                DataType accumulatorType,
                String accumulatorJavaType) {
            if (expressionInput) {
                String valueExpr = expressionForNode(input, "inputIndex");
                String castedValue = castToAccumulator(valueExpr, inputType, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            if (inputType == DataType.BOOL) {
                source.append(
                        "        byte rawValue = input.memory().base().get(ValueLayout.JAVA_BYTE, inputOffset);\n");
                if (accumulatorType == DataType.BOOL) {
                    source.append("        ")
                            .append(accumulatorJavaType)
                            .append(" value = rawValue;\n");
                } else {
                    String boolExpr = "rawValue == 0 ? 0 : 1";
                    String castedValue = castToAccumulator(boolExpr, DataType.I32, accumulatorType);
                    source.append("        ")
                            .append(accumulatorJavaType)
                            .append(" value = ")
                            .append(castedValue)
                            .append(";\n");
                }
                return;
            }
            if (inputType == DataType.I8) {
                source.append(
                        "        byte rawValue = input.memory().base().get(ValueLayout.JAVA_BYTE, inputOffset);\n");
                String castedValue = castToAccumulator("rawValue", inputType, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            if (inputType == DataType.I16) {
                source.append(
                        "        short rawValue = input.memory().base().get(ValueLayout.JAVA_SHORT_UNALIGNED, inputOffset);\n");
                String castedValue = castToAccumulator("rawValue", inputType, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            if (inputType == DataType.I32) {
                source.append(
                        "        int rawValue = input.memory().base().get(ValueLayout.JAVA_INT_UNALIGNED, inputOffset);\n");
                String castedValue = castToAccumulator("rawValue", inputType, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            if (inputType == DataType.I64) {
                source.append(
                        "        long rawValue = input.memory().base().get(ValueLayout.JAVA_LONG_UNALIGNED, inputOffset);\n");
                String castedValue = castToAccumulator("rawValue", inputType, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            if (inputType == DataType.FP16) {
                source.append(
                        "        short rawValueBits = input.memory().base().get(ValueLayout.JAVA_SHORT_UNALIGNED, inputOffset);\n");
                source.append("        float rawValue = Float.float16ToFloat(rawValueBits);\n");
                String castedValue = castToAccumulator("rawValue", DataType.FP32, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            if (inputType == DataType.BF16) {
                source.append(
                        "        short rawValueBits = input.memory().base().get(ValueLayout.JAVA_SHORT_UNALIGNED, inputOffset);\n");
                source.append("        float rawValue = BFloat16.toFloat(rawValueBits);\n");
                String castedValue = castToAccumulator("rawValue", DataType.FP32, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            if (inputType == DataType.FP32) {
                source.append(
                        "        float rawValue = input.memory().base().get(ValueLayout.JAVA_FLOAT_UNALIGNED, inputOffset);\n");
                String castedValue = castToAccumulator("rawValue", inputType, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            if (inputType == DataType.FP64) {
                source.append(
                        "        double rawValue = input.memory().base().get(ValueLayout.JAVA_DOUBLE_UNALIGNED, inputOffset);\n");
                String castedValue = castToAccumulator("rawValue", inputType, accumulatorType);
                source.append("        ")
                        .append(accumulatorJavaType)
                        .append(" value = ")
                        .append(castedValue)
                        .append(";\n");
                return;
            }
            source.append(
                    "        throw new IllegalStateException(\"Unsupported data type for reduction\");\n");
        }

        private DataType reductionAccumulatorType(ReductionOp op, DataType dataType) {
            if (op == ReductionOp.MIN || op == ReductionOp.MAX) {
                if (dataType == DataType.FP16 || dataType == DataType.BF16) {
                    return DataType.FP32;
                }
                return dataType;
            }
            return dataType;
        }

        private String seedFor(ReductionOp op, DataType accumulatorType) {
            if (op == ReductionOp.SUM) {
                if (accumulatorType == DataType.I32) {
                    return "0";
                }
                if (accumulatorType == DataType.I64) {
                    return "0L";
                }
                if (accumulatorType == DataType.FP32) {
                    return "0.0f";
                }
                if (accumulatorType == DataType.FP64) {
                    return "0.0d";
                }
                throw unsupported(accumulatorType, "sum accumulator");
            }
            if (op == ReductionOp.PROD) {
                if (accumulatorType == DataType.I32) {
                    return "1";
                }
                if (accumulatorType == DataType.I64) {
                    return "1L";
                }
                if (accumulatorType == DataType.FP32) {
                    return "1.0f";
                }
                if (accumulatorType == DataType.FP64) {
                    return "1.0d";
                }
                throw unsupported(accumulatorType, "product accumulator");
            }
            if (op == ReductionOp.MIN) {
                if (accumulatorType == DataType.BOOL) {
                    return "1";
                }
                if (accumulatorType == DataType.I8) {
                    return "Byte.MAX_VALUE";
                }
                if (accumulatorType == DataType.I16) {
                    return "Short.MAX_VALUE";
                }
                if (accumulatorType == DataType.I32) {
                    return "Integer.MAX_VALUE";
                }
                if (accumulatorType == DataType.I64) {
                    return "Long.MAX_VALUE";
                }
                if (accumulatorType == DataType.FP32) {
                    return "Float.POSITIVE_INFINITY";
                }
                if (accumulatorType == DataType.FP64) {
                    return "Double.POSITIVE_INFINITY";
                }
                throw unsupported(accumulatorType, "min accumulator");
            }
            if (op == ReductionOp.MAX) {
                if (accumulatorType == DataType.BOOL) {
                    return "0";
                }
                if (accumulatorType == DataType.I8) {
                    return "Byte.MIN_VALUE";
                }
                if (accumulatorType == DataType.I16) {
                    return "Short.MIN_VALUE";
                }
                if (accumulatorType == DataType.I32) {
                    return "Integer.MIN_VALUE";
                }
                if (accumulatorType == DataType.I64) {
                    return "Long.MIN_VALUE";
                }
                if (accumulatorType == DataType.FP32) {
                    return "Float.NEGATIVE_INFINITY";
                }
                if (accumulatorType == DataType.FP64) {
                    return "Double.NEGATIVE_INFINITY";
                }
                throw unsupported(accumulatorType, "max accumulator");
            }
            throw new IllegalStateException("Unsupported reduction op: " + op);
        }

        private String combineExpression(ReductionOp op, DataType accumulatorType) {
            if (op == ReductionOp.SUM) {
                return "acc += value";
            }
            if (op == ReductionOp.PROD) {
                return "acc *= value";
            }
            String function = op == ReductionOp.MIN ? "Math.min" : "Math.max";
            if (accumulatorType == DataType.BOOL || accumulatorType == DataType.I8) {
                return "acc = (byte) " + function + "(acc, value)";
            }
            if (accumulatorType == DataType.I16) {
                return "acc = (short) " + function + "(acc, value)";
            }
            if (accumulatorType == DataType.I32) {
                return "acc = " + function + "(acc, value)";
            }
            if (accumulatorType == DataType.I64) {
                return "acc = " + function + "(acc, value)";
            }
            if (accumulatorType == DataType.FP32) {
                return "acc = " + function + "(acc, value)";
            }
            if (accumulatorType == DataType.FP64) {
                return "acc = " + function + "(acc, value)";
            }
            throw unsupported(accumulatorType, "reduction accumulator");
        }

        private ReductionInfo collectReductionInfo(ReductionNode reduction) {
            java.util.List<Integer> axes = new java.util.ArrayList<>();
            ExprNode current = reduction;
            ReductionOp op = reduction.op();
            boolean keepDims = reduction.keepDims();
            while (current instanceof ReductionNode node
                    && node.op() == op
                    && node.keepDims() == keepDims) {
                axes.add(node.axis());
                current = node.input();
            }
            int[] axisArray = axes.stream().distinct().mapToInt(Integer::intValue).toArray();
            java.util.Arrays.sort(axisArray);
            return new ReductionInfo(current, axisArray, keepDims, op, reduction.dataType());
        }

        private record ReductionInfo(
                ExprNode input, int[] axes, boolean keepDims, ReductionOp op, DataType dataType) {}

        private void appendHelpers(StringBuilder source) {
            if (helpers.contains(HelperMethod.READ_BYTE)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static byte readByte(MemoryView<MemorySegment> view, long index) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append("    return base.get(ValueLayout.JAVA_BYTE, offset);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static byte readByte(MemorySegment base, long offset) {\n");
                    source.append("    return base.get(ValueLayout.JAVA_BYTE, offset);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.READ_SHORT)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static short readShort(MemoryView<MemorySegment> view, long index) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append(
                            "    return base.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static short readShort(MemorySegment base, long offset) {\n");
                    source.append(
                            "    return base.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.READ_FLOAT)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static float readFloat(MemoryView<MemorySegment> view, long index) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append(
                            "    return base.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static float readFloat(MemorySegment base, long offset) {\n");
                    source.append(
                            "    return base.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.READ_INT)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static int readInt(MemoryView<MemorySegment> view, long index) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append("    return base.get(ValueLayout.JAVA_INT_UNALIGNED, offset);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static int readInt(MemorySegment base, long offset) {\n");
                    source.append("    return base.get(ValueLayout.JAVA_INT_UNALIGNED, offset);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.READ_LONG)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static long readLong(MemoryView<MemorySegment> view, long index) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append(
                            "    return base.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static long readLong(MemorySegment base, long offset) {\n");
                    source.append(
                            "    return base.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.READ_DOUBLE)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static double readDouble(MemoryView<MemorySegment> view, long index) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append(
                            "    return base.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static double readDouble(MemorySegment base, long offset) {\n");
                    source.append(
                            "    return base.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.WRITE_BYTE)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static void writeByte(MemoryView<MemorySegment> view, long index, byte value) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append("    base.set(ValueLayout.JAVA_BYTE, offset, value);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static void writeByte(MemorySegment base, long offset, byte value) {\n");
                    source.append("    base.set(ValueLayout.JAVA_BYTE, offset, value);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.WRITE_SHORT)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static void writeShort(MemoryView<MemorySegment> view, long index, short value) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append(
                            "    base.set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static void writeShort(MemorySegment base, long offset, short value) {\n");
                    source.append(
                            "    base.set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.WRITE_FLOAT)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static void writeFloat(MemoryView<MemorySegment> view, long index, float value) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append(
                            "    base.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static void writeFloat(MemorySegment base, long offset, float value) {\n");
                    source.append(
                            "    base.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.WRITE_INT)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static void writeInt(MemoryView<MemorySegment> view, long index, int value) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append("    base.set(ValueLayout.JAVA_INT_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static void writeInt(MemorySegment base, long offset, int value) {\n");
                    source.append("    base.set(ValueLayout.JAVA_INT_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.WRITE_LONG)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static void writeLong(MemoryView<MemorySegment> view, long index, long value) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append(
                            "    base.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static void writeLong(MemorySegment base, long offset, long value) {\n");
                    source.append(
                            "    base.set(ValueLayout.JAVA_LONG_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.WRITE_DOUBLE)) {
                source.append("\n");
                if (style == KernelStyle.CONTIGUOUS) {
                    source.append(
                            "  private static void writeDouble(MemoryView<MemorySegment> view, long index, double value) {\n");
                    source.append(
                            "    long offset = view.byteOffset() + index * view.dataType().byteSize();\n");
                    source.append("    MemorySegment base = view.memory().base();\n");
                    source.append(
                            "    base.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                } else {
                    source.append(
                            "  private static void writeDouble(MemorySegment base, long offset, double value) {\n");
                    source.append(
                            "    base.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset, value);\n");
                    source.append("  }\n");
                }
            }
            if (helpers.contains(HelperMethod.SIGMOID)) {
                source.append("\n");
                source.append("  private static float sigmoid(float value) {\n");
                source.append("    return 1.0f / (1.0f + (float) Math.exp(-value));\n");
                source.append("  }\n");
            }
            if (helpers.contains(HelperMethod.SILU)) {
                source.append("\n");
                source.append("  private static float silu(float value) {\n");
                source.append("    return value / (1.0f + (float) Math.exp(-value));\n");
                source.append("  }\n");
            }
            if (helpers.contains(HelperMethod.GELU)) {
                source.append("\n");
                source.append("  private static float gelu(float value) {\n");
                source.append("    float cubic = value * value * value;\n");
                source.append("    float inner = 0.79788456f * (value + 0.044715f * cubic);\n");
                source.append("    return 0.5f * value * (1.0f + (float) Math.tanh(inner));\n");
                source.append("  }\n");
            }
            if (helpers.contains(HelperMethod.OFFSET_FOR_INDEX)) {
                source.append("\n");
                source.append("  private static long offsetForIndex(\n");
                source.append(
                        "          long index, long baseOffset, long[] shape, long[] stride) {\n");
                source.append("    long offset = baseOffset;\n");
                source.append("    long remaining = index;\n");
                source.append("    for (int dim = shape.length - 1; dim >= 0; dim--) {\n");
                source.append("      long size = shape[dim];\n");
                source.append("      if (size == 0) {\n");
                source.append("        return baseOffset;\n");
                source.append("      }\n");
                source.append("      long coord = remaining % size;\n");
                source.append("      remaining /= size;\n");
                source.append("      offset += coord * stride[dim];\n");
                source.append("    }\n");
                source.append("    return offset;\n");
                source.append("  }\n");
            }
        }

        private String emitNode(ExprNode node) {
            String existing = names.get(node);
            if (existing != null) {
                return existing;
            }
            if (node instanceof InputNode input) {
                String var = nextVar();
                names.put(node, var);
                String read = readInput(input, "i");
                lines.add(typeFor(node.dataType()) + " " + var + " = " + read + ";");
                return var;
            }
            if (node instanceof ScalarNode scalar) {
                String var = nextVar();
                names.put(node, var);
                lines.add(typeFor(node.dataType()) + " " + var + " = " + literal(scalar) + ";");
                return var;
            }
            if (node instanceof UnaryNode unary) {
                String inputVar = emitNode(unary.input());
                String var = nextVar();
                names.put(node, var);
                lines.add(
                        typeFor(node.dataType())
                                + " "
                                + var
                                + " = "
                                + unaryExpression(unary.op(), inputVar, node.dataType())
                                + ";");
                return var;
            }
            if (node instanceof BinaryNode binary) {
                String leftVar = emitNode(binary.left());
                String rightVar = emitNode(binary.right());
                String var = nextVar();
                names.put(node, var);
                lines.add(
                        typeFor(node.dataType())
                                + " "
                                + var
                                + " = "
                                + binaryExpression(
                                        binary.op(),
                                        leftVar,
                                        binary.left().dataType(),
                                        rightVar,
                                        binary.right().dataType(),
                                        node.dataType())
                                + ";");
                return var;
            }
            if (node instanceof TernaryNode ternary) {
                String condVar = emitNode(ternary.condition());
                String trueVar = emitNode(ternary.trueValue());
                String falseVar = emitNode(ternary.falseValue());
                String var = nextVar();
                names.put(node, var);
                String guard = booleanExpression(condVar, ternary.condition().dataType());
                lines.add(
                        typeFor(node.dataType())
                                + " "
                                + var
                                + " = ("
                                + guard
                                + " ? "
                                + trueVar
                                + " : "
                                + falseVar
                                + ");");
                return var;
            }
            if (node instanceof CastNode cast) {
                String inputVar = emitNode(cast.input());
                String var = nextVar();
                names.put(node, var);
                lines.add(
                        typeFor(cast.targetType())
                                + " "
                                + var
                                + " = "
                                + castExpression(
                                        inputVar, cast.input().dataType(), cast.targetType())
                                + ";");
                return var;
            }
            throw new IllegalStateException("Unsupported node: " + node);
        }

        private String expressionForNode(ExprNode node, String indexVar) {
            return expressionForNode(node, indexVar, Map.of());
        }

        private String expressionForNode(
                ExprNode node, String indexVar, Map<ExprNode, String> overrides) {
            String override = overrides.get(node);
            if (override != null) {
                return override;
            }
            if (node instanceof InputNode input) {
                return readInput(input, indexVar);
            }
            if (node instanceof ScalarNode scalar) {
                return literal(scalar);
            }
            if (node instanceof UnaryNode unary) {
                String inputExpr = expressionForNode(unary.input(), indexVar, overrides);
                return unaryExpression(unary.op(), inputExpr, node.dataType());
            }
            if (node instanceof BinaryNode binary) {
                String leftExpr = expressionForNode(binary.left(), indexVar, overrides);
                String rightExpr = expressionForNode(binary.right(), indexVar, overrides);
                return binaryExpression(
                        binary.op(),
                        leftExpr,
                        binary.left().dataType(),
                        rightExpr,
                        binary.right().dataType(),
                        node.dataType());
            }
            if (node instanceof TernaryNode ternary) {
                String conditionExpr = expressionForNode(ternary.condition(), indexVar, overrides);
                String trueExpr = expressionForNode(ternary.trueValue(), indexVar, overrides);
                String falseExpr = expressionForNode(ternary.falseValue(), indexVar, overrides);
                String guard = booleanExpression(conditionExpr, ternary.condition().dataType());
                return "(" + guard + " ? " + trueExpr + " : " + falseExpr + ")";
            }
            if (node instanceof CastNode cast) {
                String inputExpr = expressionForNode(cast.input(), indexVar, overrides);
                return castExpression(inputExpr, cast.input().dataType(), cast.targetType());
            }
            if (node instanceof ReductionNode) {
                throw new IllegalStateException(
                        "Reductions are not supported in elementwise expressions");
            }
            throw new IllegalStateException("Unsupported node in reduction: " + node);
        }

        private String readInput(InputNode input) {
            return readInput(input, "i");
        }

        private String readInput(InputNode input, String indexVar) {
            if (input.dataType() == DataType.BOOL || input.dataType() == DataType.I8) {
                helpers.add(HelperMethod.READ_BYTE);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "readByte(inputs[" + input.index() + "], " + indexVar + ")";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "readByte(in"
                        + input.index()
                        + ", offsetForIndex("
                        + indexVar
                        + ", in"
                        + input.index()
                        + "BaseOffset, shape, in"
                        + input.index()
                        + "Stride))";
            }
            if (input.dataType() == DataType.I16
                    || input.dataType() == DataType.FP16
                    || input.dataType() == DataType.BF16) {
                helpers.add(HelperMethod.READ_SHORT);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "readShort(inputs[" + input.index() + "], " + indexVar + ")";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "readShort(in"
                        + input.index()
                        + ", offsetForIndex("
                        + indexVar
                        + ", in"
                        + input.index()
                        + "BaseOffset, shape, in"
                        + input.index()
                        + "Stride))";
            }
            if (input.dataType() == DataType.I32) {
                helpers.add(HelperMethod.READ_INT);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "readInt(inputs[" + input.index() + "], " + indexVar + ")";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "readInt(in"
                        + input.index()
                        + ", offsetForIndex("
                        + indexVar
                        + ", in"
                        + input.index()
                        + "BaseOffset, shape, in"
                        + input.index()
                        + "Stride))";
            }
            if (input.dataType() == DataType.I64) {
                helpers.add(HelperMethod.READ_LONG);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "readLong(inputs[" + input.index() + "], " + indexVar + ")";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "readLong(in"
                        + input.index()
                        + ", offsetForIndex("
                        + indexVar
                        + ", in"
                        + input.index()
                        + "BaseOffset, shape, in"
                        + input.index()
                        + "Stride))";
            }
            if (input.dataType() == DataType.FP32) {
                helpers.add(HelperMethod.READ_FLOAT);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "readFloat(inputs[" + input.index() + "], " + indexVar + ")";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "readFloat(in"
                        + input.index()
                        + ", offsetForIndex("
                        + indexVar
                        + ", in"
                        + input.index()
                        + "BaseOffset, shape, in"
                        + input.index()
                        + "Stride))";
            }
            if (input.dataType() == DataType.FP64) {
                helpers.add(HelperMethod.READ_DOUBLE);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "readDouble(inputs[" + input.index() + "], " + indexVar + ")";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "readDouble(in"
                        + input.index()
                        + ", offsetForIndex("
                        + indexVar
                        + ", in"
                        + input.index()
                        + "BaseOffset, shape, in"
                        + input.index()
                        + "Stride))";
            }
            throw unsupported(input.dataType(), "input");
        }

        private String reductionResultExpression(DataType dataType, String accVar) {
            if (dataType == DataType.FP16) {
                return "Float.floatToFloat16(" + accVar + ")";
            }
            if (dataType == DataType.BF16) {
                return "BFloat16.fromFloat(" + accVar + ")";
            }
            return accVar;
        }

        private String writeReductionOutput(DataType dataType, String valueVar) {
            if (dataType == DataType.BOOL || dataType == DataType.I8) {
                return "outBase.set(ValueLayout.JAVA_BYTE, outOffset, " + valueVar + ");";
            }
            if (dataType == DataType.I16
                    || dataType == DataType.FP16
                    || dataType == DataType.BF16) {
                return "outBase.set(ValueLayout.JAVA_SHORT_UNALIGNED, outOffset, "
                        + valueVar
                        + ");";
            }
            if (dataType == DataType.I32) {
                return "outBase.set(ValueLayout.JAVA_INT_UNALIGNED, outOffset, " + valueVar + ");";
            }
            if (dataType == DataType.I64) {
                return "outBase.set(ValueLayout.JAVA_LONG_UNALIGNED, outOffset, " + valueVar + ");";
            }
            if (dataType == DataType.FP32) {
                return "outBase.set(ValueLayout.JAVA_FLOAT_UNALIGNED, outOffset, " + valueVar + ");";
            }
            if (dataType == DataType.FP64) {
                return "outBase.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, outOffset, " + valueVar + ");";
            }
            throw unsupported(dataType, "output");
        }

        private String writeOutput(ExprNode node, String valueVar) {
            if (node.dataType() == DataType.BOOL || node.dataType() == DataType.I8) {
                helpers.add(HelperMethod.WRITE_BYTE);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "writeByte(output, i, " + valueVar + ");";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "writeByte(outBase, offsetForIndex(i, outBaseOffset, shape, outStride), "
                        + valueVar
                        + ");";
            }
            if (node.dataType() == DataType.I16
                    || node.dataType() == DataType.FP16
                    || node.dataType() == DataType.BF16) {
                helpers.add(HelperMethod.WRITE_SHORT);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "writeShort(output, i, " + valueVar + ");";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "writeShort(outBase, offsetForIndex(i, outBaseOffset, shape, outStride), "
                        + valueVar
                        + ");";
            }
            if (node.dataType() == DataType.I32) {
                helpers.add(HelperMethod.WRITE_INT);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "writeInt(output, i, " + valueVar + ");";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "writeInt(outBase, offsetForIndex(i, outBaseOffset, shape, outStride), "
                        + valueVar
                        + ");";
            }
            if (node.dataType() == DataType.I64) {
                helpers.add(HelperMethod.WRITE_LONG);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "writeLong(output, i, " + valueVar + ");";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "writeLong(outBase, offsetForIndex(i, outBaseOffset, shape, outStride), "
                        + valueVar
                        + ");";
            }
            if (node.dataType() == DataType.FP32) {
                helpers.add(HelperMethod.WRITE_FLOAT);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "writeFloat(output, i, " + valueVar + ");";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "writeFloat(outBase, offsetForIndex(i, outBaseOffset, shape, outStride), "
                        + valueVar
                        + ");";
            }
            if (node.dataType() == DataType.FP64) {
                helpers.add(HelperMethod.WRITE_DOUBLE);
                if (style == KernelStyle.CONTIGUOUS) {
                    return "writeDouble(output, i, " + valueVar + ");";
                }
                helpers.add(HelperMethod.OFFSET_FOR_INDEX);
                return "writeDouble(outBase, offsetForIndex(i, outBaseOffset, shape, outStride), "
                        + valueVar
                        + ");";
            }
            throw unsupported(node.dataType(), "output");
        }

        private String literal(ScalarNode scalar) {
            if (scalar.dataType() == DataType.FP32) {
                float value = scalar.value().floatValue();
                return Float.toString(value) + "f";
            }
            if (scalar.dataType() == DataType.I32) {
                return Integer.toString(scalar.value().intValue());
            }
            if (scalar.dataType() == DataType.BOOL) {
                return scalar.value().intValue() == 0 ? "(byte) 0" : "(byte) 1";
            }
            throw unsupported(scalar.dataType(), "scalar");
        }

        private String unaryExpression(UnaryOp op, String inputVar, DataType dataType) {
            if (dataType == DataType.BOOL) {
                return switch (op.name()) {
                    case "logicalNot" -> "(byte) (" + inputVar + " == 0 ? 1 : 0)";
                    default ->
                            throw new IllegalStateException("Unsupported unary op: " + op.name());
                };
            }
            if (dataType == DataType.FP32) {
                return switch (op.name()) {
                    case "negate" -> "-" + inputVar;
                    case "abs" -> "Math.abs(" + inputVar + ")";
                    case "exp" -> "(float) Math.exp(" + inputVar + ")";
                    case "log" -> "(float) Math.log(" + inputVar + ")";
                    case "sqrt" -> "(float) Math.sqrt(" + inputVar + ")";
                    case "square" -> "(" + inputVar + " * " + inputVar + ")";
                    case "sin" -> "(float) Math.sin(" + inputVar + ")";
                    case "cos" -> "(float) Math.cos(" + inputVar + ")";
                    case "tanh" -> "(float) Math.tanh(" + inputVar + ")";
                    case "sigmoid" -> withHelper(HelperMethod.SIGMOID, "sigmoid(" + inputVar + ")");
                    case "relu" -> "Math.max(0.0f, " + inputVar + ")";
                    case "gelu" -> withHelper(HelperMethod.GELU, "gelu(" + inputVar + ")");
                    case "silu" -> withHelper(HelperMethod.SILU, "silu(" + inputVar + ")");
                    default ->
                            throw new IllegalStateException("Unsupported unary op: " + op.name());
                };
            }
            if (dataType == DataType.I32) {
                return switch (op.name()) {
                    case "negate" -> "-" + inputVar;
                    case "abs" -> "Math.abs(" + inputVar + ")";
                    case "square" -> "(" + inputVar + " * " + inputVar + ")";
                    case "relu" -> "Math.max(0, " + inputVar + ")";
                    default ->
                            throw new IllegalStateException(
                                    "Unsupported unary op for I32: " + op.name());
                };
            }
            throw unsupported(dataType, "unary");
        }

        private String binaryExpression(
                BinaryOp op,
                String leftVar,
                DataType leftType,
                String rightVar,
                DataType rightType,
                DataType dataType) {
            if (dataType == DataType.BOOL) {
                return switch (op.name()) {
                    case "logicalAnd" ->
                            "(byte) (("
                                    + booleanExpression(leftVar, leftType)
                                    + " && "
                                    + booleanExpression(rightVar, rightType)
                                    + ") ? 1 : 0)";
                    case "logicalOr" ->
                            "(byte) (("
                                    + booleanExpression(leftVar, leftType)
                                    + " || "
                                    + booleanExpression(rightVar, rightType)
                                    + ") ? 1 : 0)";
                    case "logicalXor" ->
                            "(byte) (("
                                    + booleanExpression(leftVar, leftType)
                                    + " ^ "
                                    + booleanExpression(rightVar, rightType)
                                    + ") ? 1 : 0)";
                    case "equal" ->
                            "(byte) (("
                                    + "(" + numericExpression(leftVar, leftType) + ")"
                                    + " == "
                                    + "(" + numericExpression(rightVar, rightType) + ")"
                                    + ") ? 1 : 0)";
                    case "lessThan" ->
                            "(byte) (("
                                    + "(" + numericExpression(leftVar, leftType) + ")"
                                    + " < "
                                    + "(" + numericExpression(rightVar, rightType) + ")"
                                    + ") ? 1 : 0)";
                    default ->
                            throw new IllegalStateException("Unsupported binary op: " + op.name());
                };
            }
            if (dataType == DataType.FP32) {
                return switch (op.name()) {
                    case "add" -> leftVar + " + " + rightVar;
                    case "subtract" -> leftVar + " - " + rightVar;
                    case "multiply" -> leftVar + " * " + rightVar;
                    case "divide" -> leftVar + " / " + rightVar;
                    case "min" -> "Math.min(" + leftVar + ", " + rightVar + ")";
                    case "max" -> "Math.max(" + leftVar + ", " + rightVar + ")";
                    case "pow" -> "(float) Math.pow(" + leftVar + ", " + rightVar + ")";
                    default ->
                            throw new IllegalStateException("Unsupported binary op: " + op.name());
                };
            }
            if (dataType == DataType.I32) {
                return switch (op.name()) {
                    case "add" -> leftVar + " + " + rightVar;
                    case "subtract" -> leftVar + " - " + rightVar;
                    case "multiply" -> leftVar + " * " + rightVar;
                    case "divide" -> leftVar + " / " + rightVar;
                    case "min" -> "Math.min(" + leftVar + ", " + rightVar + ")";
                    case "max" -> "Math.max(" + leftVar + ", " + rightVar + ")";
                    case "pow" -> "(int) Math.pow(" + leftVar + ", " + rightVar + ")";
                    default ->
                            throw new IllegalStateException("Unsupported binary op: " + op.name());
                };
            }
            throw unsupported(dataType, "binary");
        }

        private String castExpression(String inputVar, DataType from, DataType to) {
            if (from == to) {
                return inputVar;
            }
            String booleanExpr = booleanExpression(inputVar, from);
            if (to == DataType.BOOL) {
                return "(byte) (" + booleanExpr + " ? 1 : 0)";
            }
            if (to == DataType.I8) {
                return "(byte) " + numericExpression(inputVar, from);
            }
            if (to == DataType.I16) {
                return "(short) " + numericExpression(inputVar, from);
            }
            if (to == DataType.I32) {
                return "(int) " + numericExpression(inputVar, from);
            }
            if (to == DataType.I64) {
                return "(long) " + numericExpression(inputVar, from);
            }
            if (to == DataType.FP16) {
                return "Float.floatToFloat16((float) " + numericExpression(inputVar, from) + ")";
            }
            if (to == DataType.BF16) {
                return "BFloat16.fromFloat((float) " + numericExpression(inputVar, from) + ")";
            }
            if (to == DataType.FP32) {
                return "(float) " + numericExpression(inputVar, from);
            }
            if (to == DataType.FP64) {
                return "(double) " + numericExpression(inputVar, from);
            }
            throw new IllegalStateException("Unsupported cast from " + from + " to " + to);
        }

        private String booleanExpression(String inputVar, DataType from) {
            if (from == DataType.BOOL) {
                return inputVar + " != 0";
            }
            if (from == DataType.FP16) {
                return "Float.float16ToFloat(" + inputVar + ") != 0.0f";
            }
            if (from == DataType.BF16) {
                return "BFloat16.toFloat(" + inputVar + ") != 0.0f";
            }
            if (from == DataType.FP32) {
                return inputVar + " != 0.0f";
            }
            if (from == DataType.FP64) {
                return inputVar + " != 0.0";
            }
            return inputVar + " != 0";
        }

        private String numericExpression(String inputVar, DataType from) {
            if (from == DataType.FP16) {
                return "Float.float16ToFloat(" + inputVar + ")";
            }
            if (from == DataType.BF16) {
                return "BFloat16.toFloat(" + inputVar + ")";
            }
            if (from == DataType.BOOL) {
                return inputVar + " == 0 ? 0 : 1";
            }
            return inputVar;
        }

        private String castToAccumulator(String inputVar, DataType from, DataType to) {
            if (from == to) {
                return inputVar;
            }
            if (to == DataType.I32) {
                return "(int) " + inputVar;
            }
            if (to == DataType.I64) {
                return "(long) " + inputVar;
            }
            if (to == DataType.FP32) {
                return "(float) " + inputVar;
            }
            if (to == DataType.FP64) {
                return "(double) " + inputVar;
            }
            throw new IllegalStateException("Unsupported accumulator type: " + to);
        }

        private String typeFor(DataType dataType) {
            if (dataType == DataType.BOOL || dataType == DataType.I8) {
                return "byte";
            }
            if (dataType == DataType.I16
                    || dataType == DataType.FP16
                    || dataType == DataType.BF16) {
                return "short";
            }
            if (dataType == DataType.I32) {
                return "int";
            }
            if (dataType == DataType.I64) {
                return "long";
            }
            if (dataType == DataType.FP32) {
                return "float";
            }
            if (dataType == DataType.FP64) {
                return "double";
            }
            throw unsupported(dataType, "type");
        }

        private String nextVar() {
            return "v" + counter++;
        }

        private String withHelper(HelperMethod helper, String expression) {
            helpers.add(helper);
            return expression;
        }

        private IllegalStateException unsupported(DataType dataType, String context) {
            return new IllegalStateException(
                    "Unsupported data type for " + context + ": " + dataType);
        }

        private enum HelperMethod {
            READ_BYTE,
            READ_SHORT,
            READ_FLOAT,
            READ_INT,
            READ_LONG,
            READ_DOUBLE,
            WRITE_BYTE,
            WRITE_SHORT,
            WRITE_FLOAT,
            WRITE_INT,
            WRITE_LONG,
            WRITE_DOUBLE,
            SIGMOID,
            SILU,
            GELU,
            OFFSET_FOR_INDEX
        }
    }
}
