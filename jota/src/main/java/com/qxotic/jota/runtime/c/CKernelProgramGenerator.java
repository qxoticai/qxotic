package com.qxotic.jota.runtime.c;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.*;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.tensor.KernelCacheKey;
import com.qxotic.jota.tensor.KernelProgram;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

final class CKernelProgramGenerator {

    KernelProgram generate(LIRGraph graph, ScratchLayout scratchLayout, KernelCacheKey key) {
        String kernelName = "c_lir_" + key.value().substring(0, 12);
        String source = LirKernelSourceGenerator.generate(graph, scratchLayout, kernelName);
        return new KernelProgram(
                KernelProgram.Kind.SOURCE, KernelProgram.C, source, kernelName, java.util.Map.of());
    }

    private static final class LirKernelSourceGenerator {
        private final LIRGraph graph;
        private final ScratchLayout scratchLayout;
        private final String kernelName;
        private final LIRExprGraph exprGraph;
        private final Map<ScalarInput, String> scalarInputs = new IdentityHashMap<>();
        private final Map<Integer, String> scalarInputNames = new java.util.HashMap<>();
        private final Map<LIRExprNode, String> tempNames = new IdentityHashMap<>();
        private final Map<BufferRef, BufferVar> buffers = new IdentityHashMap<>();
        private final Map<BufferRef, ScratchBufferVar> scratchBuffers = new IdentityHashMap<>();
        private final List<String> lines = new ArrayList<>();
        private int tempId;
        private int indentLevel;

        private LirKernelSourceGenerator(
                LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
            this.graph = graph;
            this.scratchLayout = scratchLayout;
            this.kernelName = kernelName;
            this.exprGraph = graph.exprGraph();
        }

        static String generate(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
            LirKernelSourceGenerator generator =
                    new LirKernelSourceGenerator(graph, scratchLayout, kernelName);
            return generator.generate();
        }

        private String generate() {
            StringBuilder source = new StringBuilder();
            source.append("#include <stdint.h>\n");
            source.append("#include <stddef.h>\n");
            source.append("#include <string.h>\n");
            source.append("#include <math.h>\n\n");
            source.append("void ")
                    .append(kernelName)
                    .append("(void **buffers, uint64_t *scalars, uint64_t scratch_ptr) {\n");
            indentLevel = 1;
            emitProlog();
            emitNode(graph.body());
            for (String line : lines) {
                source.append(line);
            }
            source.append("}\n");
            return source.toString();
        }

        private void emitProlog() {
            int bufferIndex = 0;
            int scalarIndex = 0;
            for (LIRInput input : graph.inputs()) {
                if (input instanceof BufferRef buffer) {
                    String name = "input" + bufferIndex;
                    addLine("uint8_t *" + name + " = (uint8_t *)buffers[" + bufferIndex + "];");
                    buffers.put(
                            buffer, new BufferVar(name, buffer.dataType(), buffer.byteStrides()));
                    bufferIndex++;
                } else if (input instanceof ScalarInput scalar) {
                    String name = "scalar" + scalarIndex;
                    scalarInputs.put(scalar, name);
                    scalarInputNames.put(scalar.id(), name);
                    String bitsExpr = "scalars[" + scalarIndex + "]";
                    if (scalar.dataType() == DataType.FP16) {
                        addLine("uint16_t " + name + "_bits = (uint16_t)" + bitsExpr + ";");
                        addLine(
                                "_Float16 "
                                        + name
                                        + "; memcpy(&"
                                        + name
                                        + ", &"
                                        + name
                                        + "_bits, sizeof(uint16_t));");
                    } else if (scalar.dataType() == DataType.BF16) {
                        addLine("uint16_t " + name + "_bits = (uint16_t)" + bitsExpr + ";");
                        addLine(
                                "__bf16 "
                                        + name
                                        + "; memcpy(&"
                                        + name
                                        + ", &"
                                        + name
                                        + "_bits, sizeof(uint16_t));");
                    } else if (scalar.dataType() == DataType.FP32) {
                        addLine("uint32_t " + name + "_bits = (uint32_t)" + bitsExpr + ";");
                        addLine(
                                "float "
                                        + name
                                        + "; memcpy(&"
                                        + name
                                        + ", &"
                                        + name
                                        + "_bits, sizeof(float));");
                    } else if (scalar.dataType() == DataType.FP64) {
                        addLine("uint64_t " + name + "_bits = (uint64_t)" + bitsExpr + ";");
                        addLine(
                                "double "
                                        + name
                                        + "; memcpy(&"
                                        + name
                                        + ", &"
                                        + name
                                        + "_bits, sizeof(double));");
                    } else {
                        addLine(
                                typeName(scalar.dataType())
                                        + " "
                                        + name
                                        + " = ("
                                        + typeName(scalar.dataType())
                                        + ")"
                                        + bitsExpr
                                        + ";");
                    }
                    scalarIndex++;
                }
            }

            for (int i = 0; i < graph.outputs().size(); i++) {
                BufferRef buffer = graph.outputs().get(i);
                String name = "output" + i;
                addLine("uint8_t *" + name + " = (uint8_t *)buffers[" + (bufferIndex + i) + "];");
                buffers.put(buffer, new BufferVar(name, buffer.dataType(), buffer.byteStrides()));
            }

            addLine("uint8_t *scratch = (uint8_t *)(uintptr_t)scratch_ptr;");
            if (scratchLayout.requiresScratch()) {
                int slotId = 0;
                for (var entry : scratchLayout.offsets().entrySet()) {
                    BufferRef buf = entry.getKey();
                    long offset = entry.getValue();
                    String name = "scratch" + slotId++;
                    scratchBuffers.put(buf, new ScratchBufferVar(name, buf.dataType(), offset));
                    addLine("uint8_t *" + name + " = scratch + " + offset + "LL;");
                }
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

            boolean isSimpleForwardLoop =
                    step.equals("1") || (loop.step() instanceof IConst ic && ic.value() == 1);

            if (isSimpleForwardLoop) {
                for (LoopIterArg arg : loop.iterArgs()) {
                    String initExpr = emitScalarExpr(arg.init());
                    addLine(typeName(arg.dataType()) + " " + arg.name() + " = " + initExpr + ";");
                }
                addLine(
                        "for (long long "
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
                String lbVar = idx + "_lb";
                String ubVar = idx + "_ub";
                String stepVar = idx + "_step";
                addLine("long long " + lbVar + " = " + lb + ";");
                addLine("long long " + ubVar + " = " + ub + ";");
                addLine("long long " + stepVar + " = " + step + ";");

                for (LoopIterArg arg : loop.iterArgs()) {
                    String initExpr = emitScalarExpr(arg.init());
                    addLine(typeName(arg.dataType()) + " " + arg.name() + " = " + initExpr + ";");
                }
                addLine(
                        "for (long long "
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
                addLine(typeName(arg.dataType()) + " " + nextName + " = " + expr + ";");
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
            String offset = emitIndexExpr(store.offset());
            addLine(writeValue(buffer, bufferType, offset, value, valueType));
        }

        private String emitScalarExpr(LIRExprNode node) {
            LIRExprNode resolved = exprGraph.resolve(node);
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
                addLine(typeName(resolved.dataType()) + " " + var + " = " + expr + ";");
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
                case S_REF -> ((SRef) resolved).name();
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
            String offset = emitIndexExpr(load.offset());
            return readValue(buffer, load.buffer().dataType(), offset);
        }

        private String emitUnaryExpr(SUnary unary) {
            String input = emitScalarExpr(unary.input());
            DataType type = unary.dataType();
            return switch (unary.op()) {
                case NEGATE -> negateExpr(type, input);
                case ABS -> absExpr(type, input);
                case EXP -> floatUnaryExpr(type, "exp", "expf", input);
                case LOG -> floatUnaryExpr(type, "log", "logf", input);
                case SQRT -> floatUnaryExpr(type, "sqrt", "sqrtf", input);
                case SQUARE -> squareExpr(type, input);
                case SIN -> floatUnaryExpr(type, "sin", "sinf", input);
                case COS -> floatUnaryExpr(type, "cos", "cosf", input);
                case TAN -> floatUnaryExpr(type, "tan", "tanf", input);
                case TANH -> floatUnaryExpr(type, "tanh", "tanhf", input);
                case RECIPROCAL -> reciprocalExpr(type, input);
                case LOGICAL_NOT ->
                        type == DataType.BOOL
                                ? "(" + input + " ? 0 : 1)"
                                : "(" + input + " == 0 ? 1 : 0)";
                case BITWISE_NOT -> "(~(" + input + "))";
            };
        }

        private String emitBinaryExpr(SBinary binary) {
            String left = emitScalarExpr(binary.left());
            String right = emitScalarExpr(binary.right());
            DataType type = binary.dataType();
            return switch (binary.op()) {
                case ADD -> binaryArithmetic(type, "+", left, right);
                case SUBTRACT -> binaryArithmetic(type, "-", left, right);
                case MULTIPLY -> binaryArithmetic(type, "*", left, right);
                case DIVIDE -> binaryArithmetic(type, "/", left, right);
                case MIN -> minExpr(type, left, right);
                case MAX -> maxExpr(type, left, right);
                case POW -> powExpr(type, left, right);
                case LOGICAL_AND -> logicalExpr("&&", binary, left, right);
                case LOGICAL_OR -> logicalExpr("||", binary, left, right);
                case LOGICAL_XOR -> logicalXorExpr(binary, left, right);
                case BITWISE_AND -> "(" + left + " & " + right + ")";
                case BITWISE_OR -> "(" + left + " | " + right + ")";
                case BITWISE_XOR -> "(" + left + " ^ " + right + ")";
                case EQUAL -> compareExpr("==", binary.left(), binary.right(), type);
                case LESS_THAN -> compareExpr("<", binary.left(), binary.right(), type);
            };
        }

        private String logicalExpr(String op, SBinary binary, String left, String right) {
            String l = binary.left().dataType() != DataType.BOOL ? "(" + left + " != 0)" : left;
            String r = binary.right().dataType() != DataType.BOOL ? "(" + right + " != 0)" : right;
            String result = "(" + l + " " + op + " " + r + ")";
            return binary.dataType() == DataType.BOOL ? result : "(" + result + " ? 1 : 0)";
        }

        private String logicalXorExpr(SBinary binary, String left, String right) {
            String l = binary.left().dataType() != DataType.BOOL ? "(" + left + " != 0)" : left;
            String r = binary.right().dataType() != DataType.BOOL ? "(" + right + " != 0)" : right;
            String result = "(" + l + " ^ " + r + ")";
            return binary.dataType() == DataType.BOOL ? result : "(" + result + " ? 1 : 0)";
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
                case I_CONST -> ((IConst) resolved).value() + "LL";
                case I_VAR -> ((IVar) resolved).name();
                case I_BINARY ->
                        "("
                                + emitIndexExpr(((IBinary) resolved).left())
                                + " "
                                + indexOp(((IBinary) resolved).op())
                                + " "
                                + emitIndexExpr(((IBinary) resolved).right())
                                + ")";
                default ->
                        throw new IllegalStateException(
                                "Expected index node, got " + resolved.kind());
            };
        }

        private String compareExpr(
                String op, LIRExprNode leftExpr, LIRExprNode rightExpr, DataType resultType) {
            DataType leftType = leftExpr.dataType();
            DataType rightType = rightExpr.dataType();
            String left = emitScalarExpr(leftExpr);
            String right = emitScalarExpr(rightExpr);

            if (leftType == DataType.FP16 || leftType == DataType.BF16) {
                left = toFloatExpr(leftType, left);
                leftType = DataType.FP32;
            }
            if (rightType == DataType.FP16 || rightType == DataType.BF16) {
                right = toFloatExpr(rightType, right);
                rightType = DataType.FP32;
            }

            if (op.equals("<") && (leftType == DataType.BOOL || rightType == DataType.BOOL)) {
                left = castExpr(leftType, DataType.I32, left);
                right = castExpr(rightType, DataType.I32, right);
                leftType = DataType.I32;
                rightType = DataType.I32;
            }

            if (leftType == DataType.BOOL && rightType != DataType.BOOL) {
                left = castExpr(leftType, rightType, left);
            } else if (rightType == DataType.BOOL && leftType != DataType.BOOL) {
                right = castExpr(rightType, leftType, right);
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

        private BufferVar requireBuffer(BufferRef ref) {
            BufferVar buffer = buffers.get(ref);
            if (buffer != null) {
                return buffer;
            }
            ScratchBufferVar scratch = scratchBuffers.get(ref);
            if (scratch != null) {
                return scratch;
            }
            throw new IllegalStateException("Missing buffer mapping for " + ref);
        }

        private String requireScalarInputById(int id, DataType type) {
            String name = scalarInputNames.get(id);
            if (name == null) {
                throw new IllegalStateException("Missing scalar input " + id);
            }
            return name;
        }

        private String nextTempName() {
            return "t" + tempId++;
        }

        private String readValue(BufferVar buffer, DataType type, String offset) {
            String ptr = "(" + buffer.name + " + " + offset + ")";
            return "*(" + typeName(type) + " *)" + ptr;
        }

        private String writeValue(
                BufferVar buffer, DataType type, String offset, String value, DataType valueType) {
            String ptr = "(" + buffer.name + " + " + offset + ")";
            String castValue = castExpr(valueType, type, value);
            if (type == DataType.BOOL) {
                if (valueType == DataType.BOOL) {
                    castValue = "(" + value + " ? 1 : 0)";
                } else {
                    castValue = "(" + value + " != 0 ? 1 : 0)";
                }
            }
            return "*(" + typeName(type) + " *)" + ptr + " = " + castValue + ";";
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
                float value = Float.float16ToFloat((short) bits);
                return "((_Float16)" + formatFloatLiteral(value) + ")";
            }
            if (type == DataType.BF16) {
                int bf16Bits = ((int) bits) & 0xFFFF;
                float value = Float.intBitsToFloat(bf16Bits << 16);
                return "((__bf16)" + formatFloatLiteral(value) + ")";
            }
            if (type == DataType.I8) {
                return "(int8_t)" + bits + "LL";
            }
            if (type == DataType.I16) {
                return "(int16_t)" + bits + "LL";
            }
            if (type == DataType.I32) {
                return "(int32_t)" + bits + "LL";
            }
            if (type == DataType.I64) {
                return "(int64_t)" + bits + "LL";
            }
            if (type == DataType.BOOL) {
                return bits != 0 ? "1" : "0";
            }
            throw new UnsupportedOperationException("Unsupported literal dtype: " + type);
        }

        private String formatFloatLiteral(float value) {
            if (Float.isNaN(value)) {
                return "NAN";
            }
            if (Float.isInfinite(value)) {
                return value > 0 ? "INFINITY" : "-INFINITY";
            }
            String str = Float.toString(value);
            if (!str.contains("E") && !str.contains("e")) {
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
                return "NAN";
            }
            if (Double.isInfinite(value)) {
                return value > 0 ? "INFINITY" : "-INFINITY";
            }
            String str = Double.toString(value);
            if (!str.contains("E") && !str.contains("e") && !str.contains(".")) {
                str = str + ".0";
            }
            return str;
        }

        private String castExpr(DataType source, DataType target, String expr) {
            if (source == target) {
                return expr;
            }
            if (target == DataType.BOOL) {
                return "(" + expr + " != 0 ? 1 : 0)";
            }
            if (target == DataType.FP16) {
                return "((_Float16)(" + toFloatExpr(source, expr) + "))";
            }
            if (target == DataType.BF16) {
                return "((__bf16)(" + toFloatExpr(source, expr) + "))";
            }
            if (target == DataType.FP32) {
                return toFloatExpr(source, expr);
            }
            if (target == DataType.FP64) {
                return toDoubleExpr(source, expr);
            }
            if (source == DataType.FP16 || source == DataType.BF16) {
                return "(" + typeName(target) + ")(" + toFloatExpr(source, expr) + ")";
            }
            return "(" + typeName(target) + ")(" + expr + ")";
        }

        private String toFloatExpr(DataType source, String expr) {
            if (source == DataType.FP16) {
                return "(float)(" + expr + ")";
            }
            if (source == DataType.BF16) {
                return "(float)(" + expr + ")";
            }
            if (source == DataType.FP32) {
                return expr;
            }
            if (source == DataType.FP64) {
                return "(float)(" + expr + ")";
            }
            if (source == DataType.BOOL) {
                return "(" + expr + " != 0 ? 1.0f : 0.0f)";
            }
            return "(float)(" + expr + ")";
        }

        private String toDoubleExpr(DataType source, String expr) {
            if (source == DataType.FP64) {
                return expr;
            }
            if (source == DataType.FP32) {
                return "(double)(" + expr + ")";
            }
            if (source == DataType.FP16 || source == DataType.BF16) {
                return "(double)(" + toFloatExpr(source, expr) + ")";
            }
            if (source == DataType.BOOL) {
                return "(" + expr + " != 0 ? 1.0 : 0.0)";
            }
            return "(double)(" + expr + ")";
        }

        private String negateExpr(DataType type, String expr) {
            if (type == DataType.FP16) {
                return "((_Float16)(-(float)(" + expr + ")))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)(-(float)(" + expr + ")))";
            }
            return "-" + expr;
        }

        private String absExpr(DataType type, String expr) {
            if (type == DataType.FP16) {
                return "((_Float16)fabsf(" + toFloatExpr(type, expr) + "))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)fabsf(" + toFloatExpr(type, expr) + "))";
            }
            if (type == DataType.FP32) {
                return "fabsf(" + expr + ")";
            }
            if (type == DataType.FP64) {
                return "fabs(" + expr + ")";
            }
            if (type.isIntegral()) {
                return "(" + expr + " < 0 ? -" + expr + " : " + expr + ")";
            }
            throw new UnsupportedOperationException("abs requires numeric type");
        }

        private String squareExpr(DataType type, String expr) {
            if (type == DataType.FP16) {
                return "((_Float16)((" + expr + ") * (" + expr + ")))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)((" + expr + ") * (" + expr + ")))";
            }
            return "(" + expr + " * " + expr + ")";
        }

        private String reciprocalExpr(DataType type, String expr) {
            if (type == DataType.FP16) {
                return "((_Float16)(1.0f / " + toFloatExpr(type, expr) + "))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)(1.0f / " + toFloatExpr(type, expr) + "))";
            }
            if (type == DataType.FP64) {
                return "(1.0 / " + expr + ")";
            }
            return "(1.0f / " + expr + ")";
        }

        private String floatUnaryExpr(DataType type, String fp64Fn, String fp32Fn, String expr) {
            if (type == DataType.FP16) {
                return "((_Float16)" + fp32Fn + "(" + toFloatExpr(type, expr) + "))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)" + fp32Fn + "(" + toFloatExpr(type, expr) + "))";
            }
            if (type == DataType.FP64) {
                return fp64Fn + "(" + expr + ")";
            }
            if (type == DataType.FP32) {
                return fp32Fn + "(" + expr + ")";
            }
            throw new UnsupportedOperationException("Floating-point unary op requires FP types");
        }

        private String binaryArithmetic(DataType type, String op, String left, String right) {
            if (type == DataType.FP16) {
                return "((_Float16)((" + left + ") " + op + " (" + right + ")))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)((" + left + ") " + op + " (" + right + ")))";
            }
            return "(" + left + " " + op + " " + right + ")";
        }

        private String minExpr(DataType type, String left, String right) {
            if (type == DataType.FP16) {
                return "((_Float16)fminf("
                        + toFloatExpr(type, left)
                        + ", "
                        + toFloatExpr(type, right)
                        + "))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)fminf("
                        + toFloatExpr(type, left)
                        + ", "
                        + toFloatExpr(type, right)
                        + "))";
            }
            if (type == DataType.FP64) {
                return "fmin(" + left + ", " + right + ")";
            }
            if (type == DataType.FP32) {
                return "fminf(" + left + ", " + right + ")";
            }
            return "(" + left + " < " + right + " ? " + left + " : " + right + ")";
        }

        private String maxExpr(DataType type, String left, String right) {
            if (type == DataType.FP16) {
                return "((_Float16)fmaxf("
                        + toFloatExpr(type, left)
                        + ", "
                        + toFloatExpr(type, right)
                        + "))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)fmaxf("
                        + toFloatExpr(type, left)
                        + ", "
                        + toFloatExpr(type, right)
                        + "))";
            }
            if (type == DataType.FP64) {
                return "fmax(" + left + ", " + right + ")";
            }
            if (type == DataType.FP32) {
                return "fmaxf(" + left + ", " + right + ")";
            }
            return "(" + left + " > " + right + " ? " + left + " : " + right + ")";
        }

        private String powExpr(DataType type, String left, String right) {
            if (type == DataType.FP16) {
                return "((_Float16)powf("
                        + toFloatExpr(type, left)
                        + ", "
                        + toFloatExpr(type, right)
                        + "))";
            }
            if (type == DataType.BF16) {
                return "((__bf16)powf("
                        + toFloatExpr(type, left)
                        + ", "
                        + toFloatExpr(type, right)
                        + "))";
            }
            if (type == DataType.FP64) {
                return "pow(" + left + ", " + right + ")";
            }
            if (type == DataType.FP32) {
                return "powf(" + left + ", " + right + ")";
            }
            throw new UnsupportedOperationException("pow requires floating-point type");
        }

        private String typeName(DataType dataType) {
            if (dataType == DataType.BOOL) {
                return "uint8_t";
            }
            if (dataType == DataType.I8) {
                return "int8_t";
            }
            if (dataType == DataType.I16) {
                return "int16_t";
            }
            if (dataType == DataType.I32) {
                return "int32_t";
            }
            if (dataType == DataType.I64) {
                return "int64_t";
            }
            if (dataType == DataType.FP16) {
                return "_Float16";
            }
            if (dataType == DataType.BF16) {
                return "__bf16";
            }
            if (dataType == DataType.FP32) {
                return "float";
            }
            if (dataType == DataType.FP64) {
                return "double";
            }
            throw new UnsupportedOperationException("Unsupported C data type: " + dataType);
        }

        private void addLine(String line) {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < indentLevel; i++) {
                builder.append("  ");
            }
            builder.append(line).append("\n");
            lines.add(builder.toString());
        }

        private static class BufferVar {
            private final String name;
            private final DataType type;
            private final long[] byteStrides;

            protected BufferVar(String name, DataType type, long[] byteStrides) {
                this.name = name;
                this.type = type;
                this.byteStrides = byteStrides;
            }
        }

        private static final class ScratchBufferVar extends BufferVar {
            private final long offset;

            private ScratchBufferVar(String name, DataType type, long offset) {
                super(name, type, null);
                this.offset = offset;
            }
        }
    }
}
