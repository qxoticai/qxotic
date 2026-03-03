package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.*;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.clike.CLikeExprSupport;
import com.qxotic.jota.runtime.clike.CLikeKernelGenerator;
import com.qxotic.jota.runtime.clike.CLikeLogicalSupport;
import com.qxotic.jota.runtime.clike.CLikeParallelLoopSupport;
import com.qxotic.jota.runtime.clike.CLikeScalarSupport;
import com.qxotic.jota.runtime.clike.CLikeSourceSupport;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

final class MetalKernelProgramGenerator {

    private final CLikeKernelGenerator generator = new CLikeKernelGenerator(new MetalDialect());

    KernelProgram generate(LIRGraph graph, ScratchLayout scratchLayout, KernelCacheKey key) {
        return generator.generate(graph, scratchLayout, key);
    }

    static final class SourceGenerator {
        private final LIRGraph graph;
        private final ScratchLayout scratchLayout;
        private final String kernelName;
        private final LIRExprGraph exprGraph;
        private final Map<Integer, String> scalarInputNames = new HashMap<>();
        private final Map<LIRExprNode, String> tempNames = new IdentityHashMap<>();
        private final Map<BufferRef, BufferVar> buffers = new IdentityHashMap<>();
        private final List<String> lines = new ArrayList<>();
        private int tempId;
        private int indentLevel;

        private SourceGenerator(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
            this.graph = graph;
            this.scratchLayout = scratchLayout;
            this.kernelName = kernelName;
            this.exprGraph = graph.exprGraph();
        }

        static String generate(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
            SourceGenerator generator = new SourceGenerator(graph, scratchLayout, kernelName);
            return generator.generate();
        }

        private String generate() {
            StringBuilder source = new StringBuilder();
            source.append("#include <metal_stdlib>\n");
            source.append("using namespace metal;\n\n");
            source.append("inline float jota_bf16_to_float(ushort bits) {\n");
            source.append("  return as_type<float>(((uint)bits) << 16);\n");
            source.append("}\n");
            source.append("inline ushort jota_float_to_bf16(float v) {\n");
            source.append("  return (ushort)(as_type<uint>(v) >> 16);\n");
            source.append("}\n\n");

            source.append("kernel void ").append(kernelName).append("(");
            source.append(kernelSignature());
            source.append(") {\n");
            indentLevel = 1;
            lines.clear();
            emitProlog();
            if (!emitParallelTopLevel(graph.body())) {
                addLine("if (gid.x == 0u && gid.y == 0u && gid.z == 0u) {");
                indentLevel++;
                emitNode(graph.body());
                indentLevel--;
                addLine("}");
            }
            for (String line : lines) {
                source.append(line);
            }
            source.append("}\n");
            return source.toString();
        }

        private boolean emitParallelTopLevel(LIRExprNode body) {
            return CLikeParallelLoopSupport.emitParallelTopLevel(
                    body,
                    CLikeParallelLoopSupport.emitter(
                            this::isConstZero,
                            this::isConstOne,
                            this::emitIndexExpr,
                            this::nextTempName,
                            this::addLine,
                            () -> indentLevel++,
                            () -> indentLevel--,
                            this::emitNode),
                    new CLikeParallelLoopSupport.LinearIdSpec("long", "(long)gid.x", "1L"));
        }

        private String kernelSignature() {
            StringBuilder signature = new StringBuilder();
            int argIndex = 0;
            for (LIRInput input : graph.inputs()) {
                if (input instanceof BufferRef buffer) {
                    String cType = typeName(buffer.dataType());
                    signature
                            .append("const device ")
                            .append(cType)
                            .append(" *input")
                            .append(argIndex)
                            .append(" [[buffer(")
                            .append(argIndex)
                            .append(")]]");
                } else if (input instanceof ScalarInput scalar) {
                    signature
                            .append("constant ")
                            .append(typeName(scalar.dataType()))
                            .append(" *")
                            .append(scalarParamName(argIndex))
                            .append(" [[buffer(")
                            .append(argIndex)
                            .append(")]]");
                }
                signature.append(", ");
                argIndex++;
            }
            for (int i = 0; i < graph.outputs().size(); i++) {
                int slot = graph.inputs().size() + i;
                BufferRef buffer = graph.outputs().get(i);
                String cType = typeName(buffer.dataType());
                signature
                        .append("device ")
                        .append(cType)
                        .append(" *output")
                        .append(i)
                        .append(" [[buffer(")
                        .append(slot)
                        .append(")]], ");
            }
            int scratchSlot = graph.inputs().size() + graph.outputs().size();
            signature.append("device uchar *scratch [[buffer(").append(scratchSlot).append(")]], ");
            signature.append("uint3 gid [[thread_position_in_grid]]");
            return signature.toString();
        }

        private void emitProlog() {
            int argIndex = 0;
            for (LIRInput input : graph.inputs()) {
                if (input instanceof BufferRef buffer) {
                    String name = "input" + argIndex;
                    buffers.put(buffer, new BufferVar(name, true));
                } else if (input instanceof ScalarInput scalar) {
                    String name = "scalar" + argIndex;
                    String paramName = scalarParamName(argIndex);
                    scalarInputNames.put(scalar.id(), name);
                    addLine(typeName(scalar.dataType()) + " " + name + " = " + paramName + "[0];");
                }
                argIndex++;
            }
            for (int i = 0; i < graph.outputs().size(); i++) {
                BufferRef buffer = graph.outputs().get(i);
                buffers.put(buffer, new BufferVar("output" + i, false));
            }
            if (scratchLayout.requiresScratch()) {
                int slotId = 0;
                for (var entry : scratchLayout.offsets().entrySet()) {
                    BufferRef buf = entry.getKey();
                    long offset = entry.getValue();
                    String name = "scratch" + slotId++;
                    buffers.put(buf, new BufferVar(name, false));
                    addLine(
                            "device "
                                    + typeName(buf.dataType())
                                    + " *"
                                    + name
                                    + " = (device "
                                    + typeName(buf.dataType())
                                    + " *)(scratch + "
                                    + offset
                                    + "L);");
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
                    step.equals("1L") || (loop.step() instanceof IConst ic && ic.value() == 1);

            if (isSimpleForwardLoop) {
                for (LoopIterArg arg : loop.iterArgs()) {
                    String initExpr = emitScalarExpr(arg.init());
                    addLine(typeName(arg.dataType()) + " " + arg.name() + " = " + initExpr + ";");
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
                String lbVar = idx + "_lb";
                String ubVar = idx + "_ub";
                String stepVar = idx + "_step";
                addLine("long " + lbVar + " = " + lb + ";");
                addLine("long " + ubVar + " = " + ub + ";");
                addLine("long " + stepVar + " = " + step + ";");
                for (LoopIterArg arg : loop.iterArgs()) {
                    String initExpr = emitScalarExpr(arg.init());
                    addLine(typeName(arg.dataType()) + " " + arg.name() + " = " + initExpr + ";");
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
            return CLikeSourceSupport.extractYield(body);
        }

        private boolean isConstZero(LIRExprNode node) {
            return CLikeSourceSupport.isConstZero(node, exprGraph);
        }

        private boolean isConstOne(LIRExprNode node) {
            return CLikeSourceSupport.isConstOne(node, exprGraph);
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
                if (var.equals(expr)) {
                    return expr;
                }
                addLine(typeName(resolved.dataType()) + " " + var + " = " + expr + ";");
                tempNames.put(resolved, var);
                return var;
            }
            return emitScalarExprInline(resolved);
        }

        private String emitScalarExprInline(LIRExprNode resolved) {
            return switch (resolved.kind()) {
                case S_CONST -> scalarLiteral(((SConst) resolved).rawBits(), resolved.dataType());
                case S_INPUT -> requireScalarInputById(((SInput) resolved).inputId());
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
            return CLikeSourceSupport.shouldMaterializeTemp(node, expr);
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
                case EXP -> floatUnaryExpr(type, "exp", input);
                case LOG -> floatUnaryExpr(type, "log", input);
                case SQRT -> floatUnaryExpr(type, "sqrt", input);
                case SQUARE -> squareExpr(type, input);
                case SIN -> floatUnaryExpr(type, "sin", input);
                case COS -> floatUnaryExpr(type, "cos", input);
                case TAN -> floatUnaryExpr(type, "tan", input);
                case TANH -> floatUnaryExpr(type, "tanh", input);
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
            if (binary.op() == BinaryOperator.ADD) {
                String l =
                        CLikeLogicalSupport.maybeConvertBoolToNumeric(
                                left, binary.left().dataType(), type);
                String r =
                        CLikeLogicalSupport.maybeConvertBoolToNumeric(
                                right, binary.right().dataType(), type);
                return binaryArithmetic(type, "+", l, r);
            }
            if (binary.op() == BinaryOperator.SUBTRACT) {
                return binaryArithmetic(type, "-", left, right);
            }
            if (binary.op() == BinaryOperator.MULTIPLY) {
                return binaryArithmetic(type, "*", left, right);
            }
            if (binary.op() == BinaryOperator.DIVIDE) {
                return binaryArithmetic(type, "/", left, right);
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
                return CLikeLogicalSupport.logicalBinary(
                        "&&",
                        left,
                        binary.left().dataType(),
                        right,
                        binary.right().dataType(),
                        type);
            }
            if (binary.op() == BinaryOperator.LOGICAL_OR) {
                return CLikeLogicalSupport.logicalBinary(
                        "||",
                        left,
                        binary.left().dataType(),
                        right,
                        binary.right().dataType(),
                        type);
            }
            if (binary.op() == BinaryOperator.LOGICAL_XOR) {
                return CLikeLogicalSupport.logicalXor(
                        left, binary.left().dataType(), right, binary.right().dataType(), type);
            }
            if (binary.op() == BinaryOperator.BITWISE_AND) {
                return castExpr(type, type, "(" + left + " & " + right + ")");
            }
            if (binary.op() == BinaryOperator.BITWISE_OR) {
                return castExpr(type, type, "(" + left + " | " + right + ")");
            }
            if (binary.op() == BinaryOperator.BITWISE_XOR) {
                return castExpr(type, type, "(" + left + " ^ " + right + ")");
            }
            if (binary.op() == BinaryOperator.SHIFT_LEFT) {
                return castExpr(
                        type, type, "(" + left + " << " + normalizedShift(type, right) + ")");
            }
            if (binary.op() == BinaryOperator.SHIFT_RIGHT) {
                return castExpr(
                        type, type, "(" + left + " >> " + normalizedShift(type, right) + ")");
            }
            if (binary.op() == BinaryOperator.SHIFT_RIGHT_UNSIGNED) {
                return unsignedRightShiftExpr(type, left, right);
            }
            if (binary.op() == BinaryOperator.EQUAL) {
                return compareExpr("==", binary.left(), binary.right(), type);
            }
            if (binary.op() == BinaryOperator.LESS_THAN) {
                return compareExpr("<", binary.left(), binary.right(), type);
            }
            throw new UnsupportedOperationException("Unsupported binary operator: " + binary.op());
        }

        private String normalizedShift(DataType type, String right) {
            return CLikeExprSupport.normalizedShift(type, right);
        }

        private String unsignedRightShiftExpr(DataType type, String left, String right) {
            String shift = normalizedShift(type, right);
            String unsignedType =
                    switch (typeName(type)) {
                        case "char" -> "uchar";
                        case "short" -> "ushort";
                        case "int" -> "uint";
                        case "long" -> "ulong";
                        default ->
                                throw new UnsupportedOperationException(
                                        "Unsupported shift type: " + type);
                    };
            String expr = "((" + unsignedType + ")(" + left + ") >> " + shift + ")";
            return castExpr(type, type, expr);
        }

        private String emitTernaryExpr(STernary ternary) {
            String cond = emitScalarExpr(ternary.condition());
            String tVal = emitScalarExpr(ternary.trueValue());
            String fVal = emitScalarExpr(ternary.falseValue());
            return CLikeScalarSupport.ternaryExpr(cond, ternary.condition().dataType(), tVal, fVal);
        }

        private String emitCastExpr(SCast cast) {
            String input = emitScalarExpr(cast.input());
            return castExpr(cast.input().dataType(), cast.targetType(), input);
        }

        private String emitIndexExpr(LIRExprNode node) {
            LIRExprNode resolved = exprGraph.resolve(node);
            return switch (resolved.kind()) {
                case I_CONST -> ((IConst) resolved).value() + "L";
                case I_VAR -> ((IVar) resolved).name();
                case I_BINARY ->
                        "("
                                + emitIndexExpr(((IBinary) resolved).left())
                                + " "
                                + indexOp(((IBinary) resolved).op())
                                + " "
                                + emitIndexExpr(((IBinary) resolved).right())
                                + ")";
                case I_FROM_SCALAR -> {
                    String scalarExpr = emitScalarExpr(((IFromScalar) resolved).scalarExpr());
                    yield "((long)(" + scalarExpr + "))";
                }
                default ->
                        throw new IllegalStateException(
                                "Expected index node, got " + resolved.kind());
            };
        }

        private String compareExpr(
                String op, LIRExprNode leftExpr, LIRExprNode rightExpr, DataType resultType) {
            String left = emitScalarExpr(leftExpr);
            String right = emitScalarExpr(rightExpr);
            return CLikeScalarSupport.comparisonExpr(
                    op,
                    left,
                    leftExpr.dataType(),
                    right,
                    rightExpr.dataType(),
                    resultType,
                    this::castExpr,
                    this::toFloatExpr);
        }

        private String indexOp(IndexBinaryOp op) {
            return CLikeExprSupport.indexOp(op);
        }

        private BufferVar requireBuffer(BufferRef ref) {
            BufferVar buffer = buffers.get(ref);
            if (buffer == null) {
                throw new IllegalStateException("Missing buffer mapping for " + ref);
            }
            return buffer;
        }

        private String requireScalarInputById(int id) {
            String name = scalarInputNames.get(id);
            if (name == null) {
                throw new IllegalStateException("Missing scalar input " + id);
            }
            return name;
        }

        private String nextTempName() {
            String name;
            do {
                name = "t" + tempId++;
            } while (scalarInputNames.containsValue(name));
            return name;
        }

        private String readValue(BufferVar buffer, DataType type, String offset) {
            String space = buffer.readOnly ? "const device" : "device";
            String ptr = "((" + space + " uchar*)" + buffer.name + " + " + offset + ")";
            return "*((" + space + " " + typeName(type) + "*)" + ptr + ")";
        }

        private String writeValue(
                BufferVar buffer, DataType type, String offset, String value, DataType valueType) {
            if (buffer.readOnly) {
                throw new IllegalStateException("Attempting to write to read-only buffer");
            }
            String ptr = "((device uchar*)" + buffer.name + " + " + offset + ")";
            String castValue = castExpr(valueType, type, value);
            if (type == DataType.BOOL) {
                castValue = CLikeScalarSupport.boolStoreValue(value, valueType);
            }
            return "*((device " + typeName(type) + "*)" + ptr + ") = " + castValue + ";";
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
                return "((half)" + formatFloatLiteral(value) + ")";
            }
            if (type == DataType.BF16) {
                int bf16Bits = ((int) bits) & 0xFFFF;
                return "(ushort)" + bf16Bits + "u";
            }
            if (type == DataType.I8) {
                return "(char)" + bits + "L";
            }
            if (type == DataType.I16) {
                return "(short)" + bits + "L";
            }
            if (type == DataType.I32) {
                return "(int)" + bits + "L";
            }
            if (type == DataType.I64) {
                return "(long)" + bits + "L";
            }
            if (type == DataType.BOOL) {
                return bits != 0 ? "1" : "0";
            }
            throw new UnsupportedOperationException("Unsupported literal dtype for Metal: " + type);
        }

        private String formatFloatLiteral(float value) {
            return CLikeExprSupport.formatFloatLiteral(value);
        }

        private String formatDoubleLiteral(double value) {
            return CLikeExprSupport.formatDoubleLiteral(value);
        }

        private String castExpr(DataType source, DataType target, String expr) {
            if (source == target) {
                return expr;
            }
            if (target == DataType.BOOL) {
                return "(" + expr + " != 0 ? 1 : 0)";
            }
            if (target == DataType.FP16) {
                return "((half)(" + toFloatExpr(source, expr) + "))";
            }
            if (target == DataType.BF16) {
                return "jota_float_to_bf16(" + toFloatExpr(source, expr) + ")";
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
                return "jota_bf16_to_float(" + expr + ")";
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
            if (type == DataType.FP16 || type == DataType.BF16) {
                return castExpr(DataType.FP32, type, "-" + toFloatExpr(type, expr));
            }
            return "-" + expr;
        }

        private String absExpr(DataType type, String expr) {
            if (type == DataType.FP16 || type == DataType.BF16) {
                return castExpr(DataType.FP32, type, "fabs(" + toFloatExpr(type, expr) + ")");
            }
            if (type == DataType.FP32 || type == DataType.FP64) {
                return "fabs(" + expr + ")";
            }
            if (type.isIntegral()) {
                return "(" + expr + " < 0 ? -" + expr + " : " + expr + ")";
            }
            throw new UnsupportedOperationException("abs requires numeric type");
        }

        private String squareExpr(DataType type, String expr) {
            if (type == DataType.FP16 || type == DataType.BF16) {
                String f = toFloatExpr(type, expr);
                return castExpr(DataType.FP32, type, "(" + f + " * " + f + ")");
            }
            return "(" + expr + " * " + expr + ")";
        }

        private String reciprocalExpr(DataType type, String expr) {
            if (type == DataType.FP16 || type == DataType.BF16) {
                return castExpr(DataType.FP32, type, "(1.0f / " + toFloatExpr(type, expr) + ")");
            }
            if (type == DataType.FP64) {
                return "(1.0 / " + expr + ")";
            }
            return "(1.0f / " + expr + ")";
        }

        private String floatUnaryExpr(DataType type, String fn, String expr) {
            if (type == DataType.FP16 || type == DataType.BF16) {
                return castExpr(DataType.FP32, type, fn + "(" + toFloatExpr(type, expr) + ")");
            }
            if (type == DataType.FP32 || type == DataType.FP64) {
                return fn + "(" + expr + ")";
            }
            throw new UnsupportedOperationException(
                    "Floating-point unary op requires FP16/BF16/FP32/FP64");
        }

        private String binaryArithmetic(DataType type, String op, String left, String right) {
            if (type == DataType.FP16 || type == DataType.BF16) {
                String l = toFloatExpr(type, left);
                String r = toFloatExpr(type, right);
                return castExpr(DataType.FP32, type, "(" + l + " " + op + " " + r + ")");
            }
            return "(" + left + " " + op + " " + right + ")";
        }

        private String minExpr(DataType type, String left, String right) {
            if (type == DataType.FP16 || type == DataType.BF16) {
                String l = toFloatExpr(type, left);
                String r = toFloatExpr(type, right);
                return castExpr(DataType.FP32, type, "fmin(" + l + ", " + r + ")");
            }
            if (type == DataType.FP32 || type == DataType.FP64) {
                return "fmin(" + left + ", " + right + ")";
            }
            return "(" + left + " < " + right + " ? " + left + " : " + right + ")";
        }

        private String maxExpr(DataType type, String left, String right) {
            if (type == DataType.FP16 || type == DataType.BF16) {
                String l = toFloatExpr(type, left);
                String r = toFloatExpr(type, right);
                return castExpr(DataType.FP32, type, "fmax(" + l + ", " + r + ")");
            }
            if (type == DataType.FP32 || type == DataType.FP64) {
                return "fmax(" + left + ", " + right + ")";
            }
            return "(" + left + " > " + right + " ? " + left + " : " + right + ")";
        }

        private String powExpr(DataType type, String left, String right) {
            if (type == DataType.FP16 || type == DataType.BF16) {
                String l = toFloatExpr(type, left);
                String r = toFloatExpr(type, right);
                return castExpr(DataType.FP32, type, "pow(" + l + ", " + r + ")");
            }
            if (type == DataType.FP32 || type == DataType.FP64) {
                return "pow(" + left + ", " + right + ")";
            }
            throw new UnsupportedOperationException("pow requires floating-point type");
        }

        private String scalarParamName(int index) {
            return "scalarPtr" + index;
        }

        private String typeName(DataType dataType) {
            if (dataType == DataType.BOOL) {
                return "uchar";
            }
            if (dataType == DataType.I8) {
                return "char";
            }
            if (dataType == DataType.I16) {
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
            if (dataType == DataType.FP16) {
                return "half";
            }
            if (dataType == DataType.BF16) {
                return "ushort";
            }
            throw new UnsupportedOperationException("Unsupported Metal data type: " + dataType);
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
            private final boolean readOnly;

            private BufferVar(String name, boolean readOnly) {
                this.name = name;
                this.readOnly = readOnly;
            }
        }
    }
}
