package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.*;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.tir.BinaryOperator;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

public abstract class CLikeLirSourceGenerator {

    @FunctionalInterface
    protected interface InputEmitter {
        void emit(LIRInput input, int inputIndex);
    }

    @FunctionalInterface
    protected interface OutputEmitter {
        void emit(BufferRef output, int outputIndex);
    }

    @FunctionalInterface
    protected interface ScratchEmitter {
        void emit(BufferRef scratchBuffer, long offset, int scratchSlot);
    }

    private final LIRGraph graph;
    private final ScratchLayout scratchLayout;
    private final String kernelName;
    private final LIRExprGraph exprGraph;
    private final Map<Integer, String> scalarInputNames = new HashMap<>();
    private final Map<LIRExprNode, String> tempNames = new IdentityHashMap<>();
    private final Map<BufferRef, BufferBinding> buffers = new IdentityHashMap<>();
    private final List<String> lines = new ArrayList<>();
    private int tempId;
    private int indentLevel;

    protected CLikeLirSourceGenerator(
            LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        this.graph = graph;
        this.scratchLayout = scratchLayout;
        this.kernelName = kernelName;
        this.exprGraph = graph.exprGraph();
    }

    protected final LIRGraph graph() {
        return graph;
    }

    protected final ScratchLayout scratchLayout() {
        return scratchLayout;
    }

    protected final String kernelName() {
        return kernelName;
    }

    public final String generate() {
        StringBuilder source = new StringBuilder();
        appendPreamble(source);
        source.append(kernelDeclaration()).append(" {\n");
        scalarInputNames.clear();
        tempNames.clear();
        buffers.clear();
        tempId = 0;
        indentLevel = 1;
        lines.clear();
        emitProlog();
        if (!emitTopLevel()) {
            emitNode(graph.body());
        }
        for (String line : lines) {
            source.append(line);
        }
        source.append("}\n");
        return source.toString();
    }

    protected abstract void appendPreamble(StringBuilder source);

    protected abstract String kernelDeclaration();

    protected abstract void emitProlog();

    protected boolean tryEmitParallelTopLevel(LIRExprNode body) {
        return false;
    }

    protected String serialFallbackGuardCondition() {
        return null;
    }

    protected abstract CLikeParallelLoopSupport.LinearIdSpec linearIdSpec();

    protected abstract String indexTypeName();

    protected abstract String indexLiteral(long value);

    protected abstract String renderDataTypeToken(DataType dataType);

    protected abstract String renderScalarLiteralFromBits(long bits, DataType type);

    protected abstract String renderCastExpression(DataType source, DataType target, String expr);

    protected abstract String renderFloat32ConversionExpression(DataType source, String expr);

    protected abstract String renderFloat64ConversionExpression(DataType source, String expr);

    protected abstract String renderBufferReadExpression(
            BufferBinding buffer, DataType type, String offset);

    protected abstract String renderBufferWriteStatement(
            BufferBinding buffer, DataType type, String offset, String value, DataType valueType);

    protected abstract String unsignedShiftCarrierTypeName(DataType type);

    protected abstract String float32MathFunctionName(String base);

    protected abstract String float64MathFunctionName(String base);

    protected boolean shouldAvoidTempNameCollisionWithScalars() {
        return true;
    }

    protected void maybeEmitSimpleLoopPragma(StructuredFor loop) {}

    protected final void registerScalarInputName(int id, String name) {
        scalarInputNames.put(id, name);
    }

    protected final void registerBuffer(BufferRef ref, String name) {
        buffers.put(ref, new BufferBinding(name, false));
    }

    protected final void registerReadOnlyBuffer(BufferRef ref, String name) {
        buffers.put(ref, new BufferBinding(name, true));
    }

    protected final BufferBinding requireBuffer(BufferRef ref) {
        BufferBinding buffer = buffers.get(ref);
        if (buffer == null) {
            throw new IllegalStateException("Missing buffer mapping for " + ref);
        }
        return buffer;
    }

    protected final String inputBufferName(int inputIndex) {
        return "input" + inputIndex;
    }

    protected final String outputBufferName(int outputIndex) {
        return "output" + outputIndex;
    }

    protected final String scalarLocalName(int inputIndex) {
        return "scalar" + inputIndex;
    }

    protected final String scratchBufferName(int scratchSlot) {
        return "scratch" + scratchSlot;
    }

    protected final void bindReadOnlyInputBuffer(BufferRef inputBuffer, int inputIndex) {
        registerReadOnlyBuffer(inputBuffer, inputBufferName(inputIndex));
    }

    protected final void bindScalarInput(ScalarInput scalar, int inputIndex, String valueExpr) {
        String name = scalarLocalName(inputIndex);
        registerScalarInputName(scalar.id(), name);
        addLine(renderDataTypeToken(scalar.dataType()) + " " + name + " = " + valueExpr + ";");
    }

    protected final ScalarInput requireScalarInput(LIRInput input) {
        if (input instanceof ScalarInput scalar) {
            return scalar;
        }
        throw new UnsupportedOperationException("Unsupported input type: " + input.getClass());
    }

    protected final void emitInputs(InputEmitter inputEmitter) {
        for (int i = 0; i < graph.inputs().size(); i++) {
            inputEmitter.emit(graph.inputs().get(i), i);
        }
    }

    protected final void emitOutputs(OutputEmitter outputEmitter) {
        for (int i = 0; i < graph.outputs().size(); i++) {
            outputEmitter.emit(graph.outputs().get(i), i);
        }
    }

    protected final void emitScratchBindings(ScratchEmitter scratchEmitter) {
        if (!scratchLayout.requiresScratch()) {
            return;
        }
        List<Map.Entry<BufferRef, Long>> entries =
                new ArrayList<>(scratchLayout.offsets().entrySet());
        entries.sort(
                Comparator.comparingLong((Map.Entry<BufferRef, Long> e) -> e.getValue())
                        .thenComparingInt(e -> e.getKey().id()));
        int slotId = 0;
        for (Map.Entry<BufferRef, Long> entry : entries) {
            scratchEmitter.emit(entry.getKey(), entry.getValue(), slotId++);
        }
    }

    private boolean emitTopLevel() {
        return tryEmitParallelTopLevel(graph.body());
    }

    protected final boolean emitParallelTopLevel(LIRExprNode body) {
        boolean emitted =
                CLikeParallelLoopSupport.emitParallelTopLevel(
                        body,
                        CLikeParallelLoopSupport.emitter(
                                this::isConstZero,
                                this::isConstOne,
                                this::emitIndexExpr,
                                this::nextTempName,
                                this::addLine,
                                this::indent,
                                this::outdent,
                                this::emitNode),
                        linearIdSpec());
        if (emitted) {
            return true;
        }
        String guard = serialFallbackGuardCondition();
        if (guard == null || guard.isBlank()) {
            return false;
        }
        addLine("if (" + guard + ") {");
        indent();
        emitNode(graph.body());
        outdent();
        addLine("}");
        return true;
    }

    protected final void emitNode(LIRExprNode node) {
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

        boolean isSimpleForwardLoop = isConstOne(loop.step());
        if (isSimpleForwardLoop) {
            for (LoopIterArg arg : loop.iterArgs()) {
                String initExpr = emitScalarExpr(arg.init());
                addLine(
                        renderDataTypeToken(arg.dataType())
                                + " "
                                + arg.name()
                                + " = "
                                + initExpr
                                + ";");
            }
            maybeEmitSimpleLoopPragma(loop);
            addLine(
                    "for ("
                            + indexTypeName()
                            + " "
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
            indent();
            emitStructuredBody(loop);
            outdent();
            addLine("}");
            return;
        }

        String lbVar = idx + "_lb";
        String ubVar = idx + "_ub";
        String stepVar = idx + "_step";
        addLine(indexTypeName() + " " + lbVar + " = " + lb + ";");
        addLine(indexTypeName() + " " + ubVar + " = " + ub + ";");
        addLine(indexTypeName() + " " + stepVar + " = " + step + ";");

        for (LoopIterArg arg : loop.iterArgs()) {
            String initExpr = emitScalarExpr(arg.init());
            addLine(
                    renderDataTypeToken(arg.dataType())
                            + " "
                            + arg.name()
                            + " = "
                            + initExpr
                            + ";");
        }
        addLine(
                "for ("
                        + indexTypeName()
                        + " "
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
        indent();
        emitStructuredBody(loop);
        outdent();
        addLine("}");
    }

    private void emitStructuredBody(StructuredFor loop) {
        Block body = loop.body();
        Yield yield = CLikeSourceSupport.extractYield(body);
        for (int i = 0; i < body.statements().size() - 1; i++) {
            emitNode(body.statements().get(i));
        }

        List<String> nextNames = new ArrayList<>(loop.iterArgs().size());
        for (int i = 0; i < loop.iterArgs().size(); i++) {
            LoopIterArg arg = loop.iterArgs().get(i);
            String nextName = arg.name() + "_next" + tempId++;
            String expr = emitScalarExpr(yield.values().get(i));
            addLine(renderDataTypeToken(arg.dataType()) + " " + nextName + " = " + expr + ";");
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

    private boolean isConstZero(LIRExprNode node) {
        return CLikeSourceSupport.isConstZero(node, exprGraph);
    }

    private boolean isConstOne(LIRExprNode node) {
        return CLikeSourceSupport.isConstOne(node, exprGraph);
    }

    private void emitStore(Store store) {
        BufferBinding buffer = requireBuffer(store.buffer());
        String value = emitScalarExpr(store.value());
        DataType bufferType = store.buffer().dataType();
        DataType valueType = store.value().dataType();
        String offset = emitIndexExpr(store.offset());
        addLine(renderBufferWriteStatement(buffer, bufferType, offset, value, valueType));
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
            if (!CLikeSourceSupport.shouldMaterializeTemp(resolved, expr)) {
                return expr;
            }
            String var = nextTempName();
            if (var.equals(expr)) {
                return expr;
            }
            addLine(renderDataTypeToken(resolved.dataType()) + " " + var + " = " + expr + ";");
            tempNames.put(resolved, var);
            return var;
        }
        return emitScalarExprInline(resolved);
    }

    private String emitScalarExprInline(LIRExprNode resolved) {
        return switch (resolved.kind()) {
            case S_CONST ->
                    renderScalarLiteralFromBits(((SConst) resolved).rawBits(), resolved.dataType());
            case S_INPUT -> requireScalarInputById(((SInput) resolved).inputId());
            case S_REF -> ((SRef) resolved).name();
            case S_FROM_INDEX -> {
                String indexExpr = emitIndexExpr(((SFromIndex) resolved).indexExpr());
                if (resolved.dataType() == DataType.I64) {
                    yield indexExpr;
                }
                yield renderCastExpression(DataType.I64, resolved.dataType(), indexExpr);
            }
            case S_LOAD -> emitScalarLoadExpr((SLoad) resolved);
            case S_UNARY -> emitUnaryExpr((SUnary) resolved);
            case S_BINARY -> emitBinaryExpr((SBinary) resolved);
            case S_TERNARY -> emitTernaryExpr((STernary) resolved);
            case S_CAST -> emitCastExpr((SCast) resolved);
            default ->
                    throw new IllegalStateException("Expected scalar node, got " + resolved.kind());
        };
    }

    private String emitScalarLoadExpr(SLoad load) {
        BufferBinding buffer = requireBuffer(load.buffer());
        String offset = emitIndexExpr(load.offset());
        return renderBufferReadExpression(buffer, load.buffer().dataType(), offset);
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
                    "&&", left, binary.left().dataType(), right, binary.right().dataType(), type);
        }
        if (binary.op() == BinaryOperator.LOGICAL_OR) {
            return CLikeLogicalSupport.logicalBinary(
                    "||", left, binary.left().dataType(), right, binary.right().dataType(), type);
        }
        if (binary.op() == BinaryOperator.LOGICAL_XOR) {
            return CLikeLogicalSupport.logicalXor(
                    left, binary.left().dataType(), right, binary.right().dataType(), type);
        }
        if (binary.op() == BinaryOperator.BITWISE_AND) {
            return renderCastExpression(type, type, "(" + left + " & " + right + ")");
        }
        if (binary.op() == BinaryOperator.BITWISE_OR) {
            return renderCastExpression(type, type, "(" + left + " | " + right + ")");
        }
        if (binary.op() == BinaryOperator.BITWISE_XOR) {
            return renderCastExpression(type, type, "(" + left + " ^ " + right + ")");
        }
        if (binary.op() == BinaryOperator.SHIFT_LEFT) {
            return renderCastExpression(
                    type, type, "(" + left + " << " + normalizedShift(type, right) + ")");
        }
        if (binary.op() == BinaryOperator.SHIFT_RIGHT) {
            return renderCastExpression(
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

    private String unsignedRightShiftExpr(DataType type, String left, String right) {
        String shift = normalizedShift(type, right);
        String unsignedType = unsignedShiftCarrierTypeName(type);
        String expr = "((" + unsignedType + ")(" + left + ") >> " + shift + ")";
        return renderCastExpression(type, type, expr);
    }

    private String emitTernaryExpr(STernary ternary) {
        String cond = emitScalarExpr(ternary.condition());
        String tVal = emitScalarExpr(ternary.trueValue());
        String fVal = emitScalarExpr(ternary.falseValue());
        return CLikeScalarSupport.ternaryExpr(cond, ternary.condition().dataType(), tVal, fVal);
    }

    private String emitCastExpr(SCast cast) {
        String input = emitScalarExpr(cast.input());
        return renderCastExpression(cast.input().dataType(), cast.targetType(), input);
    }

    private String emitIndexExpr(LIRExprNode node) {
        LIRExprNode resolved = exprGraph.resolve(node);
        return switch (resolved.kind()) {
            case I_CONST -> indexLiteral(((IConst) resolved).value());
            case I_VAR -> ((IVar) resolved).name();
            case I_BINARY ->
                    "("
                            + emitIndexExpr(((IBinary) resolved).left())
                            + " "
                            + CLikeExprSupport.indexOp(((IBinary) resolved).op())
                            + " "
                            + emitIndexExpr(((IBinary) resolved).right())
                            + ")";
            case I_FROM_SCALAR -> {
                String scalarExpr = emitScalarExpr(((IFromScalar) resolved).scalarExpr());
                yield "((" + indexTypeName() + ")(" + scalarExpr + "))";
            }
            default ->
                    throw new IllegalStateException("Expected index node, got " + resolved.kind());
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
                this::renderCastExpression,
                this::renderFloat32ConversionExpression);
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
        } while (shouldAvoidTempNameCollisionWithScalars() && scalarInputNames.containsValue(name));
        return name;
    }

    private String normalizedShift(DataType type, String right) {
        return CLikeExprSupport.normalizedShift(type, right);
    }

    private String negateExpr(DataType type, String expr) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            return renderCastExpression(
                    DataType.FP32, type, "-" + renderFloat32ConversionExpression(type, expr));
        }
        return "-" + expr;
    }

    private String absExpr(DataType type, String expr) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            return renderCastExpression(
                    DataType.FP32,
                    type,
                    float32MathFunctionName("fabs")
                            + "("
                            + renderFloat32ConversionExpression(type, expr)
                            + ")");
        }
        if (type == DataType.FP32) {
            return float32MathFunctionName("fabs") + "(" + expr + ")";
        }
        if (type == DataType.FP64) {
            return float64MathFunctionName("fabs") + "(" + expr + ")";
        }
        if (type.isIntegral()) {
            return "(" + expr + " < 0 ? -" + expr + " : " + expr + ")";
        }
        throw new UnsupportedOperationException("abs requires numeric type");
    }

    private String squareExpr(DataType type, String expr) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            String f = renderFloat32ConversionExpression(type, expr);
            return renderCastExpression(DataType.FP32, type, "(" + f + " * " + f + ")");
        }
        return "(" + expr + " * " + expr + ")";
    }

    private String reciprocalExpr(DataType type, String expr) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            return renderCastExpression(
                    DataType.FP32,
                    type,
                    "(1.0f / " + renderFloat32ConversionExpression(type, expr) + ")");
        }
        if (type == DataType.FP64) {
            return "(1.0 / " + expr + ")";
        }
        return "(1.0f / " + expr + ")";
    }

    private String floatUnaryExpr(DataType type, String fnBase, String expr) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            return renderCastExpression(
                    DataType.FP32,
                    type,
                    float32MathFunctionName(fnBase)
                            + "("
                            + renderFloat32ConversionExpression(type, expr)
                            + ")");
        }
        if (type == DataType.FP64) {
            return float64MathFunctionName(fnBase) + "(" + expr + ")";
        }
        if (type == DataType.FP32) {
            return float32MathFunctionName(fnBase) + "(" + expr + ")";
        }
        throw new UnsupportedOperationException("Floating-point unary op requires FP types");
    }

    private String binaryArithmetic(DataType type, String op, String left, String right) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            String l = renderFloat32ConversionExpression(type, left);
            String r = renderFloat32ConversionExpression(type, right);
            return renderCastExpression(DataType.FP32, type, "(" + l + " " + op + " " + r + ")");
        }
        return "(" + left + " " + op + " " + right + ")";
    }

    private String minExpr(DataType type, String left, String right) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            String l = renderFloat32ConversionExpression(type, left);
            String r = renderFloat32ConversionExpression(type, right);
            return renderCastExpression(
                    DataType.FP32,
                    type,
                    float32MathFunctionName("fmin") + "(" + l + ", " + r + ")");
        }
        if (type == DataType.FP64) {
            return float64MathFunctionName("fmin") + "(" + left + ", " + right + ")";
        }
        if (type == DataType.FP32) {
            return float32MathFunctionName("fmin") + "(" + left + ", " + right + ")";
        }
        return "(" + left + " < " + right + " ? " + left + " : " + right + ")";
    }

    private String maxExpr(DataType type, String left, String right) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            String l = renderFloat32ConversionExpression(type, left);
            String r = renderFloat32ConversionExpression(type, right);
            return renderCastExpression(
                    DataType.FP32,
                    type,
                    float32MathFunctionName("fmax") + "(" + l + ", " + r + ")");
        }
        if (type == DataType.FP64) {
            return float64MathFunctionName("fmax") + "(" + left + ", " + right + ")";
        }
        if (type == DataType.FP32) {
            return float32MathFunctionName("fmax") + "(" + left + ", " + right + ")";
        }
        return "(" + left + " > " + right + " ? " + left + " : " + right + ")";
    }

    private String powExpr(DataType type, String left, String right) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            String l = renderFloat32ConversionExpression(type, left);
            String r = renderFloat32ConversionExpression(type, right);
            return renderCastExpression(
                    DataType.FP32, type, float32MathFunctionName("pow") + "(" + l + ", " + r + ")");
        }
        if (type == DataType.FP64) {
            return float64MathFunctionName("pow") + "(" + left + ", " + right + ")";
        }
        if (type == DataType.FP32) {
            return float32MathFunctionName("pow") + "(" + left + ", " + right + ")";
        }
        throw new UnsupportedOperationException("pow requires floating-point type");
    }

    protected final String renderStandardCastExpression(
            DataType source, DataType target, String expr) {
        if (source == target) {
            return expr;
        }
        if (target == DataType.BOOL) {
            return "(" + expr + " != 0 ? 1 : 0)";
        }
        if (target == DataType.FP16 || target == DataType.BF16) {
            String asFp32 = renderFloat32ConversionExpression(source, expr);
            return "(" + renderDataTypeToken(target) + ")(" + asFp32 + ")";
        }
        if (target == DataType.FP32) {
            return renderFloat32ConversionExpression(source, expr);
        }
        if (target == DataType.FP64) {
            return renderFloat64ConversionExpression(source, expr);
        }
        if (source == DataType.FP16 || source == DataType.BF16) {
            return "("
                    + renderDataTypeToken(target)
                    + ")("
                    + renderFloat32ConversionExpression(source, expr)
                    + ")";
        }
        return "(" + renderDataTypeToken(target) + ")(" + expr + ")";
    }

    protected final String renderFloat16FamilyLiteralFromBits(long bits, DataType type) {
        if (type == DataType.FP16) {
            String value = CLikeExprSupport.formatFloatLiteral(Float.float16ToFloat((short) bits));
            return renderCastExpression(DataType.FP32, DataType.FP16, value);
        }
        if (type == DataType.BF16) {
            int bf16Bits = ((int) bits) & 0xFFFF;
            String value =
                    CLikeExprSupport.formatFloatLiteral(Float.intBitsToFloat(bf16Bits << 16));
            return renderCastExpression(DataType.FP32, DataType.BF16, value);
        }
        throw new IllegalArgumentException("Expected FP16/BF16, got: " + type);
    }

    protected final String renderLiteralFromTypeModel(
            CLikeDataTypeModel typeModel, long bits, DataType type) {
        if (type == DataType.FP16 || type == DataType.BF16) {
            return renderFloat16FamilyLiteralFromBits(bits, type);
        }
        return typeModel.renderLiteral(bits, type);
    }

    protected final String storeValueExpr(DataType targetType, String value, DataType valueType) {
        if (targetType == DataType.BOOL) {
            return CLikeScalarSupport.boolStoreValue(value, valueType);
        }
        return renderCastExpression(valueType, targetType, value);
    }

    protected final String byteOffsetPointer(
            String bytePointerType, String bufferName, String offsetExpr) {
        return "((" + bytePointerType + ")" + bufferName + " + " + offsetExpr + ")";
    }

    protected final String derefRead(String valuePointerType, String pointerExpr) {
        return "*(" + valuePointerType + ")" + pointerExpr;
    }

    protected final String derefWrite(
            String valuePointerType, String pointerExpr, String valueExpr) {
        return "*(" + valuePointerType + ")" + pointerExpr + " = " + valueExpr + ";";
    }

    protected final void addLine(String line) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < indentLevel; i++) {
            builder.append("  ");
        }
        builder.append(line).append("\n");
        lines.add(builder.toString());
    }

    private void indent() {
        indentLevel++;
    }

    private void outdent() {
        indentLevel--;
    }

    protected static final class BufferBinding {
        public final String name;
        public final boolean readOnly;

        private BufferBinding(String name, boolean readOnly) {
            this.name = name;
            this.readOnly = readOnly;
        }
    }
}
