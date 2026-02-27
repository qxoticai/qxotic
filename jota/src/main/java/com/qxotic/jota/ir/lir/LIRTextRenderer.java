package com.qxotic.jota.ir.lir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.TextRenderUtils;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.ir.tir.UnaryOperator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;

/** Renders unified LIR graphs as MLIR-style text representation. */
public final class LIRTextRenderer {

    private final StringBuilder output = new StringBuilder();
    private final int indentSize;
    private int currentIndent;
    private int nextVarId;
    private LIRExprGraph exprGraph;

    private final Map<LIRExprNode, String> nodeToVar = new IdentityHashMap<>();
    private final Map<Integer, String> bufferIdToVar = new HashMap<>();
    private final Map<Integer, String> scalarIdToVar = new HashMap<>();

    public LIRTextRenderer() {
        this(2);
    }

    public LIRTextRenderer(int indentSize) {
        this.indentSize = indentSize;
    }

    public String render(LIRGraph graph) {
        output.setLength(0);
        currentIndent = 0;
        nextVarId = 0;
        exprGraph = graph.exprGraph();
        nodeToVar.clear();
        bufferIdToVar.clear();
        scalarIdToVar.clear();

        StringBuilder params = new StringBuilder();
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            String var = allocateVar();
            if (input instanceof BufferRef buf) {
                bufferIdToVar.put(buf.id(), var);
                params.append(var)
                        .append(": ")
                        .append(formatMemRefType(buf.dataType(), buf.layout()));
            } else if (input instanceof ScalarInput scalar) {
                scalarIdToVar.put(scalar.id(), var);
                params.append(var)
                        .append(": ")
                        .append(TextRenderUtils.formatDataType(scalar.dataType()));
            }
            if (i < graph.inputs().size() - 1 || !graph.outputs().isEmpty()) {
                params.append(", ");
            }
        }

        for (int i = 0; i < graph.outputs().size(); i++) {
            BufferRef outputBuf = graph.outputs().get(i);
            String var = allocateVar();
            bufferIdToVar.put(outputBuf.id(), var);
            params.append(var)
                    .append(": ")
                    .append(formatMemRefType(outputBuf.dataType(), outputBuf.layout()));
            if (i < graph.outputs().size() - 1) {
                params.append(", ");
            }
        }

        appendLine("module {");
        increaseIndent();
        appendLine("func.func @lir_ops(" + params + ") {");
        increaseIndent();
        renderNode(graph.body());
        decreaseIndent();
        appendLine("}");
        decreaseIndent();
        appendLine("}");

        return output.toString();
    }

    private void renderNode(LIRExprNode node) {
        switch (node) {
            case Block block -> {
                for (LIRExprNode stmt : block.statements()) {
                    renderNode(stmt);
                }
            }
            case Store store -> {
                String value = renderScalarValue(store.value());
                String offset = renderIndexValue(store.offset());
                BufferRef buffer = store.buffer();
                appendLine(
                        "memref.store "
                                + value
                                + ", "
                                + bufferVar(buffer)
                                + "["
                                + offset
                                + "] : "
                                + formatMemRefType(buffer.dataType(), buffer.layout()));
            }
            case StructuredFor loop -> {
                String lb = renderIndexValue(loop.lowerBound());
                String ub = renderIndexValue(loop.upperBound());
                String step = renderIndexValue(loop.step());

                StringBuilder header = new StringBuilder();
                header.append("scf.for %")
                        .append(loop.indexName())
                        .append(" = ")
                        .append(lb)
                        .append(" to ")
                        .append(ub)
                        .append(" step ")
                        .append(step);

                if (!loop.iterArgs().isEmpty()) {
                    header.append(" iter_args(");
                    for (int i = 0; i < loop.iterArgs().size(); i++) {
                        LoopIterArg arg = loop.iterArgs().get(i);
                        if (i > 0) {
                            header.append(", ");
                        }
                        String init = renderScalarValue(arg.init());
                        header.append("%").append(arg.name()).append(" = ").append(init);
                    }
                    header.append(") -> (");
                    for (int i = 0; i < loop.iterArgs().size(); i++) {
                        if (i > 0) {
                            header.append(", ");
                        }
                        header.append(
                                TextRenderUtils.formatDataType(loop.iterArgs().get(i).dataType()));
                    }
                    header.append(")");
                }

                header.append(" {");
                appendLine(header.toString());
                increaseIndent();
                renderNode(loop.body());
                decreaseIndent();
                appendLine("}");
            }
            case Yield yield -> {
                if (yield.values().isEmpty()) {
                    appendLine("scf.yield");
                    return;
                }
                StringBuilder line = new StringBuilder();
                line.append("scf.yield ");
                for (int i = 0; i < yield.values().size(); i++) {
                    if (i > 0) {
                        line.append(", ");
                    }
                    line.append(renderScalarValue(yield.values().get(i)));
                }
                appendLine(line.toString());
            }
            default -> throw new IllegalStateException("Unexpected node kind: " + node.kind());
        }
    }

    private String renderScalarValue(LIRExprNode node) {
        LIRExprNode resolved = exprGraph.resolve(node);
        String cached = nodeToVar.get(resolved);
        if (cached != null) {
            return cached;
        }

        return switch (resolved.kind()) {
            case S_INPUT -> {
                int id = ((SInput) resolved).inputId();
                String var = scalarIdToVar.get(id);
                if (var == null) {
                    var = allocateVar();
                    scalarIdToVar.put(id, var);
                }
                yield var;
            }
            case S_REF -> "%" + ((SRef) resolved).name();
            case S_CONST -> {
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                String value =
                        TextRenderUtils.formatScalarValue(
                                ((SConst) resolved).rawBits(), resolved.dataType());
                appendLine(
                        var
                                + " = arith.constant "
                                + value
                                + TextRenderUtils.formatOpTypeSuffix(resolved.dataType()));
                yield var;
            }
            case S_FROM_INDEX -> {
                String index = renderIndexValue(((SFromIndex) resolved).indexExpr());
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(
                        var
                                + " = arith.index_cast "
                                + index
                                + " : index to "
                                + TextRenderUtils.formatDataType(resolved.dataType()));
                yield var;
            }
            case S_LOAD -> {
                SLoad load = (SLoad) resolved;
                String offset = renderIndexValue(load.offset());
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(
                        var
                                + " = memref.load "
                                + bufferVar(load.buffer())
                                + "["
                                + offset
                                + "] : "
                                + formatMemRefType(load.dataType(), load.buffer().layout()));
                yield var;
            }
            case S_UNARY -> {
                SUnary unary = (SUnary) resolved;
                String input = renderScalarValue(unary.input());
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(
                        var
                                + " = "
                                + formatUnaryOp(unary.op())
                                + " "
                                + input
                                + TextRenderUtils.formatOpTypeSuffix(unary.dataType()));
                yield var;
            }
            case S_BINARY -> {
                SBinary binary = (SBinary) resolved;
                String left = renderScalarValue(binary.left());
                String right = renderScalarValue(binary.right());
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(
                        var
                                + " = "
                                + formatBinaryOp(binary.op())
                                + " "
                                + left
                                + ", "
                                + right
                                + TextRenderUtils.formatOpTypeSuffix(binary.dataType()));
                yield var;
            }
            case S_TERNARY -> {
                STernary ternary = (STernary) resolved;
                String cond = renderScalarValue(ternary.condition());
                String tVal = renderScalarValue(ternary.trueValue());
                String fVal = renderScalarValue(ternary.falseValue());
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(
                        var
                                + " = arith.select "
                                + cond
                                + ", "
                                + tVal
                                + ", "
                                + fVal
                                + TextRenderUtils.formatOpTypeSuffix(ternary.dataType()));
                yield var;
            }
            case S_CAST -> {
                SCast cast = (SCast) resolved;
                String input = renderScalarValue(cast.input());
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(
                        var
                                + " = "
                                + formatCastOp(cast.input().dataType(), cast.targetType())
                                + " "
                                + input
                                + TextRenderUtils.formatOpTypeSuffix(cast.targetType()));
                yield var;
            }
            default ->
                    throw new IllegalStateException("Expected scalar node, got " + resolved.kind());
        };
    }

    private String renderIndexValue(LIRExprNode node) {
        LIRExprNode resolved = exprGraph.resolve(node);
        String cached = nodeToVar.get(resolved);
        if (cached != null) {
            return cached;
        }

        return switch (resolved.kind()) {
            case I_VAR -> "%" + ((IVar) resolved).name();
            case I_CONST -> {
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(var + " = arith.constant " + ((IConst) resolved).value() + " : index");
                yield var;
            }
            case I_BINARY -> {
                IBinary binary = (IBinary) resolved;
                String left = renderIndexValue(binary.left());
                String right = renderIndexValue(binary.right());
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(
                        var
                                + " = "
                                + formatIndexOp(binary.op())
                                + " "
                                + left
                                + ", "
                                + right
                                + " : index");
                yield var;
            }
            case I_FROM_SCALAR -> {
                IFromScalar fromScalar = (IFromScalar) resolved;
                String scalar = renderScalarValue(fromScalar.scalarExpr());
                String var = allocateVar();
                nodeToVar.put(resolved, var);
                appendLine(var + " = arith.index_cast " + scalar + " : index");
                yield var;
            }
            default ->
                    throw new IllegalStateException("Expected index node, got " + resolved.kind());
        };
    }

    private String formatIndexOp(IndexBinaryOp op) {
        return switch (op) {
            case ADD -> "arith.addi";
            case SUBTRACT -> "arith.subi";
            case MULTIPLY -> "arith.muli";
            case DIVIDE -> "arith.divsi";
            case MODULO -> "arith.remsi";
            case BITWISE_AND -> "arith.andi";
            case BITWISE_XOR -> "arith.xori";
            case SHIFT_LEFT -> "arith.shli";
            case SHIFT_RIGHT -> "arith.shrsi";
            case UNSIGNED_SHIFT_RIGHT -> "arith.shrui";
        };
    }

    private String formatUnaryOp(UnaryOperator op) {
        return switch (op) {
            case NEGATE -> "arith.negf";
            case ABS -> "math.abs";
            case EXP -> "math.exp";
            case LOG -> "math.log";
            case SQRT -> "math.sqrt";
            case TANH -> "math.tanh";
            case RECIPROCAL -> "arith.divf 1.0";
            default -> "arith." + op.name().toLowerCase();
        };
    }

    private String formatBinaryOp(BinaryOperator op) {
        return switch (op) {
            case ADD -> "arith.addf";
            case SUBTRACT -> "arith.subf";
            case MULTIPLY -> "arith.mulf";
            case DIVIDE -> "arith.divf";
            case MIN -> "arith.minf";
            case MAX -> "arith.maxf";
            case POW -> "math.powf";
            default -> "arith." + op.name().toLowerCase();
        };
    }

    private String formatCastOp(DataType source, DataType target) {
        if (source.isFloatingPoint() && target.isIntegral()) {
            return "arith.fptosi";
        } else if (source.isIntegral() && target.isFloatingPoint()) {
            return "arith.sitofp";
        } else if (source.byteSize() == target.byteSize()) {
            return "arith.bitcast";
        }
        return "arith.extf";
    }

    private String formatMemRefType(DataType dataType, Layout layout) {
        int rank = (int) layout.shape().flatRank();
        long[] shape = new long[rank];
        long[] strides = new long[rank];
        long byteSize = dataType.byteSize();
        for (int i = 0; i < rank; i++) {
            shape[i] = layout.shape().flatAt(i);
            strides[i] = layout.stride().flatAt(i) * byteSize;
        }
        boolean contiguous = TextRenderUtils.isContiguous(shape, strides, (int) byteSize);
        return TextRenderUtils.formatMemRefType(
                dataType, shape, strides, (int) byteSize, contiguous);
    }

    private String bufferVar(BufferRef buffer) {
        return bufferIdToVar.getOrDefault(buffer.id(), "%buf" + buffer.id());
    }

    private String allocateVar() {
        return "%" + nextVarId++;
    }

    private void appendLine(String line) {
        appendIndent();
        output.append(line);
        output.append("\n");
    }

    private void appendIndent() {
        for (int i = 0; i < currentIndent * indentSize; i++) {
            output.append(" ");
        }
    }

    private void increaseIndent() {
        currentIndent++;
    }

    private void decreaseIndent() {
        currentIndent--;
    }
}
