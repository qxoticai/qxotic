package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.TextRenderUtils;
import java.util.HashMap;
import java.util.Map;

/**
 * Renders IR-L graphs as MLIR-style text representation.
 *
 * <p>Output format characteristics:
 *
 * <ul>
 *   <li>SSA-style variable naming with descriptive buffer names
 *   <li>Explicit data types on all operations
 *   <li>Inline index expressions
 *   <li>Full buffer metadata (shape, strides)
 *   <li>Structured loop notation: {@code for %i = 0 to 4 step 1}
 *   <li>Flat representation with intermediate variables
 * </ul>
 */
public class LIRTextRenderer implements LIRVisitor<String> {

    private final StringBuilder output;
    private final int indentSize;
    private int currentIndent;
    private int nextVarId;
    private final Map<ScalarExpr, String> exprToVar;
    private final Map<Integer, String> bufferIdToName;
    private int nextInputId;
    private int nextOutputId;

    /** Creates a new renderer with default settings. */
    public LIRTextRenderer() {
        this(2);
    }

    /** Creates a new renderer with specified indent size. */
    public LIRTextRenderer(int indentSize) {
        this.output = new StringBuilder();
        this.indentSize = indentSize;
        this.currentIndent = 0;
        this.nextVarId = 0;
        this.exprToVar = new HashMap<>();
        this.bufferIdToName = new HashMap<>();
        this.nextInputId = 0;
        this.nextOutputId = 0;
    }

    /** Renders an LIRGraph to MLIR-style text. */
    public String render(LIRGraph graph) {
        output.setLength(0);
        currentIndent = 0;
        nextVarId = 0;
        nextInputId = 0;
        nextOutputId = 0;
        exprToVar.clear();
        bufferIdToName.clear();

        appendLine("LIRGraph {");
        increaseIndent();

        // Render inputs and build name map
        appendLine("inputs: [");
        increaseIndent();
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            String suffix = (i < graph.inputs().size() - 1) ? "," : "";
            String name = "in" + nextInputId++;
            bufferIdToName.put(input.id(), name);
            String formatted =
                    switch (input) {
                        case BufferRef buf ->
                                TextRenderUtils.formatBuffer(
                                        "in", nextInputId - 1, buf.dataType(), buf.layout());
                        case ScalarInput scalar ->
                                "%in"
                                        + (nextInputId - 1)
                                        + ": scalar "
                                        + TextRenderUtils.formatDataType(scalar.dataType());
                    };
            appendLine(formatted + suffix);
        }
        decreaseIndent();
        appendLine("]");

        // Render outputs and build buffer name map
        appendLine("outputs: [");
        increaseIndent();
        for (int i = 0; i < graph.outputs().size(); i++) {
            BufferRef outputBuf = graph.outputs().get(i);
            String suffix = (i < graph.outputs().size() - 1) ? "," : "";
            String name = "out" + nextOutputId++;
            bufferIdToName.put(outputBuf.id(), name);
            appendLine(
                    TextRenderUtils.formatBuffer(
                                    "out",
                                    nextOutputId - 1,
                                    outputBuf.dataType(),
                                    outputBuf.layout())
                            + suffix);
        }
        decreaseIndent();
        appendLine("]");

        // Render body
        appendLine("body {");
        increaseIndent();
        graph.body().accept(this);
        decreaseIndent();
        appendLine("}");

        decreaseIndent();
        appendLine("}");

        return output.toString();
    }

    private void appendLine(String line) {
        appendIndent();
        output.append(line);
        output.append("\n");
    }

    private void append(String text) {
        output.append(text);
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

    private String allocateVar(ScalarExpr expr, String rendered) {
        String varName = "%" + nextVarId++;
        exprToVar.put(expr, varName);
        return varName + " = " + rendered;
    }

    private String getVar(ScalarExpr expr) {
        if (exprToVar.containsKey(expr)) {
            return exprToVar.get(expr);
        }
        return expr.accept(this);
    }

    /**
     * Renders the RHS of an expression (the operation without variable assignment). Used for
     * ScalarLet to avoid creating intermediate variables.
     */
    private String renderExprRhs(ScalarExpr expr) {
        return switch (expr) {
            case ScalarLiteral lit -> visitScalarLiteral(lit);
            case ScalarInput input -> visitScalarInput(input);
            case ScalarRef ref -> visitScalarRef(ref);
            case ScalarFromIndex sfi ->
                    "from_index " + sfi.dataType() + " " + sfi.index().accept(this);
            case ScalarLoad load ->
                    "load "
                            + load.dataType()
                            + " %"
                            + bufferName(load.buffer())
                            + "["
                            + load.offset().accept(this)
                            + "]";
            case ScalarCast cast -> "cast " + cast.targetType() + " " + getVar(cast.input());
            case ScalarUnary unary ->
                    TextRenderUtils.formatUnaryOp(unary.op())
                            + " "
                            + unary.dataType()
                            + " "
                            + getVar(unary.input());
            case ScalarBinary binary ->
                    TextRenderUtils.formatBinaryOp(binary.op())
                            + " "
                            + binary.dataType()
                            + " "
                            + getVar(binary.left())
                            + ", "
                            + getVar(binary.right());
            case ScalarTernary ternary ->
                    "select "
                            + ternary.dataType()
                            + " "
                            + getVar(ternary.condition())
                            + ", "
                            + getVar(ternary.trueValue())
                            + ", "
                            + getVar(ternary.falseValue());
        };
    }

    // ==================== Index Expressions ====================

    @Override
    public String visitIndexVar(IndexVar node) {
        return "%" + node.name();
    }

    @Override
    public String visitIndexConst(IndexConst node) {
        return String.valueOf(node.value());
    }

    @Override
    public String visitIndexBinary(IndexBinary node) {
        String left = node.left().accept(this);
        String right = node.right().accept(this);
        String op = formatIndexBinaryOp(node.op());

        // Always wrap IndexBinary children in parentheses to show exact tree structure
        if (node.left() instanceof IndexBinary) {
            left = "(" + left + ")";
        }
        if (node.right() instanceof IndexBinary) {
            right = "(" + right + ")";
        }

        return left + " " + op + " " + right;
    }

    private String formatIndexBinaryOp(IndexBinary.IndexBinaryOp op) {
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

    // ==================== Scalar Expressions ====================

    @Override
    public String visitScalarLiteral(ScalarLiteral node) {
        DataType dt = node.dataType();
        if (dt == DataType.FP32) {
            return node.asFloat() + "f";
        } else if (dt == DataType.FP64) {
            return node.asDouble() + "";
        } else if (dt == DataType.I32) {
            return String.valueOf(node.asInt());
        } else if (dt == DataType.I64) {
            return String.valueOf(node.asLong());
        } else if (dt == DataType.BOOL) {
            return node.asBool() ? "true" : "false";
        } else if (dt == DataType.FP16) {
            return TextRenderUtils.formatFloat16((short) node.rawBits());
        } else if (dt == DataType.BF16) {
            return TextRenderUtils.formatBFloat16((short) node.rawBits());
        } else {
            return "0x" + Long.toHexString(node.rawBits());
        }
    }

    @Override
    public String visitScalarUnary(ScalarUnary node) {
        String input = getVar(node.input());
        String op = TextRenderUtils.formatUnaryOp(node.op());
        DataType dt = node.dataType();
        String result = allocateVar(node, op + " " + dt + " " + input);
        appendLine(result);
        return exprToVar.get(node);
    }

    @Override
    public String visitScalarBinary(ScalarBinary node) {
        String left = getVar(node.left());
        String right = getVar(node.right());
        String op = TextRenderUtils.formatBinaryOp(node.op());
        DataType dt = node.dataType();
        String result = allocateVar(node, op + " " + dt + " " + left + ", " + right);
        appendLine(result);
        return exprToVar.get(node);
    }

    @Override
    public String visitScalarTernary(ScalarTernary node) {
        String cond = getVar(node.condition());
        String trueVal = getVar(node.trueValue());
        String falseVal = getVar(node.falseValue());
        DataType dt = node.dataType();
        String result =
                allocateVar(node, "select " + dt + " " + cond + ", " + trueVal + ", " + falseVal);
        appendLine(result);
        return exprToVar.get(node);
    }

    @Override
    public String visitScalarCast(ScalarCast node) {
        String input = getVar(node.input());
        DataType target = node.targetType();
        String result = allocateVar(node, "cast " + target + " " + input);
        appendLine(result);
        return exprToVar.get(node);
    }

    @Override
    public String visitScalarLoad(ScalarLoad node) {
        String offset = node.offset().accept(this);
        BufferRef buffer = node.buffer();
        DataType dt = node.dataType();
        String result =
                allocateVar(node, "load " + dt + " %" + bufferName(buffer) + "[" + offset + "]");
        appendLine(result);
        return exprToVar.get(node);
    }

    @Override
    public String visitScalarInput(ScalarInput node) {
        // Scalar inputs are referenced directly by name (no load needed)
        return "%scalar" + node.id();
    }

    @Override
    public String visitScalarFromIndex(ScalarFromIndex node) {
        String index = node.index().accept(this);
        DataType dt = node.dataType();
        String result = allocateVar(node, "from_index " + dt + " " + index);
        appendLine(result);
        return exprToVar.get(node);
    }

    @Override
    public String visitScalarRef(ScalarRef node) {
        // Reference to a let-bound scalar
        return "%" + node.name();
    }

    @Override
    public String visitScalarLet(ScalarLet node) {
        // Render as SSA definition with the let's name
        String rhs = renderExprRhs(node.value());
        String varName = "%" + node.name();
        exprToVar.put(node.value(), varName); // Register so future refs use this name
        appendLine(varName + " = " + rhs);
        return "";
    }

    private String bufferName(BufferRef buffer) {
        // Look up the buffer name from the map, or use a generic name
        return bufferIdToName.getOrDefault(buffer.id(), "buffer" + buffer.id());
    }

    // ==================== Memory Operations ====================

    @Override
    public String visitBufferRef(BufferRef node) {
        return "%" + bufferName(node);
    }

    @Override
    public String visitLoad(Load node) {
        String offset = node.offset().accept(this);
        BufferRef buffer = node.buffer();
        DataType dt = node.dataType();
        appendLine("load " + dt + " %" + bufferName(buffer) + "[" + offset + "]");
        return "";
    }

    @Override
    public String visitStore(Store node) {
        String offset = node.offset().accept(this);
        BufferRef buffer = node.buffer();
        String value = getVar(node.value());
        appendLine("store %" + bufferName(buffer) + "[" + offset + "], " + value);
        return "";
    }

    // ==================== Control Flow ====================

    @Override
    public String visitLoop(Loop node) {
        String bound = node.bound().accept(this);
        String loopType = node.isParallel() ? "parallel.for" : "for";
        appendLine(loopType + " %" + node.indexName() + " in [0, " + bound + ") {");
        increaseIndent();
        node.body().accept(this);
        decreaseIndent();
        appendLine("}");
        return "";
    }

    @Override
    public String visitStructuredFor(StructuredFor node) {
        String lb = node.lowerBound().accept(this);
        String ub = node.upperBound().accept(this);
        String step = node.step().accept(this);

        StringBuilder header = new StringBuilder();
        header.append("for %")
                .append(node.indexName())
                .append(" = ")
                .append(lb)
                .append(" to ")
                .append(ub)
                .append(" step ")
                .append(step);

        if (!node.iterArgs().isEmpty()) {
            header.append(" iter_args(");
            for (int i = 0; i < node.iterArgs().size(); i++) {
                LoopIterArg arg = node.iterArgs().get(i);
                if (i > 0) {
                    header.append(", ");
                }
                header.append("%")
                        .append(arg.name())
                        .append(" = ")
                        .append(renderExprRhs(arg.init()));
            }
            header.append(") -> (");
            for (int i = 0; i < node.iterArgs().size(); i++) {
                if (i > 0) {
                    header.append(", ");
                }
                header.append(node.iterArgs().get(i).dataType());
            }
            header.append(")");
        }

        header.append(" {");
        appendLine(header.toString());
        increaseIndent();
        node.body().accept(this);
        decreaseIndent();
        appendLine("}");
        return "";
    }

    @Override
    public String visitTiledLoop(TiledLoop node) {
        String totalBound = node.totalBound().accept(this);
        appendLine(
                "tiled.for %"
                        + node.outerName()
                        + ", %"
                        + node.innerName()
                        + " in [0, "
                        + totalBound
                        + ") tile="
                        + node.tileSize()
                        + " {");
        increaseIndent();
        node.body().accept(this);
        decreaseIndent();
        appendLine("}");
        return "";
    }

    @Override
    public String visitLoopNest(LoopNest node) {
        appendLine("loop.nest {");
        increaseIndent();
        for (Loop loop : node.loops()) {
            String bound = loop.bound().accept(this);
            String loopType = loop.isParallel() ? "parallel.for" : "for";
            appendLine(loopType + " %" + loop.indexName() + " in [0, " + bound + ")");
        }
        node.body().accept(this);
        decreaseIndent();
        appendLine("}");
        return "";
    }

    @Override
    public String visitBlock(Block node) {
        for (LIRNode statement : node.statements()) {
            statement.accept(this);
        }
        return "";
    }

    @Override
    public String visitYield(Yield node) {
        if (node.values().isEmpty()) {
            appendLine("yield");
            return "";
        }
        StringBuilder line = new StringBuilder();
        line.append("yield ");
        for (int i = 0; i < node.values().size(); i++) {
            if (i > 0) {
                line.append(", ");
            }
            line.append(getVar(node.values().get(i)));
        }
        appendLine(line.toString());
        return "";
    }
}
