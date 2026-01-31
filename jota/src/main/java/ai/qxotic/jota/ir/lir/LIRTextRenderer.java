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
 *   <li>Compact loop notation: {@code parallel.for %i in [0, 4)}
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
                                "in"
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

    // ==================== Accumulators ====================

    @Override
    public String visitAccumulator(Accumulator node) {
        String identity = formatIdentityValue(node.identityBits(), node.dataType());
        appendLine(
                "accumulator %"
                        + node.name()
                        + ": "
                        + node.dataType()
                        + " = "
                        + identity
                        + " # "
                        + node.op());
        return "";
    }

    private String formatIdentityValue(long bits, DataType dt) {
        if (dt == DataType.FP32) {
            return Float.intBitsToFloat((int) bits) + "f";
        } else if (dt == DataType.FP64) {
            return Double.longBitsToDouble(bits) + "";
        } else if (dt == DataType.I32) {
            return String.valueOf((int) bits);
        } else if (dt == DataType.I64) {
            return String.valueOf(bits);
        } else {
            return String.valueOf(bits);
        }
    }

    @Override
    public String visitAccumulatorRead(AccumulatorRead node) {
        String varName = "%" + nextVarId++;
        appendLine(varName + " = read.acc %" + node.name() + " : " + node.dataType());
        return varName;
    }

    @Override
    public String visitAccumulatorUpdate(AccumulatorUpdate node) {
        String value = getVar(node.value());
        appendLine("update.acc %" + node.name() + ", " + value);
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
}
