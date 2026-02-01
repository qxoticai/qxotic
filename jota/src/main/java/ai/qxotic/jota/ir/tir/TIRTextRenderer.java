package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.TextRenderUtils;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Renders IR-T (Tensor IR) graphs as MLIR-style text representation.
 *
 * <p>Output format characteristics:
 *
 * <ul>
 *   <li>SSA-style variable naming with %0, %1, etc.
 *   <li>Explicit data types on all operations (FP32, I32, etc.)
 *   <li>Full layout information (shape:stride format)
 *   <li>Compact operation syntax: {@code %2 = add FP32 %0, %1}
 *   <li>Structured representation with inputs, outputs, and body sections
 * </ul>
 */
public class TIRTextRenderer implements TIRVisitor<String> {

    private final StringBuilder output;
    private final int indentSize;
    private int currentIndent;
    private int nextVarId;
    private int nextInputId;
    private int nextOutputId;
    private final Map<TIRNode, String> nodeToVar;
    private boolean inGraphRenderMode;

    /** Creates a new renderer with default settings. */
    public TIRTextRenderer() {
        this(2);
    }

    /** Creates a new renderer with specified indent size. */
    public TIRTextRenderer(int indentSize) {
        this.output = new StringBuilder();
        this.indentSize = indentSize;
        this.currentIndent = 0;
        this.nextVarId = 0;
        this.nodeToVar = new IdentityHashMap<>();
        this.inGraphRenderMode = false;
    }

    /** Renders a TIRGraph to MLIR-style text. */
    public String render(TIRGraph graph) {
        output.setLength(0);
        currentIndent = 0;
        nextVarId = 0;
        nextInputId = 0;
        nextOutputId = 0;
        nodeToVar.clear();
        inGraphRenderMode = true;

        appendLine("TIRGraph {");
        increaseIndent();

        appendLine("inputs: [");
        increaseIndent();
        for (int i = 0; i < graph.inputs().size(); i++) {
            TIRNode input = graph.inputs().get(i);
            if (!nodeToVar.containsKey(input)) {
                allocateInputVar(input);
            }
            String var = nodeToVar.get(input);
            String suffix = (i < graph.inputs().size() - 1) ? "," : "";
            appendLine(var + ": " + formatInput(input, i) + suffix);
        }
        decreaseIndent();
        appendLine("]");

        appendLine("outputs: [");
        increaseIndent();
        for (int i = 0; i < graph.outputs().size(); i++) {
            TIRNode outputNode = graph.outputs().get(i);
            if (!nodeToVar.containsKey(outputNode)) {
                allocateOutputVar(outputNode);
            }
            String var = nodeToVar.get(outputNode);
            String suffix = (i < graph.outputs().size() - 1) ? "," : "";
            appendLine(var + ": " + formatOutput(outputNode, i) + suffix);
        }
        decreaseIndent();
        appendLine("]");

        appendLine("body {");
        increaseIndent();
        for (TIRNode outputNode : graph.outputs()) {
            renderNodeForGraph(outputNode);
        }
        decreaseIndent();
        appendLine("}");

        decreaseIndent();
        appendLine("}");

        inGraphRenderMode = false;
        return output.toString();
    }

    private void renderNodeForGraph(TIRNode node) {
        if (nodeToVar.containsKey(node) && node instanceof TensorInput) {
            return;
        }

        if (!nodeToVar.containsKey(node)) {
            allocateVar(node);
        }

        if (node instanceof TensorInput) {
            renderInputNodeForGraph((TensorInput) node);
        } else if (node instanceof ScalarConstant) {
            renderScalarConstantForGraph((ScalarConstant) node);
        } else if (node instanceof IotaConstant) {
            renderIotaConstantForGraph((IotaConstant) node);
        } else if (node instanceof UnaryOp) {
            renderUnaryOpForGraph((UnaryOp) node);
        } else if (node instanceof BinaryOp) {
            renderBinaryOpForGraph((BinaryOp) node);
        } else if (node instanceof TernaryOp) {
            renderTernaryOpForGraph((TernaryOp) node);
        } else if (node instanceof CastOp) {
            renderCastOpForGraph((CastOp) node);
        } else if (node instanceof ReductionOp) {
            renderReductionOpForGraph((ReductionOp) node);
        } else if (node instanceof ViewTransform) {
            renderViewTransformForGraph((ViewTransform) node);
        } else if (node instanceof Contiguous) {
            renderContiguousForGraph((Contiguous) node);
        }
    }

    private void renderInputNodeForGraph(TensorInput node) {
        String var = nodeToVar.get(node);
        String inputDef =
                var
                        + " = input["
                        + node.id()
                        + "] "
                        + formatDataType(node.dataType())
                        + "["
                        + node.layout()
                        + "]";
        appendLine(inputDef);
    }

    private void renderScalarConstantForGraph(ScalarConstant node) {
        String var = nodeToVar.get(node);
        String value = TextRenderUtils.formatScalarValue(node.rawBits(), node.dataType());
        String typeLayout =
                ai.qxotic.jota.ir.TextRenderUtils.formatTypeLayout(
                        node.dataType(),
                        node.layout().shape().toArray(),
                        node.layout().stride().toArray(),
                        true);
        String scalarDef = var + " = " + value + " " + typeLayout;
        appendLine(scalarDef);
    }

    private void renderIotaConstantForGraph(IotaConstant node) {
        String var = nodeToVar.get(node);
        String iotaDef =
                var
                        + " = iota("
                        + node.count()
                        + ") "
                        + formatDataType(node.dataType())
                        + "["
                        + node.layout()
                        + "]";
        appendLine(iotaDef);
    }

    private void renderUnaryOpForGraph(UnaryOp node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        String input = nodeToVar.get(node.input());
        String op = TextRenderUtils.formatUnaryOp(node.op());
        String var = nodeToVar.get(node);
        String unaryDef = var + " = " + op + " " + formatDataType(node.dataType()) + " " + input;
        appendLine(unaryDef);
    }

    private void renderBinaryOpForGraph(BinaryOp node) {
        if (!nodeToVar.containsKey(node.left())) {
            renderNodeForGraph(node.left());
        }
        if (!nodeToVar.containsKey(node.right())) {
            renderNodeForGraph(node.right());
        }
        String left = nodeToVar.get(node.left());
        String right = nodeToVar.get(node.right());
        String op = TextRenderUtils.formatBinaryOp(node.op());
        String var = nodeToVar.get(node);
        String binaryDef =
                var
                        + " = "
                        + op
                        + " "
                        + formatDataType(node.dataType())
                        + " "
                        + left
                        + ", "
                        + right;
        appendLine(binaryDef);
    }

    private void renderTernaryOpForGraph(TernaryOp node) {
        if (!nodeToVar.containsKey(node.cond())) {
            renderNodeForGraph(node.cond());
        }
        if (!nodeToVar.containsKey(node.trueExpr())) {
            renderNodeForGraph(node.trueExpr());
        }
        if (!nodeToVar.containsKey(node.falseExpr())) {
            renderNodeForGraph(node.falseExpr());
        }
        String cond = nodeToVar.get(node.cond());
        String trueExpr = nodeToVar.get(node.trueExpr());
        String falseExpr = nodeToVar.get(node.falseExpr());
        String op = TextRenderUtils.formatTernaryOp(node.op());
        String var = nodeToVar.get(node);
        String ternaryDef =
                var
                        + " = "
                        + op
                        + " "
                        + formatDataType(node.dataType())
                        + " "
                        + cond
                        + ", "
                        + trueExpr
                        + ", "
                        + falseExpr;
        appendLine(ternaryDef);
    }

    private void renderCastOpForGraph(CastOp node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        String input = nodeToVar.get(node.input());
        String var = nodeToVar.get(node);
        String castDef = var + " = cast " + formatDataType(node.targetDataType()) + " " + input;
        appendLine(castDef);
    }

    private void renderReductionOpForGraph(ReductionOp node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        String input = nodeToVar.get(node.input());
        String op = TextRenderUtils.formatReductionOp(node.op());
        String var = nodeToVar.get(node);
        StringBuilder axesStr = new StringBuilder();
        for (int i = 0; i < node.axes().length; i++) {
            if (i > 0) {
                axesStr.append(",");
            }
            axesStr.append(node.axes()[i]);
        }
        String reductionDef =
                var
                        + " = "
                        + op
                        + " "
                        + formatDataType(node.dataType())
                        + " "
                        + input
                        + " axes=["
                        + axesStr
                        + "]"
                        + (" keepDims=" + node.keepDims());
        appendLine(reductionDef);
    }

    private void renderViewTransformForGraph(ViewTransform node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        String input = nodeToVar.get(node.input());
        String var = nodeToVar.get(node);
        String viewDef =
                var + " = " + node.hint() + " " + formatDataType(node.dataType()) + " " + input;
        appendLine(viewDef);
    }

    private void renderContiguousForGraph(Contiguous node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        String input = nodeToVar.get(node.input());
        String var = nodeToVar.get(node);
        String contiguousDef =
                var + " = contiguous " + formatDataType(node.dataType()) + " " + input;
        appendLine(contiguousDef);
    }

    /** Renders a single TIRNode to a string. */
    public String renderNode(TIRNode node) {
        output.setLength(0);
        currentIndent = 0;
        nextVarId = 0;
        nodeToVar.clear();
        inGraphRenderMode = false;
        node.accept(this);
        return output.toString().trim();
    }

    private String formatInput(TIRNode node, int id) {
        if (node instanceof TensorInput input) {
            return ai.qxotic.jota.ir.TextRenderUtils.formatTypeLayout(
                    input.dataType(),
                    input.layout().shape().toArray(),
                    input.layout().stride().toArray(),
                    true); // element-based for TIR
        }
        return formatNodeType(node);
    }

    private String formatOutput(TIRNode node, int id) {
        // Always show full layout with contiguity for outputs (for debugging)
        return ai.qxotic.jota.ir.TextRenderUtils.formatTypeLayout(
                node.dataType(),
                node.layout().shape().toArray(),
                node.layout().stride().toArray(),
                true); // element-based for TIR
    }

    private String formatNodeType(TIRNode node) {
        return formatDataType(node.dataType()) + "[" + node.layout() + "]";
    }

    private String formatDataType(DataType dataType) {
        return ai.qxotic.jota.ir.TextRenderUtils.formatDataType(dataType);
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

    private String allocateVar(TIRNode node) {
        String varName = "%" + nextVarId++;
        nodeToVar.put(node, varName);
        return varName;
    }

    private String allocateInputVar(TIRNode node) {
        String varName = "%in" + nextInputId++;
        nodeToVar.put(node, varName);
        return varName;
    }

    private String allocateOutputVar(TIRNode node) {
        String varName = "%out" + nextOutputId++;
        nodeToVar.put(node, varName);
        return varName;
    }

    private String getVar(TIRNode node) {
        if (nodeToVar.containsKey(node)) {
            return nodeToVar.get(node);
        }
        String result = node.accept(this);
        return result;
    }

    @Override
    public String visitTensorInput(TensorInput node) {
        String var = allocateVar(node);
        String inputDef =
                var
                        + " = input["
                        + node.id()
                        + "] "
                        + formatDataType(node.dataType())
                        + "["
                        + node.layout()
                        + "]";
        appendLine(inputDef);
        return var;
    }

    @Override
    public String visitScalarConstant(ScalarConstant node) {
        String var = allocateVar(node);
        String value = TextRenderUtils.formatScalarValue(node.rawBits(), node.dataType());
        String typeLayout =
                ai.qxotic.jota.ir.TextRenderUtils.formatTypeLayout(
                        node.dataType(),
                        node.layout().shape().toArray(),
                        node.layout().stride().toArray(),
                        true);
        String scalarDef = var + " = " + value + " " + typeLayout;
        appendLine(scalarDef);
        return var;
    }

    @Override
    public String visitIotaConstant(IotaConstant node) {
        String var = allocateVar(node);
        String value = TextRenderUtils.formatFloat16((short) node.layout().shape().toArray()[0]);
        String iotaDef =
                var
                        + " = iota("
                        + node.count()
                        + ") "
                        + formatDataType(node.dataType())
                        + "["
                        + node.layout()
                        + "]";
        appendLine(iotaDef);
        return var;
    }

    @Override
    public String visitUnaryOp(UnaryOp node) {
        String input = node.input().accept(this);
        String op = TextRenderUtils.formatUnaryOp(node.op());
        String var = allocateVar(node);
        String unaryDef = var + " = " + op + " " + formatDataType(node.dataType()) + " " + input;
        appendLine(unaryDef);
        return var;
    }

    @Override
    public String visitBinaryOp(BinaryOp node) {
        String left = node.left().accept(this);
        String right = node.right().accept(this);
        String op = TextRenderUtils.formatBinaryOp(node.op());
        String var = allocateVar(node);
        String binaryDef =
                var
                        + " = "
                        + op
                        + " "
                        + formatDataType(node.dataType())
                        + " "
                        + left
                        + ", "
                        + right;
        appendLine(binaryDef);
        return var;
    }

    @Override
    public String visitTernaryOp(TernaryOp node) {
        String cond = node.cond().accept(this);
        String trueExpr = node.trueExpr().accept(this);
        String falseExpr = node.falseExpr().accept(this);
        String op = TextRenderUtils.formatTernaryOp(node.op());
        String var = allocateVar(node);
        String ternaryDef =
                var
                        + " = "
                        + op
                        + " "
                        + formatDataType(node.dataType())
                        + " "
                        + cond
                        + ", "
                        + trueExpr
                        + ", "
                        + falseExpr;
        appendLine(ternaryDef);
        return var;
    }

    @Override
    public String visitCastOp(CastOp node) {
        String input = node.input().accept(this);
        String var = allocateVar(node);
        String castDef = var + " = cast " + formatDataType(node.targetDataType()) + " " + input;
        appendLine(castDef);
        return var;
    }

    @Override
    public String visitReductionOp(ReductionOp node) {
        String input = node.input().accept(this);
        String op = TextRenderUtils.formatReductionOp(node.op());
        String var = allocateVar(node);
        StringBuilder axesStr = new StringBuilder();
        for (int i = 0; i < node.axes().length; i++) {
            if (i > 0) {
                axesStr.append(",");
            }
            axesStr.append(node.axes()[i]);
        }
        String reductionDef =
                var
                        + " = "
                        + op
                        + " "
                        + formatDataType(node.dataType())
                        + " "
                        + input
                        + " axes=["
                        + axesStr
                        + "]"
                        + " keep_dims="
                        + node.keepDims();
        appendLine(reductionDef);
        return var;
    }

    @Override
    public String visitViewTransform(ViewTransform node) {
        String input = node.input().accept(this);
        String var = allocateVar(node);
        String viewDef =
                var + " = " + node.hint() + " " + formatDataType(node.dataType()) + " " + input;
        appendLine(viewDef);
        return var;
    }

    @Override
    public String visitContiguous(Contiguous node) {
        String input = node.input().accept(this);
        String var = allocateVar(node);
        String contiguousDef =
                var + " = contiguous " + formatDataType(node.dataType()) + " " + input;
        appendLine(contiguousDef);
        return var;
    }
}
