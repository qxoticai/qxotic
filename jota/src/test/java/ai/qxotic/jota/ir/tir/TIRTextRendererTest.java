package ai.qxotic.jota.ir.tir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.tensor.IRTracer;
import ai.qxotic.jota.tensor.LazyComputation;
import ai.qxotic.jota.tensor.Tensor;
import java.util.List;
import org.junit.jupiter.api.Test;

class TIRTextRendererTest {

    @Test
    void testUnaryOpRendering() {
        Tensor input = Tensor.of(new float[] {1.0f, 2.0f, 3.0f});
        Tensor result = IRTracer.trace(input, Tensor::negate);

        LazyComputation comp = result.computation().orElseThrow();
        Object graph = comp.attributes().get("graph");
        assertTrue(graph instanceof TIRGraph);

        TIRGraph tirGraph = (TIRGraph) graph;
        String rendered = new TIRTextRenderer().render(tirGraph);

        assertNotNull(rendered);
        assertTrue(rendered.contains("TIRGraph {"));
        assertTrue(rendered.contains("inputs: ["));
        assertTrue(rendered.contains("outputs: ["));
        assertTrue(rendered.contains("body {"));
        assertTrue(rendered.contains("negate fp32"));
    }

    @Test
    void testBinaryOpRendering() {
        Tensor input1 = Tensor.of(new float[] {1.0f, 2.0f, 3.0f});
        Tensor input2 = Tensor.of(new float[] {4.0f, 5.0f, 6.0f});
        Tensor result =
                IRTracer.trace(
                        List.of(input1, input2), tensors -> tensors.get(0).add(tensors.get(1)));

        LazyComputation comp = result.computation().orElseThrow();
        Object graph = comp.attributes().get("graph");
        assertTrue(graph instanceof TIRGraph);

        TIRGraph tirGraph = (TIRGraph) graph;
        String rendered = new TIRTextRenderer().render(tirGraph);

        assertNotNull(rendered);
        assertTrue(rendered.contains("TIRGraph {"));
        assertTrue(rendered.contains("inputs: ["));
        assertTrue(rendered.contains("outputs: ["));
        assertTrue(rendered.contains("add fp32"));
    }

    @Test
    void testTIRGraphToString() {
        Tensor input = Tensor.of(new float[] {1.0f, 2.0f, 3.0f});
        Tensor result = IRTracer.trace(input, Tensor::negate);

        LazyComputation comp = result.computation().orElseThrow();
        Object graph = comp.attributes().get("graph");
        assertTrue(graph instanceof TIRGraph);

        TIRGraph tirGraph = (TIRGraph) graph;
        String toString = tirGraph.toString();

        assertNotNull(toString);
        assertTrue(toString.contains("TIRGraph {"));
        assertTrue(toString.contains("negate fp32"));
    }

    @Test
    void testScalarConstantRendering() {
        Tensor scalar = Tensor.full(10.0f, DataType.FP32, Shape.of(3));
        Tensor result = IRTracer.trace(scalar, t -> t.add(5.0f));

        LazyComputation comp = result.computation().orElseThrow();
        Object graph = comp.attributes().get("graph");
        assertTrue(graph instanceof TIRGraph);

        TIRGraph tirGraph = (TIRGraph) graph;
        String rendered = new TIRTextRenderer().render(tirGraph);

        assertNotNull(rendered);
        assertTrue(rendered.contains("TIRGraph {"));
        assertTrue(rendered.contains("5.0f"));
    }

    @Test
    void testRenderNode() {
        TIRNode node =
                new UnaryOp(
                        UnaryOperator.EXP,
                        new TensorInput(0, DataType.FP32, Layout.rowMajor(Shape.of(3))));

        TIRTextRenderer renderer = new TIRTextRenderer();
        String rendered = renderer.renderNode(node);
        System.out.println("=== testRenderNode output ===");
        System.out.println(rendered);

        assertNotNull(rendered);
        assertTrue(rendered.contains("exp"));
        assertTrue(rendered.contains("fp32"));
    }

    @Test
    void testGeluRendering() {
        // Build GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float[] inputData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        Shape shape = Shape.of(inputData.length);

        TIRNode x = new TensorInput(0, DataType.FP32, Layout.rowMajor(shape));

        // Constants
        TIRNode c_0_044715 =
                ScalarConstant.broadcast(Float.floatToRawIntBits(0.044715f), DataType.FP32, shape);
        TIRNode c_sqrt_2_pi =
                ScalarConstant.broadcast(Float.floatToRawIntBits(0.7978846f), DataType.FP32, shape);
        TIRNode c_1 = ScalarConstant.broadcast(Float.floatToRawIntBits(1.0f), DataType.FP32, shape);
        TIRNode c_0_5 =
                ScalarConstant.broadcast(Float.floatToRawIntBits(0.5f), DataType.FP32, shape);

        // Build GELU expression tree
        TIRNode x_squared = new BinaryOp(BinaryOperator.MULTIPLY, x, x);
        TIRNode x_cubed = new BinaryOp(BinaryOperator.MULTIPLY, x_squared, x);
        TIRNode scaled_cubic = new BinaryOp(BinaryOperator.MULTIPLY, c_0_044715, x_cubed);
        TIRNode inner_sum = new BinaryOp(BinaryOperator.ADD, x, scaled_cubic);
        TIRNode scaled_inner = new BinaryOp(BinaryOperator.MULTIPLY, c_sqrt_2_pi, inner_sum);
        TIRNode tanh_result = new UnaryOp(UnaryOperator.TANH, scaled_inner);
        TIRNode one_plus_tanh = new BinaryOp(BinaryOperator.ADD, c_1, tanh_result);
        TIRNode x_times_bracket = new BinaryOp(BinaryOperator.MULTIPLY, x, one_plus_tanh);
        TIRNode gelu = new BinaryOp(BinaryOperator.MULTIPLY, c_0_5, x_times_bracket);

        TIRGraph graph = new TIRGraph(List.of(x), List.of(gelu));

        // Render
        TIRTextRenderer renderer = new TIRTextRenderer();
        String rendered = renderer.render(graph);

        // Print for debugging
        System.out.println("=== GELU TIR Output ===");
        System.out.println(rendered);
        System.out.println("=======================");

        // Verify key elements are present
        assertTrue(rendered.contains("TIRGraph {"), "Should contain TIRGraph header");
        assertTrue(rendered.contains("inputs:"), "Should contain inputs section");
        assertTrue(rendered.contains("outputs:"), "Should contain outputs section");
        assertTrue(rendered.contains("body {"), "Should contain body section");
        assertTrue(rendered.contains("multiply fp32"), "Should contain multiply operations");
        assertTrue(rendered.contains("add fp32"), "Should contain add operations");
        assertTrue(rendered.contains("tanh fp32"), "Should contain tanh operation");
        assertTrue(rendered.contains("0.044715f"), "Should contain cubic coefficient");
        assertTrue(rendered.contains("0.7978846f"), "Should contain sqrt(2/pi) constant");
        assertTrue(rendered.contains("0.5f"), "Should contain 0.5 constant");
    }
}
