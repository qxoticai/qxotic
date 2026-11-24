package com.qxotic.jota.stablehlo.verify;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Return-type inference: shapes, bounds, element types, and its use by the verifier. */
class TypeInferenceTest {

    /** The inferred types of the first op named {@code name} in {@code text}. */
    private static List<InferredType> inferFirst(String text, String name) {
        Program program = MlirParser.parse(text, "test");
        List<List<InferredType>> found = new ArrayList<>();
        program.operation()
                .walk(
                        op -> {
                            if (found.isEmpty() && op.name().equals(name)) {
                                found.add(TypeInference.infer(op).orElseThrow());
                            }
                        });
        return found.getFirst();
    }

    private static String type(List<InferredType> inferred) {
        return inferred.getFirst().type().mlir();
    }

    @Test
    void elementwiseTakesTheMostSpecificShape() {
        String text =
                "func.func @f(%a: tensor<?x3xf32>, %b: tensor<2x?xf32>) -> tensor<?x?xf32> {\n"
                        + "  %0 = stablehlo.add %a, %b : (tensor<?x3xf32>, tensor<2x?xf32>) ->"
                        + " tensor<?x?xf32>\n"
                        + "  return %0 : tensor<?x?xf32>\n}";
        assertEquals("tensor<2x3xf32>", type(inferFirst(text, "stablehlo.add")));
        assertEquals(List.of(), Verifier.verify(MlirParser.parse(text, "test")));
    }

    @Test
    void boundsPropagateAndCombine() {
        String text =
                "func.func @f(%a: tensor<?xf32, #stablehlo.bounds<4>>, %b: tensor<?xf32,"
                        + " #stablehlo.bounds<2>>, %c: tensor<3xf32>) -> tensor<?xf32> {\n"
                        + "  %0 = stablehlo.add %a, %b : (tensor<?xf32, #stablehlo.bounds<4>>,"
                        + " tensor<?xf32, #stablehlo.bounds<2>>) -> tensor<?xf32>\n"
                        + "  %1 = stablehlo.concatenate %a, %c, dim = 0 : (tensor<?xf32,"
                        + " #stablehlo.bounds<4>>, tensor<3xf32>) -> tensor<?xf32>\n"
                        + "  return %1 : tensor<?xf32>\n}";
        // merging takes the smallest bound, concatenating adds a bound to a size
        assertEquals(
                "tensor<?xf32, #stablehlo.bounds<2>>", type(inferFirst(text, "stablehlo.add")));
        assertEquals(
                "tensor<?xf32, #stablehlo.bounds<7>>",
                type(inferFirst(text, "stablehlo.concatenate")));
    }

    @Test
    void longBoundsSpellingIsTheSameEncoding() {
        String text =
                "func.func @f(%a: tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>)"
                        + " -> tensor<?x?xf32> {\n"
                        + "  %0 = stablehlo.transpose %a, dims = [1, 0] : (tensor<?x?xf32,"
                        + " #stablehlo.type_extensions<bounds = [3, -1]>>) -> tensor<?x?xf32>\n"
                        + "  return %0 : tensor<?x?xf32>\n}";
        assertEquals(
                "tensor<?x?xf32, #stablehlo.bounds<?, 3>>",
                type(inferFirst(text, "stablehlo.transpose")));
    }

    @Test
    void verifierRejectsAResultBeyondTheInferredBound() {
        String text =
                "func.func @f(%a: tensor<?xf32, #stablehlo.bounds<4>>, %c: tensor<3xf32>) ->"
                        + " tensor<8xf32> {\n"
                        + "  %0 = stablehlo.concatenate %a, %c, dim = 0 : (tensor<?xf32,"
                        + " #stablehlo.bounds<4>>, tensor<3xf32>) -> tensor<8xf32>\n"
                        + "  return %0 : tensor<8xf32>\n}";
        List<Diagnostic> diagnostics = Verifier.verify(MlirParser.parse(text, "test"));
        assertEquals(1, diagnostics.size(), diagnostics.toString());
        assertEquals("infer", diagnostics.getFirst().constraint());
        assertEquals("stablehlo.concatenate", diagnostics.getFirst().operation().name());
    }

    @Test
    void branchesTakeTheLeastSpecificType() {
        String text =
                "func.func @f(%p: tensor<i1>, %a: tensor<2xf32>, %b: tensor<?xf32,"
                        + " #stablehlo.bounds<3>>) -> tensor<?xf32> {\n"
                        + "  %0 = \"stablehlo.if\"(%p) ({\n"
                        + "    stablehlo.return %a : tensor<2xf32>\n"
                        + "  }, {\n"
                        + "    stablehlo.return %b : tensor<?xf32, #stablehlo.bounds<3>>\n"
                        + "  }) : (tensor<i1>) -> tensor<?xf32>\n"
                        + "  return %0 : tensor<?xf32>\n}";
        assertEquals("tensor<?xf32, #stablehlo.bounds<3>>", type(inferFirst(text, "stablehlo.if")));
    }

    @Test
    void reductionsTakeTheElementTypeOfTheirBody() {
        String text =
                "func.func @f(%a: tensor<4x5xbf16>, %i: tensor<f32>) -> tensor<5xf32> {\n"
                        + "  %0 = stablehlo.reduce(%a init: %i) across dimensions = [0] :"
                        + " (tensor<4x5xbf16>, tensor<f32>) -> tensor<5xf32>\n"
                        + "   reducer(%x: tensor<f32>, %y: tensor<f32>) {\n"
                        + "    %s = stablehlo.add %x, %y : tensor<f32>\n"
                        + "    stablehlo.return %s : tensor<f32>\n"
                        + "  }\n"
                        + "  return %0 : tensor<5xf32>\n}";
        assertEquals("tensor<5xf32>", type(inferFirst(text, "stablehlo.reduce")));
    }

    @Test
    void contractionsInferOnlyTheShape() {
        String text =
                "func.func @f(%a: tensor<8x2x3xbf16>, %b: tensor<8x3x4xbf16>) -> tensor<8x2x4xf32>"
                        + " {\n"
                        + "  %0 = stablehlo.dot_general %a, %b, batching_dims = [0] x [0],"
                        + " contracting_dims = [2] x [1] : (tensor<8x2x3xbf16>, tensor<8x3x4xbf16>)"
                        + " -> tensor<8x2x4xf32>\n"
                        + "  return %0 : tensor<8x2x4xf32>\n}";
        InferredType t = inferFirst(text, "stablehlo.dot_general").getFirst();
        assertFalse(t.elementTypeInferred());
        assertTrue(t.isCompatibleWith(TensorType.of(FloatType.F32, 8, 2, 4)));
        assertFalse(t.isCompatibleWith(TensorType.of(FloatType.F32, 8, 4, 2)));
        assertEquals(List.of(), Verifier.verify(MlirParser.parse(text, "test")));
    }

    @Test
    void opsWithoutInferenceReturnEmpty() {
        Block b = new Block();
        Value a = b.addArgument(TensorType.of(FloatType.F32, 6), null);
        Operation reshape =
                Operation.create(
                        "stablehlo.reshape",
                        List.of(a),
                        List.of(TensorType.of(FloatType.F32, 2, 3)),
                        List.of(),
                        List.of(),
                        List.of());
        assertTrue(TypeInference.infer(reshape).isEmpty());
    }

    @Test
    void buildersGetTheirResultTypes() {
        Block b = new Block();
        Value a = b.addArgument(TensorType.of(FloatType.F32, 4, 6), null);
        List<Type> sliced =
                TypeInference.resultTypes(
                        "stablehlo.slice",
                        List.of(a),
                        List.of(
                                new NamedAttribute("start_indices", DenseArrayAttr.i64(1, 0)),
                                new NamedAttribute("limit_indices", DenseArrayAttr.i64(4, 6)),
                                new NamedAttribute("strides", DenseArrayAttr.i64(2, 4))),
                        List.of(),
                        List.of());
        assertEquals(List.of(TensorType.of(FloatType.F32, 2, 2)), sliced);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        TypeInference.resultTypes(
                                "stablehlo.convert", List.of(a), List.of(), List.of(), List.of()));
    }

    @Test
    void malformedOpsFailToInfer() {
        Block b = new Block();
        Value a = b.addArgument(TensorType.of(FloatType.F32, 4, 6), null);
        Operation transpose =
                Operation.create(
                        "stablehlo.transpose",
                        List.of(a),
                        List.of(TensorType.of(FloatType.F32, 6, 4)),
                        List.of(new NamedAttribute("permutation", DenseArrayAttr.i64(0, 0))),
                        List.of(),
                        List.of());
        assertThrows(
                TypeInference.TypeInferenceException.class, () -> TypeInference.infer(transpose));
    }
}
