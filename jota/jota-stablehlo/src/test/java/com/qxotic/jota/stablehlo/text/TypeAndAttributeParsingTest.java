package com.qxotic.jota.stablehlo.text;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.ComparisonDirection;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.FunctionType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.QuantizedType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.TupleType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class TypeAndAttributeParsingTest {

    @ParameterizedTest
    @ValueSource(
            strings = {
                "tensor<2x?x3xf32>",
                "tensor<f8E4M3FN>",
                "tensor<4xcomplex<f64>>",
                "tensor<3xui4>",
                "tensor<0xindex>",
                "tuple<tensor<i32>, !stablehlo.token>",
                "(tensor<f32>, tensor<f32>) -> tensor<f32>",
                "() -> (tensor<f32>, tensor<i1>)",
                "tensor<?x?xf32, #stablehlo.bounds<3, ?>>",
                "!stablehlo.future<tensor<f32>>"
            })
    void typesRoundTrip(String text) {
        Type t = MlirParser.parseType(text);
        assertEquals(text, t.mlir());
        assertEquals(t, MlirParser.parseType(t.mlir()));
    }

    @Test
    void tensorTypeStructure() {
        TensorType t = (TensorType) MlirParser.parseType("tensor<2x?x3xbf16>");
        assertEquals(List.of(2L, TensorType.DYNAMIC, 3L), t.shape());
        assertEquals(FloatType.BF16, t.elementType());
        assertTrue(!t.hasStaticShape());
        FunctionType f = (FunctionType) MlirParser.parseType("(tensor<f32>) -> tuple<>");
        assertEquals(new TupleType(List.of()), f.results().getFirst());
    }

    @Test
    void boundsSpellingsAreEquivalent() {
        assertEquals(
                MlirParser.parseType("tensor<?x?xf32, #stablehlo.bounds<3, ?>>"),
                MlirParser.parseType(
                        "tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, ?]>>"));
    }

    @Test
    void quantizedTypes() {
        TensorType t =
                (TensorType) MlirParser.parseType("tensor<2x!quant.uniform<i8:f32, 0.5:10>>");
        QuantizedType q = (QuantizedType) t.elementType();
        assertEquals(IntegerType.I8, q.storageType());
        assertEquals(List.of(0.5), q.scales());
        assertEquals(List.of(10L), q.zeroPoints());
        assertTrue(!q.isPerAxis());

        QuantizedType perAxis =
                (QuantizedType)
                        ((TensorType)
                                        MlirParser.parseType(
                                                "tensor<2x!quant.uniform<i8<-127:127>:f32:0,"
                                                        + " {0.1:-1, 0.2}>>"))
                                .elementType();
        assertEquals(0, perAxis.quantizationDimension());
        assertEquals(-127L, perAxis.storageMin());
        assertEquals(List.of(-1L, 0L), perAxis.zeroPoints());
        assertEquals(
                perAxis,
                ((TensorType) MlirParser.parseType("tensor<" + perAxis.mlir() + ">"))
                        .elementType());
    }

    private static DenseElementsAttr dense(String text) {
        Attribute a = MlirParser.parseAttribute(text);
        DenseElementsAttr d = assertInstanceOf(DenseElementsAttr.class, a);
        assertEquals(d, MlirParser.parseAttribute(d.toString()), "printed form re-parses");
        return d;
    }

    @Test
    void denseLiterals() {
        DenseElementsAttr ints = dense("dense<[[1, -2], [3, 4]]> : tensor<2x2xi32>");
        assertEquals(-2, ints.getLong(1));
        assertEquals(4, ints.getLong(3));

        DenseElementsAttr splat = dense("dense<1.5> : tensor<3xf32>");
        assertTrue(splat.isSplat());
        assertEquals(1.5, splat.getDouble(2));

        DenseElementsAttr blob = dense("dense<\"0x0000803F00000040\"> : tensor<2xf32>");
        assertEquals(1.0, blob.getDouble(0));
        assertEquals(2.0, blob.getDouble(1));

        DenseElementsAttr complex = dense("dense<(1.0, -2.0)> : tensor<complex<f32>>");
        assertInstanceOf(ComplexType.class, complex.type().elementType());
        assertEquals(-2.0, FloatFormat.F32.decode(complex.complexPartBits(0, 1)));

        DenseElementsAttr bools = dense("dense<[true, false]> : tensor<2xi1>");
        assertTrue(bools.getBoolean(0));
        assertTrue(!bools.getBoolean(1));

        assertEquals(255, dense("dense<255> : tensor<ui8>").getLong(0));
        assertEquals(-1, dense("dense<0xFFFF> : tensor<i16>").getLong(0));

        DenseElementsAttr nan = dense("dense<0x7FC00000> : tensor<f32>");
        assertTrue(Double.isNaN(nan.getDouble(0)));

        DenseElementsAttr fp8 = dense("dense<[1.0, 448.0]> : tensor<2xf8E4M3FN>");
        assertEquals(448.0, fp8.getDouble(1));

        DenseElementsAttr empty = dense("dense<> : tensor<0xf32>");
        assertEquals(0, empty.numElements());
    }

    @Test
    void largeDenseLiteralsPrintAsExactHexBlobs() {
        // MLIR's default elementsAttrHexElementLimit: up to 100 elements print as a list
        DenseElementsAttr small = dense(denseF64(100));
        assertTrue(small.toString().startsWith("dense<[0.000000e+00, 1.000000e-01, "));
        assertTrue(small.toString().contains(", 0.30000000000000004, "));
        DenseElementsAttr d = dense(denseF64(101));
        assertTrue(d.toString().startsWith("dense<\"0x"));
        assertEquals(3.9000000000000004, d.getDouble(39));
        assertEquals(d, dense(d.toString()));
        assertEquals("dense<> : tensor<2x0xf32>", dense("dense<> : tensor<2x0xf32>").toString());
    }

    private static String denseF64(int n) {
        StringBuilder sb = new StringBuilder("dense<[");
        for (int i = 0; i < n; i++) {
            sb.append(i > 0 ? ", " : "").append(i * 0.1);
        }
        return sb.append("]> : tensor<").append(n).append("xf64>").toString();
    }

    /** Float literals are spelled as MLIR's {@code printFloatValue} spells them. */
    @Test
    void floatLiteralsUseMlirSpelling() {
        assertEquals(
                "dense<-1.000000e+00> : tensor<bf16>",
                dense("dense<-1.0> : tensor<bf16>").toString());
        assertEquals(
                "dense<0.000000e+00> : tensor<f32>", dense("dense<0.0> : tensor<f32>").toString());
        assertEquals(
                "dense<-0.000000e+00> : tensor<f32>",
                dense("dense<-0.0> : tensor<f32>").toString());
        assertEquals(
                "dense<0.333333343> : tensor<f32>",
                dense("dense<0.333333343> : tensor<f32>").toString());
        assertEquals(
                "dense<0.33333333333333331> : tensor<f64>",
                dense("dense<0.3333333333333333> : tensor<f64>").toString());
        assertEquals(
                "dense<1.000000e+20> : tensor<f32>",
                dense("dense<1.0e20> : tensor<f32>").toString());
        assertEquals(
                "dense<3.40282347E+38> : tensor<f32>",
                dense("dense<0x7F7FFFFF> : tensor<f32>").toString());
        assertEquals(
                "dense<0x7FC00000> : tensor<f32>",
                dense("dense<0x7FC00000> : tensor<f32>").toString());
        assertEquals(
                "dense<0xFC00> : tensor<f16>", dense("dense<0xFC00> : tensor<f16>").toString());
        assertEquals(
                "dense<(1.000000e+00,-2.500000e-01)> : tensor<complex<f32>>",
                dense("dense<(1.0, -0.25)> : tensor<complex<f32>>").toString());
    }

    /** Programs MLIR's parser rejects (stablehlo/tests), rejected here while parsing too. */
    @Test
    void rejectsWhatMlirRejectsWhileParsing() {
        for (String type :
                List.of(
                        "!stablehlo.future<f32>",
                        "!stablehlo.future<tensor<f32>, i32>",
                        "!stablehlo.foobar",
                        "tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3]>>")) {
            assertThrows(ParseException.class, () -> MlirParser.parseType(type), type);
        }
        for (String attr :
                List.of(
                        "#stablehlo.conv<[c, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>",
                        "#stablehlo.conv<[b, f, 1, 0, 1]x[o, 0, 1, i]->[f, b, 0, 1]>",
                        "#stablehlo.dot<foo = [0]>",
                        "#stablehlo.gather<offset_dims = [1], bar = 2>")) {
            assertThrows(ParseException.class, () -> MlirParser.parseAttribute(attr), attr);
        }
        assertThrows(
                ParseException.class,
                () ->
                        MlirParser.parse(
                                "func.func @f(%a: tensor<4x2xf32>, %i: tensor<f32>) ->"
                                        + " tensor<4xf32> {\n"
                                        + "  %0 = stablehlo.reduce(%a init: %i) applies"
                                        + " stablehlo.divide across dimensions = [1] :"
                                        + " (tensor<4x2xf32>, tensor<f32>) -> tensor<4xf32>\n"
                                        + "  return %0 : tensor<4xf32>\n}"));
        // the raw spelling of a printable layout reads as the compact one MLIR prints
        assertEquals(
                MlirParser.parseAttribute(
                        "#stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>"),
                MlirParser.parseAttribute(
                        "#stablehlo.conv<raw input_batch_dimension = 0, input_feature_dimension ="
                                + " 3, input_spatial_dimensions = [1, 2],"
                                + " kernel_input_feature_dimension = 2,"
                                + " kernel_output_feature_dimension = 3, kernel_spatial_dimensions"
                                + " = [0, 1], output_batch_dimension = 0, output_feature_dimension"
                                + " = 3, output_spatial_dimensions = [1, 2]>"));
    }

    @Test
    void denseLiteralErrors() {
        assertThrows(
                ParseException.class, () -> MlirParser.parseAttribute("dense<256> : tensor<ui8>"));
        assertThrows(
                ParseException.class,
                () -> MlirParser.parseAttribute("dense<[1, 2, 3]> : tensor<2xi32>"));
        assertThrows(
                ParseException.class, () -> MlirParser.parseAttribute("dense<true> : tensor<i32>"));
    }

    @Test
    void stableHloAttributes() {
        DenseArrayAttr dims = (DenseArrayAttr) MlirParser.parseAttribute("array<i64: 1, 0>");
        assertArrayEquals(new long[] {1, 0}, dims.values());

        assertEquals(
                ComparisonDirection.GT,
                ComparisonDirection.from(
                        MlirParser.parseAttribute("#stablehlo<comparison_direction GT>")));

        GatherDimensionNumbers g =
                GatherDimensionNumbers.from(
                        MlirParser.parseAttribute(
                                "#stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0],"
                                        + " start_index_map = [0], index_vector_dim = 1>"));
        assertArrayEquals(new long[] {1}, g.offsetDims());
        assertArrayEquals(new long[] {0}, g.startIndexMap());
        assertEquals(1, g.indexVectorDim());
        assertEquals(g.toAttribute(), MlirParser.parseAttribute(g.toAttribute().toString()));
    }

    @Test
    void convolutionDimensionNumbers() {
        ConvDimensionNumbers c =
                ConvDimensionNumbers.from(
                        MlirParser.parseAttribute(
                                "#stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>"));
        assertEquals(0, c.inputBatchDimension());
        assertEquals(3, c.inputFeatureDimension());
        assertArrayEquals(new long[] {1, 2}, c.inputSpatialDimensions());
        assertEquals(2, c.kernelInputFeatureDimension());
        assertEquals(3, c.kernelOutputFeatureDimension());
        assertEquals("[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]", c.compact());

        ConvDimensionNumbers raw =
                ConvDimensionNumbers.from(
                        MlirParser.parseAttribute(
                                "#stablehlo.conv<raw input_batch_dimension = 0,"
                                        + " input_feature_dimension = 3,"
                                        + " input_spatial_dimensions = [1, 2],"
                                        + " kernel_input_feature_dimension = 2,"
                                        + " kernel_output_feature_dimension = 3,"
                                        + " kernel_spatial_dimensions = [0, 1],"
                                        + " output_batch_dimension = 0,"
                                        + " output_feature_dimension = 3,"
                                        + " output_spatial_dimensions = [1, 2]>"));
        assertEquals(c.compact(), raw.compact());
    }

    @Test
    void errorsReportLineAndColumn() {
        String text =
                """
                func.func @f(%a: tensor<f32>) -> tensor<f32> {
                  %0 = stablehlo.add %a, %x : tensor<f32>
                  return %0 : tensor<f32>
                }
                """;
        ParseException e =
                assertThrows(ParseException.class, () -> MlirParser.parse(text, "f.mlir"));
        assertEquals(2, e.line());
        assertEquals(26, e.column());
        assertTrue(e.getMessage().contains("undefined value %x"), e.getMessage());
    }

    @Test
    void typeMismatchIsReported() {
        String text =
                """
                func.func @f(%a: tensor<f32>) -> tensor<f32> {
                  %0 = stablehlo.add %a, %a : tensor<f64>
                  return %0 : tensor<f64>
                }
                """;
        ParseException e = assertThrows(ParseException.class, () -> MlirParser.parse(text));
        assertTrue(e.getMessage().contains("operand #0 has type tensor<f32>"), e.getMessage());
    }
}
