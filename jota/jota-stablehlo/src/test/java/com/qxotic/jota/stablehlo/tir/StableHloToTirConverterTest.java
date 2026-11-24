package com.qxotic.jota.stablehlo.tir;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryDomains;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.text.MlirParser;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Each case converts a small StableHLO function to TIR and runs it with the TIR interpreter. */
class StableHloToTirConverterTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUp() {
        domain = MemoryDomains.of(MemoryAllocators.ofArena(Arena.ofAuto()));
    }

    /** Runs {@code @main} (no arguments) and returns its first result. */
    private static DenseElementsAttr run(String body, String resultType) {
        String text = "func.func @main() -> " + resultType + " {\n" + body + "\n}";
        Program program = MlirParser.parse(text);
        TirConversion c = StableHloToTirConverter.convert(program, "main");
        var out = c.interpret(List.of(), domain);
        return TirConversion.toDense(out.getFirst(), c.resultTypes().getFirst());
    }

    private static long[] longs(DenseElementsAttr d) {
        long[] out = new long[(int) d.numElements()];
        for (int i = 0; i < out.length; i++) {
            out[i] = d.getLong(i);
        }
        return out;
    }

    private static double[] doubles(DenseElementsAttr d) {
        double[] out = new double[(int) d.numElements()];
        for (int i = 0; i < out.length; i++) {
            out[i] = d.getDouble(i);
        }
        return out;
    }

    @Test
    void elementwiseArithmeticWithFunctionArguments() {
        Program p =
                MlirParser.parse(
                        """
                        func.func @main(%a: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<3xf32> {
                          %0 = stablehlo.multiply %a, %b : tensor<3xf32>
                          %1 = stablehlo.add %0, %a : tensor<3xf32>
                          %2 = stablehlo.rsqrt %1 : tensor<3xf32>
                          return %2 : tensor<3xf32>
                        }
                        """);
        TirConversion c = StableHloToTirConverter.convert(p, "main");
        var a =
                TirConversion.materialize(
                        (DenseElementsAttr)
                                MlirParser.parseAttribute("dense<[1.0, 2.0, 3.0]> : tensor<3xf32>"),
                        domain);
        var b =
                TirConversion.materialize(
                        (DenseElementsAttr)
                                MlirParser.parseAttribute("dense<[3.0, 7.0, 5.0]> : tensor<3xf32>"),
                        domain);
        var out = c.interpret(List.of(a, b), domain);
        double[] got = doubles(TirConversion.toDense(out.getFirst(), c.resultTypes().getFirst()));
        assertArrayEquals(new double[] {0.5, 0.25, 1.0 / Math.sqrt(18)}, got, 1e-6);
        assertEquals(2, c.inputs().size());
    }

    @Test
    void comparisonsAreNaNCorrect() {
        DenseElementsAttr r =
                run(
                        """
                          %a = stablehlo.constant dense<[1.0, 0x7FC00000, 2.0]> : tensor<3xf32>
                          %b = stablehlo.constant dense<[1.0, 1.0, 1.0]> : tensor<3xf32>
                          %ge = stablehlo.compare GE, %a, %b : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
                          %ne = stablehlo.compare NE, %a, %b : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
                          %c = stablehlo.concatenate %ge, %ne, dim = 0 : (tensor<3xi1>, tensor<3xi1>) -> tensor<6xi1>
                          return %c : tensor<6xi1>
                        """,
                        "tensor<6xi1>");
        // GE: 1>=1, NaN>=1 is false, 2>=1 ; NE: 1!=1 false, NaN!=1 true, 2!=1 true
        assertArrayEquals(new long[] {1, 0, 1, 0, 1, 1}, longs(r));
    }

    @Test
    void shiftsOutOfRangeFollowStableHlo() {
        DenseElementsAttr r =
                run(
                        """
                          %x = stablehlo.constant dense<[-8, 1, 1, -8]> : tensor<4xi8>
                          %s = stablehlo.constant dense<[1, 9, -1, 8]> : tensor<4xi8>
                          %l = stablehlo.shift_left %x, %s : tensor<4xi8>
                          %r = stablehlo.shift_right_arithmetic %x, %s : tensor<4xi8>
                          %c = stablehlo.concatenate %l, %r, dim = 0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<8xi8>
                          return %c : tensor<8xi8>
                        """,
                        "tensor<8xi8>");
        // shift_left: -16, 0 (9 >= 8), 0 (negative), 0 ; arithmetic: -4, 0, 0, -1 (sign fill)
        assertArrayEquals(new long[] {-16, 0, 0, 0, -4, 0, 0, -1}, longs(r));
    }

    @Test
    void floatToBoolConvertTestsNonZero() {
        DenseElementsAttr r =
                run(
                        """
                          %x = stablehlo.constant dense<[0.0, 0.5, -0.25, 0x7FC00000]> : tensor<4xf32>
                          %b = stablehlo.convert %x : (tensor<4xf32>) -> tensor<4xi1>
                          return %b : tensor<4xi1>
                        """,
                        "tensor<4xi1>");
        assertArrayEquals(new long[] {0, 1, 1, 1}, longs(r));
    }

    @Test
    void broadcastInDimWithPermutedDimensions() {
        DenseElementsAttr r =
                run(
                        """
                          %x = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
                          %b = stablehlo.broadcast_in_dim %x, dims = [2, 0] : (tensor<2x3xi32>) -> tensor<3x2x2xi32>
                          return %b : tensor<3x2x2xi32>
                        """,
                        "tensor<3x2x2xi32>");
        // result[i, j, k] = x[k, i]
        assertArrayEquals(new long[] {1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6}, longs(r));
    }

    @Test
    void sliceReverseAndTranspose() {
        DenseElementsAttr r =
                run(
                        """
                          %x = stablehlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi64>
                          %s = stablehlo.slice %x [0:3:2, 1:4:2] : (tensor<3x4xi64>) -> tensor<2x2xi64>
                          %v = stablehlo.reverse %s, dims = [1] : tensor<2x2xi64>
                          %t = stablehlo.transpose %v, dims = [1, 0] : (tensor<2x2xi64>) -> tensor<2x2xi64>
                          return %t : tensor<2x2xi64>
                        """,
                        "tensor<2x2xi64>");
        // slice: [[1, 3], [9, 11]] ; reverse cols: [[3, 1], [11, 9]] ; transpose: [[3, 11], [1, 9]]
        assertArrayEquals(new long[] {3, 11, 1, 9}, longs(r));
    }

    @Test
    void padWithInteriorAndNegativeEdges() {
        DenseElementsAttr r =
                run(
                        """
                          %x = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
                          %p = stablehlo.constant dense<0> : tensor<i32>
                          %y = stablehlo.pad %x, %p, low = [-1], high = [2], interior = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<6xi32>
                          return %y : tensor<6xi32>
                        """,
                        "tensor<6xi32>");
        // interior: [1, 0, 2, 0, 3], drop 1 low, add 2 high: [0, 2, 0, 3, 0, 0]
        assertArrayEquals(new long[] {0, 2, 0, 3, 0, 0}, longs(r));
    }

    @Test
    void gatherRowsWithClampedIndices() {
        DenseElementsAttr r =
                run(
                        """
                          %x = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
                          %i = stablehlo.constant dense<[[2], [0], [7]]> : tensor<3x1xi32>
                          %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2>}> : (tensor<3x2xi32>, tensor<3x1xi32>) -> tensor<3x2xi32>
                          return %g : tensor<3x2xi32>
                        """,
                        "tensor<3x2xi32>");
        // index 7 is clamped to the last row
        assertArrayEquals(new long[] {5, 6, 1, 2, 5, 6}, longs(r));
    }

    @Test
    void dynamicSliceAndUpdateSlice() {
        DenseElementsAttr r =
                run(
                        """
                          %x = stablehlo.constant dense<[[0, 1, 2], [3, 4, 5], [6, 7, 8]]> : tensor<3x3xi32>
                          %u = stablehlo.constant dense<[[-1, -2]]> : tensor<1x2xi32>
                          %i = stablehlo.constant dense<2> : tensor<i64>
                          %j = stablehlo.constant dense<5> : tensor<i64>
                          %w = stablehlo.dynamic_update_slice %x, %u, %i, %j : (tensor<3x3xi32>, tensor<1x2xi32>, tensor<i64>, tensor<i64>) -> tensor<3x3xi32>
                          %s = stablehlo.dynamic_slice %w, %i, %j, sizes = [2, 2] : (tensor<3x3xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
                          return %s : tensor<2x2xi32>
                        """,
                        "tensor<2x2xi32>");
        // update at clamped (2, 1): row 2 = [6, -1, -2]; slice at clamped (1, 1): [[4, 5], [-1,
        // -2]]
        assertArrayEquals(new long[] {4, 5, -1, -2}, longs(r));
    }

    @Test
    void batchedDotGeneral() {
        DenseElementsAttr r =
                run(
                        """
                          %a = stablehlo.constant dense<[[[1.0, 2.0]], [[3.0, 4.0]]]> : tensor<2x1x2xf32>
                          %b = stablehlo.constant dense<[[[1.0], [1.0]], [[2.0], [-1.0]]]> : tensor<2x2x1xf32>
                          %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x1x2xf32>, tensor<2x2x1xf32>) -> tensor<2x1x1xf32>
                          return %d : tensor<2x1x1xf32>
                        """,
                        "tensor<2x1x1xf32>");
        assertArrayEquals(new double[] {3.0, 2.0}, doubles(r), 0.0);
    }

    @Test
    void reduceWithInitValue() {
        DenseElementsAttr r =
                run(
                        """
                          %x = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
                          %c = stablehlo.constant dense<10> : tensor<i32>
                          %s = stablehlo.reduce(%x init: %c) applies stablehlo.add across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
                          return %s : tensor<2xi32>
                        """,
                        "tensor<2xi32>");
        assertArrayEquals(new long[] {16, 25}, longs(r));
    }

    @Test
    void callsAreInlined() {
        Program p =
                MlirParser.parse(
                        """
                        func.func @main() -> tensor<2xi32> {
                          %c = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
                          %r = func.call @twice(%c) : (tensor<2xi32>) -> tensor<2xi32>
                          return %r : tensor<2xi32>
                        }
                        func.func private @twice(%x: tensor<2xi32>) -> tensor<2xi32> {
                          %y = stablehlo.add %x, %x : tensor<2xi32>
                          return %y : tensor<2xi32>
                        }
                        """);
        TirConversion c = StableHloToTirConverter.convert(p, "main");
        var out = c.interpret(List.of(), domain);
        assertArrayEquals(
                new long[] {2, 4},
                longs(TirConversion.toDense(out.getFirst(), c.resultTypes().getFirst())));
    }

    @Test
    void unsupportedConstructsAreReported() {
        Program p =
                MlirParser.parse(
                        """
                        func.func @main(%x: tensor<f32>) -> tensor<f32> {
                          %r = stablehlo.while(%i = %x) : tensor<f32>
                           cond {
                            %c = stablehlo.compare LT, %i, %i : (tensor<f32>, tensor<f32>) -> tensor<i1>
                            stablehlo.return %c : tensor<i1>
                          } do {
                            stablehlo.return %i : tensor<f32>
                          }
                          return %r : tensor<f32>
                        }
                        """);
        UnsupportedStableHloException e =
                assertThrows(
                        UnsupportedStableHloException.class,
                        () -> StableHloToTirConverter.convert(p, "main"));
        assertEquals("stablehlo.while", e.operation());

        Program unsigned =
                MlirParser.parse(
                        "func.func @main(%x: tensor<2xui8>) -> tensor<2xui8> {\n"
                                + "  return %x : tensor<2xui8>\n}");
        assertTrue(
                assertThrows(
                                UnsupportedStableHloException.class,
                                () -> StableHloToTirConverter.convert(unsigned, "main"))
                        .getMessage()
                        .contains("ui8"));
    }
}
