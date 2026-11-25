package com.qxotic.jota.stablehlo.text;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.stablehlo.ir.Program;
import org.junit.jupiter.api.Test;

/** Value names and spellings of {@link MlirPrinter#print}, checked against MLIR's printer. */
class MlirPrinterTest {

    /**
     * MLIR's names: {@code %cst} / {@code %c} constants and the {@code _<n>} suffixes of one shared
     * conflict counter, nested regions numbered after their whole parent region (the reducer's
     * {@code %4}), {@code %iterArg} alike in both while regions, named ODS results; the expected
     * text is jaxlib 0.12.1's print of the same program.
     */
    @Test
    void namesValuesAsMlir() {
        Program p =
                MlirParser.parse(
                        """
                        func.func @main(%a: tensor<4xf32>, %s: tensor<2xui64>) -> (tensor<f32>, tensor<i32>) {
                          %0 = stablehlo.constant dense<1.0> : tensor<4xf32>
                          %1 = stablehlo.constant dense<0.0> : tensor<f32>
                          %2 = stablehlo.add %a, %0 : tensor<4xf32>
                          %3 = stablehlo.constant dense<1> : tensor<i32>
                          %4 = stablehlo.constant dense<2> : tensor<i32>
                          %5 = stablehlo.reduce(%2 init: %1) across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                            reducer(%x: tensor<f32>, %y: tensor<f32>) {
                              %c = stablehlo.constant dense<2.0> : tensor<f32>
                              %m = stablehlo.multiply %x, %c : tensor<f32>
                              %r = stablehlo.add %m, %y : tensor<f32>
                              stablehlo.return %r : tensor<f32>
                            }
                          %6:2 = stablehlo.while(%i = %3, %j = %4) : tensor<i32>, tensor<i32>
                            cond {
                              %k = stablehlo.constant dense<10> : tensor<i32>
                              %c = stablehlo.compare LT, %i, %k : (tensor<i32>, tensor<i32>) -> tensor<i1>
                              stablehlo.return %c : tensor<i1>
                            } do {
                              %k = stablehlo.add %i, %j : tensor<i32>
                              stablehlo.return %k, %j : tensor<i32>, tensor<i32>
                            }
                          %o, %p = stablehlo.rng_bit_generator %s, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
                          %7 = stablehlo.add %5, %1 : tensor<f32>
                          return %7, %6#0 : tensor<f32>, tensor<i32>
                        }
                        """,
                        "names.mlir");
        assertEquals(
                """
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<2xui64>) -> (tensor<f32>, tensor<i32>) {
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
                    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.add %arg0, %cst : tensor<4xf32>
                    %c = stablehlo.constant dense<1> : tensor<i32>
                    %c_1 = stablehlo.constant dense<2> : tensor<i32>
                    %1 = stablehlo.reduce(%0 init: %cst_0) across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
                      %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
                      %4 = stablehlo.multiply %arg2, %cst_2 : tensor<f32>
                      %5 = stablehlo.add %4, %arg3 : tensor<f32>
                      stablehlo.return %5 : tensor<f32>
                    }
                    %2:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %c_1) : tensor<i32>, tensor<i32>
                    cond {
                      %c_3 = stablehlo.constant dense<10> : tensor<i32>
                      %4 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
                      stablehlo.return %4 : tensor<i1>
                    } do {
                      %4 = stablehlo.add %iterArg, %iterArg_2 : tensor<i32>
                      stablehlo.return %4, %iterArg_2 : tensor<i32>, tensor<i32>
                    }
                    %output_state, %output = stablehlo.rng_bit_generator %arg1, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
                    %3 = stablehlo.add %1, %cst_0 : tensor<f32>
                    return %3, %2#0 : tensor<f32>, tensor<i32>
                  }
                }
                """,
                MlirPrinter.print(p));
    }
}
