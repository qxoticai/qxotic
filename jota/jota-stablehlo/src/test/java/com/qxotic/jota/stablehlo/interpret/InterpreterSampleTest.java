package com.qxotic.jota.stablehlo.interpret;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.text.MlirParser;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Runs the bundled samples on the {@link Interpreter}: each {@code testdata} program must reproduce
 * XLA's expected values (its {@code check.expect_*} calls pass, XLA semantics), and each of
 * StableHLO's interpreter tests must pass its checks (reference semantics). Programs using an
 * unsupported feature are skipped (reported as aborted).
 */
class InterpreterSampleTest {

    static Stream<Arguments> programs() {
        List<Arguments> out = new ArrayList<>();
        for (String dir : List.of("text/testdata", "text/interpret")) {
            for (Path file : Samples.files(dir, ".mlir")) {
                List<String> chunks = MlirParser.splitInputFile(Samples.read(file));
                for (int i = 0; i < chunks.size(); i++) {
                    String chunk = chunks.get(i);
                    if (!Samples.isNegativeOrEmpty(chunk) && chunk.contains("func.func")) {
                        out.add(Arguments.of(dir + "/" + file.getFileName() + "#" + i, chunk));
                    }
                }
            }
        }
        return out.stream();
    }

    static boolean runChecks(String name, String text) {
        return InterpreterSamples.runChecks(name, text);
    }

    /** Runs {@code @main}, or every function without arguments; false when unsupported. */
    static boolean runChecksImpl(String name, String text) {
        Program p = MlirParser.parse(text, name);
        Interpreter.Semantics semantics =
                name.startsWith("text/interpret")
                        ? Interpreter.Semantics.REFERENCE
                        : Interpreter.Semantics.XLA;
        List<String> functions = new ArrayList<>();
        if (p.function("main").isPresent()) {
            functions.add("main");
        } else {
            p.functions().stream()
                    .filter(f -> f.functionType().mlir().startsWith("() ->"))
                    .forEach(f -> functions.add(f.symName()));
        }
        try {
            for (String f : functions) {
                Interpreter.run(p, f, List.of(), semantics);
            }
            return true;
        } catch (InterpreterException e) {
            return false;
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("programs")
    void passesItsChecks(String name, String text) {
        Assumptions.assumeTrue(runChecks(name, text), "uses an unsupported feature");
    }

    @Test
    void coversMostSamples() {
        long testdata = 0;
        long interpret = 0;
        for (Arguments a : programs().toList()) {
            String name = (String) a.get()[0];
            if (runChecks(name, (String) a.get()[1])) {
                if (name.startsWith("text/testdata")) {
                    testdata++;
                } else {
                    interpret++;
                }
            }
        }
        // measured 2026-11-25: 119 of 123 testdata programs, 411 interpreter tests
        assertTrue(testdata >= 119, "testdata programs run: " + testdata);
        assertTrue(interpret >= 411, "interpreter tests run: " + interpret);
    }

    @Test
    void implementationDefinedResultsFollowTheChosenSemantics() {
        Program p =
                MlirParser.parse(
                        """
                        func.func @main() -> tensor<2xi32> {
                          %a = stablehlo.constant dense<[7, -2147483648]> : tensor<2xi32>
                          %b = stablehlo.constant dense<[0, -1]> : tensor<2xi32>
                          %0 = stablehlo.divide %a, %b : tensor<2xi32>
                          return %0 : tensor<2xi32>
                        }
                        """);
        Tensor xla = (Tensor) Interpreter.run(p, "main").getFirst();
        Tensor ref =
                (Tensor)
                        Interpreter.run(p, "main", List.of(), Interpreter.Semantics.REFERENCE)
                                .getFirst();
        assertEquals("dense<[-1, -2147483648]> : tensor<2xi32>", xla.toString());
        assertEquals("dense<0> : tensor<2xi32>", ref.toString());
    }

    @Test
    void failedChecksAndUnsupportedOpsAreReported() {
        Program wrong =
                MlirParser.parse(
                        """
                        func.func @main() {
                          %a = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
                          %0 = stablehlo.add %a, %a : tensor<2xf32>
                          check.expect_eq_const %0, dense<[2.0, 5.0]> : tensor<2xf32>
                          return
                        }
                        """);
        assertThrows(Interpreter.CheckFailure.class, () -> Interpreter.run(wrong, "main"));
        Program fft =
                MlirParser.parse(
                        """
                        func.func @main(%x: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
                          %0 = stablehlo.fft %x, type = FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
                          return %0 : tensor<4xcomplex<f32>>
                        }
                        """);
        assertThrows(
                InterpreterException.class,
                () ->
                        Interpreter.run(
                                fft,
                                "main",
                                List.of(
                                        Tensor.of(
                                                (com.qxotic.jota.stablehlo.attr.DenseElementsAttr)
                                                        MlirParser.parseAttribute(
                                                                "dense<(1.0,0.0)> :"
                                                                    + " tensor<4xcomplex<f32>>")))));
    }
}
