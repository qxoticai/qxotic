package com.qxotic.jota.stablehlo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.StructuralEquality;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.transform.CanonicalizeDefaults;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class StableHloToolTest {

    private static final Path SAMPLE =
            Samples.root().resolve("text/testdata/add_any_int8_2_int8_2.mlir");

    private record Run(int status, String out, String err) {}

    private static Run run(String... args) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        int status = StableHloTool.run(args, new PrintStream(out), new PrintStream(err));
        return new Run(status, out.toString(), err.toString());
    }

    @Test
    void infoDescribesEachFormat(@TempDir Path dir) {
        Run text = run("info", SAMPLE.toString());
        assertEquals(0, text.status(), text.err());
        assertTrue(text.out().contains("format: MLIR text"), text.out());
        assertTrue(text.out().contains("function: @main"), text.out());
        assertTrue(text.out().contains(" stablehlo.add"), text.out());

        Path portable = dir.resolve("a.bc");
        assertEquals(
                0,
                run(
                                "convert",
                                SAMPLE.toString(),
                                "--to",
                                "portable",
                                "--target",
                                "1.0.0",
                                "-o",
                                portable.toString())
                        .status());
        assertTrue(
                run("info", portable.toString())
                        .out()
                        .contains("portable artifact (StableHLO 1.0.0"));

        Path plain = dir.resolve("a.mlirbc");
        assertEquals(
                0,
                run(
                                "convert",
                                SAMPLE.toString(),
                                "--to",
                                "bytecode",
                                "--target",
                                "1.25.3",
                                "-o",
                                plain.toString())
                        .status());
        String info = run("info", plain.toString()).out();
        assertTrue(
                info.contains("MLIR bytecode v6") && info.contains("stablehlo dialect 1.25.3"),
                info);
    }

    @Test
    void everyConversionReadsBackToTheSameProgram(@TempDir Path dir) throws Exception {
        Program source = CanonicalizeDefaults.apply(MlirParser.parse(SAMPLE));
        for (String[] conversion :
                new String[][] {
                    {"text"}, {"generic"}, {"portable"}, {"bytecode"}, {"bytecode", "4"}
                }) {
            Path out = dir.resolve(String.join("-", conversion));
            Run r =
                    conversion.length == 1
                            ? run(
                                    "convert",
                                    SAMPLE.toString(),
                                    "--to",
                                    conversion[0],
                                    "-o",
                                    out.toString())
                            : run(
                                    "convert",
                                    SAMPLE.toString(),
                                    "--to",
                                    conversion[0],
                                    "--bytecode-version",
                                    conversion[1],
                                    "-o",
                                    out.toString());
            assertEquals(0, r.status(), r.err());
            // and back to text through the tool, which detects the format
            Path back = dir.resolve(out.getFileName() + ".mlir");
            assertEquals(0, run("print", out.toString(), "-o", back.toString()).status());
            Program read = CanonicalizeDefaults.apply(MlirParser.parse(back));
            assertTrue(
                    StructuralEquality.diff(source, read).isEmpty(),
                    String.join(" ", conversion) + ": " + StructuralEquality.diff(source, read));
        }
    }

    @Test
    void printsPrettyOrGeneric() {
        assertTrue(run("print", SAMPLE.toString()).out().contains("stablehlo.add %"));
        assertTrue(
                run("print", SAMPLE.toString(), "--generic").out().contains("\"stablehlo.add\"("));
    }

    @Test
    void verifyReportsDiagnosticsWithTheirLine(@TempDir Path dir) throws Exception {
        Run ok = run("verify", SAMPLE.toString());
        assertEquals(0, ok.status(), ok.out());

        Path bad = dir.resolve("bad.mlir");
        Files.writeString(
                bad,
                """
                func.func @main(%a: tensor<2xf32>) -> tensor<3xf32> {
                  %0 = stablehlo.slice %a [0:3] : (tensor<2xf32>) -> tensor<3xf32>
                  return %0 : tensor<3xf32>
                }
                """);
        Run r = run("verify", bad.toString());
        assertEquals(1, r.status(), r.out());
        assertTrue(r.out().contains("bad.mlir:2: "), r.out());
    }

    @Test
    void convertsToTir() {
        Run r = run("to-tir", SAMPLE.toString());
        assertEquals(0, r.status(), r.err());
        assertTrue(r.out().contains("results: "), r.out());
    }

    @Test
    void runsAFunctionOnTheInterpreter() {
        Run r = run("run", SAMPLE.toString(), "--function", "expected");
        assertEquals(0, r.status(), r.err());
        assertTrue(r.out().startsWith("dense<"), r.out());
        assertEquals(0, run("run", SAMPLE.toString()).status()); // @main checks pass
    }

    @Test
    void usageErrorsExitWith2() {
        assertEquals(2, run().status());
        assertEquals(2, run("frobnicate", SAMPLE.toString()).status());
        assertEquals(2, run("print").status());
        assertEquals(2, run("print", SAMPLE.toString(), "--to", "text").status());
        assertEquals(2, run("convert", SAMPLE.toString(), "--to", "text").status());
        Run missing = run("print", "/nonexistent/file.mlir");
        assertEquals(1, missing.status());
    }
}
