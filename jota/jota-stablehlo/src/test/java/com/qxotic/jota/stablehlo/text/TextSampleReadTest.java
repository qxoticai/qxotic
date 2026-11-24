package com.qxotic.jota.stablehlo.text;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.StructuralEquality;
import com.qxotic.jota.stablehlo.op.GenericOp;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Reads every bundled text sample (JAX exports from {@code stablehlo/testdata}, interpreter tests,
 * printer / round-trip test files), then checks that both forms printed by {@link MlirPrinter}
 * (pretty and generic) parse back to a structurally identical program.
 */
class TextSampleReadTest {

    static Stream<Arguments> textSamples() {
        List<Arguments> out = new ArrayList<>();
        for (String dir : List.of("text/testdata", "text/interpret", "text/syntax")) {
            for (Path file : Samples.files(dir, ".mlir")) {
                List<String> chunks = MlirParser.splitInputFile(Samples.read(file));
                for (int i = 0; i < chunks.size(); i++) {
                    String chunk = chunks.get(i);
                    if (Samples.isNegativeOrEmpty(chunk)) {
                        continue;
                    }
                    String name =
                            dir + "/" + file.getFileName() + (chunks.size() > 1 ? "#" + i : "");
                    out.add(Arguments.of(name, chunk));
                }
            }
        }
        return out.stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("textSamples")
    void readsAndRoundTripsThroughGenericForm(String name, String text) {
        // some samples are modules that only carry attributes: an empty body is valid
        Program program = MlirParser.parse(text, name);
        roundTrip(name, program, MlirPrinter.printGeneric(program), "generic");
        assertEquals(
                MlirPrinter.printGeneric(program),
                MlirPrinter.printGeneric(MlirParser.parse(MlirPrinter.printGeneric(program))));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("textSamples")
    void roundTripsThroughPrettyForm(String name, String text) {
        Program program = MlirParser.parse(text, name);
        String pretty = MlirPrinter.print(program);
        Program reparsed = roundTrip(name, program, pretty, "pretty");
        // printing is deterministic and a fixed point
        assertEquals(pretty, MlirPrinter.print(reparsed));
    }

    private static Program roundTrip(String name, Program program, String printed, String form) {
        Program reparsed;
        try {
            reparsed = MlirParser.parse(printed, form + ":" + name);
        } catch (ParseException e) {
            throw new AssertionError(form + " form does not parse:\n" + printed, e);
        }
        Optional<String> diff = StructuralEquality.diff(program, reparsed);
        if (diff.isPresent()) {
            fail(form + " round trip differs: " + diff.get() + "\n" + printed);
        }
        return reparsed;
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("textSamples")
    void everyStableHloOpHasATypedView(String name, String text) {
        Program program = MlirParser.parse(text, name);
        program.operation()
                .walk(
                        op -> {
                            if (op.dialect().equals("stablehlo")) {
                                assertFalse(
                                        op.asOp() instanceof GenericOp,
                                        "no typed record for " + op.name());
                            }
                        });
    }

    @Test
    void sampleSetIsNotEmpty() {
        long count = textSamples().count();
        assertTrue(count > 300, "expected the bundled sample set, found " + count + " chunks");
    }
}
