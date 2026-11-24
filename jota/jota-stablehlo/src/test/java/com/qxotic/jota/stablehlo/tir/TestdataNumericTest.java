package com.qxotic.jota.stablehlo.tir;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryDomains;
import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.text.MlirParser;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The {@code stablehlo/testdata} samples embed their inputs ({@code @inputs}) and the values XLA
 * computed ({@code @expected}): every sample the converter supports must reproduce them on TIR.
 * Samples using unsupported constructs are skipped (reported as such).
 */
class TestdataNumericTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUp() {
        domain = MemoryDomains.of(MemoryAllocators.ofArena(Arena.ofAuto()));
    }

    static Stream<Path> samples() {
        return Samples.files("text/testdata", ".mlir").stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("samples")
    void reproducesExpectedValues(Path file) throws IOException {
        Program program = MlirParser.parse(file);
        Assumptions.assumeTrue(program.function("expected").isPresent(), "no @expected");
        String mismatch;
        try {
            mismatch = NumericComparison.check(program, domain);
        } catch (UnsupportedStableHloException e) {
            Assumptions.abort("unsupported: " + e.getMessage());
            return;
        }
        assertNull(mismatch, file.getFileName() + ": " + mismatch);
    }

    @Test
    void aMeaningfulShareOfSamplesIsSupported() throws IOException {
        int supported = 0;
        for (Path file : samples().toList()) {
            Program program = MlirParser.parse(file);
            if (program.function("expected").isEmpty()) {
                continue;
            }
            try {
                StableHloToTirConverter.convert(program, "main");
                supported++;
            } catch (UnsupportedStableHloException e) {
                // counted as unsupported
            }
        }
        assertTrue(supported >= 55, "only " + supported + " samples convert");
    }
}
