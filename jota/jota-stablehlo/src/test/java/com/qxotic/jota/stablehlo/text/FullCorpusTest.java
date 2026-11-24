package com.qxotic.jota.stablehlo.text;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.StructuralEquality;
import com.qxotic.jota.stablehlo.portable.BytecodeHeader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Runs the readers over a whole local {@code openxla/stablehlo} checkout. Disabled unless {@code
 * -Dstablehlo.repo=/path/to/openxla/stablehlo} is given.
 */
@EnabledIfSystemProperty(named = "stablehlo.repo", matches = ".+")
class FullCorpusTest {

    private static Path repo() {
        return Path.of(System.getProperty("stablehlo.repo")).resolve("stablehlo");
    }

    @TestFactory
    Stream<DynamicTest> textCorpus() {
        List<Path> files = new ArrayList<>();
        files.addAll(Samples.list(repo().resolve("testdata"), ".mlir"));
        files.addAll(Samples.list(repo().resolve("tests/interpret"), ".mlir"));
        files.addAll(
                Samples.list(repo().resolve("tests/vhlo"), ".mlir").stream()
                        .filter(p -> p.getFileName().toString().startsWith("stablehlo_legalize"))
                        .filter(p -> !p.getFileName().toString().contains("mixed"))
                        .toList());
        assertTrue(files.size() > 2000, "corpus not found under " + repo());
        return files.stream()
                .map(
                        file ->
                                DynamicTest.dynamicTest(
                                        file.getFileName().toString(),
                                        () -> {
                                            for (String chunk :
                                                    MlirParser.splitInputFile(Samples.read(file))) {
                                                if (Samples.isNegativeOrEmpty(chunk)) {
                                                    continue;
                                                }
                                                Program p =
                                                        MlirParser.parse(chunk, file.toString());
                                                for (String printed :
                                                        List.of(
                                                                MlirPrinter.printGeneric(p),
                                                                MlirPrinter.print(p))) {
                                                    Program q =
                                                            MlirParser.parse(printed, "printed");
                                                    Optional<String> diff =
                                                            StructuralEquality.diff(p, q);
                                                    assertTrue(
                                                            diff.isEmpty(),
                                                            () -> diff.get() + "\n" + printed);
                                                }
                                            }
                                        }));
    }

    @TestFactory
    Stream<DynamicTest> portableArtifacts() {
        return Samples.list(repo().resolve("tests/vhlo"), ".mlir.bc").stream()
                .map(
                        file ->
                                DynamicTest.dynamicTest(
                                        file.getFileName().toString(),
                                        () ->
                                                assertTrue(
                                                        BytecodeHeader.read(
                                                                        Files.readAllBytes(file))
                                                                .isPortableArtifact())));
    }

    /** Every versioned artifact deserializes to the program it was serialized from. */
    @TestFactory
    Stream<DynamicTest> portableArtifactsEqualTheirSource() {
        return Samples.list(repo().resolve("tests/vhlo"), ".mlir.bc").stream()
                .filter(p -> p.getFileName().toString().startsWith("stablehlo_legalize_to_vhlo"))
                .map(
                        file ->
                                DynamicTest.dynamicTest(
                                        file.getFileName().toString(),
                                        () -> {
                                            Path source =
                                                    file.resolveSibling(
                                                            file.getFileName()
                                                                    .toString()
                                                                    .replace(".mlir.bc", ".mlir"));
                                            Program a =
                                                    com.qxotic.jota.stablehlo.transform
                                                            .CanonicalizeDefaults.apply(
                                                            com.qxotic.jota.stablehlo.portable
                                                                    .PortableArtifact.deserialize(
                                                                    file));
                                            Program b =
                                                    com.qxotic.jota.stablehlo.transform
                                                            .CanonicalizeDefaults.apply(
                                                            MlirParser.parse(source));
                                            Optional<String> diff = StructuralEquality.diff(a, b);
                                            assertTrue(diff.isEmpty(), () -> diff.get());
                                        }));
    }

    /** testdata programs supported by the TIR converter reproduce XLA's expected values. */
    @TestFactory
    Stream<DynamicTest> testdataOnTir() {
        var domain =
                com.qxotic.jota.memory.MemoryDomains.of(
                        com.qxotic.jota.memory.MemoryAllocators.ofArena(
                                java.lang.foreign.Arena.ofAuto()));
        return Samples.list(repo().resolve("testdata"), ".mlir").stream()
                .map(
                        file ->
                                DynamicTest.dynamicTest(
                                        file.getFileName().toString(),
                                        () -> {
                                            Program p = MlirParser.parse(file);
                                            org.junit.jupiter.api.Assumptions.assumeTrue(
                                                    p.function("expected").isPresent());
                                            String mismatch;
                                            try {
                                                mismatch =
                                                        com.qxotic.jota.stablehlo.tir
                                                                .NumericComparison.check(p, domain);
                                            } catch (
                                                    com.qxotic.jota.stablehlo.tir
                                                                    .UnsupportedStableHloException
                                                            e) {
                                                org.junit.jupiter.api.Assumptions.abort(
                                                        e.getMessage());
                                                return;
                                            }
                                            assertTrue(mismatch == null, mismatch);
                                        }));
    }
}
