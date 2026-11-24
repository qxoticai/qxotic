package com.qxotic.jota.stablehlo.portable;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.bytecode.MlirBytecodeReader;
import com.qxotic.jota.stablehlo.bytecode.MlirBytecodeWriter;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.StructuralEquality;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.transform.CanonicalizeDefaults;
import com.qxotic.jota.stablehlo.vhlo.VhloException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/** Serialization: StableHLO → VHLO for a target version → MLIR bytecode, and back. */
class PortableArtifactSerializeTest {

    private static final Pattern VERSION_IN_NAME = Pattern.compile("(\\d+)_(\\d+)_(\\d+)\\.mlir$");

    static void assertSameProgram(Program expected, Program actual) {
        Optional<String> diff =
                StructuralEquality.diff(
                        CanonicalizeDefaults.apply(expected), CanonicalizeDefaults.apply(actual));
        if (diff.isPresent()) {
            fail(diff.get());
        }
    }

    static Stream<Path> sourcePrograms() {
        return Samples.files("portable", ".mlir").stream()
                .filter(p -> p.getFileName().toString().startsWith("stablehlo_legalize_to_vhlo."));
    }

    /** The programs StableHLO serialized for each version serialize again, for that version. */
    @ParameterizedTest(name = "{0}")
    @MethodSource("sourcePrograms")
    void serializesSourceProgramForItsVersion(Path file) throws IOException {
        Matcher m = VERSION_IN_NAME.matcher(file.getFileName().toString());
        assertTrue(m.find());
        StableHloVersion target =
                new StableHloVersion(
                        Integer.parseInt(m.group(1)),
                        Integer.parseInt(m.group(2)),
                        Integer.parseInt(m.group(3)));
        Program source = MlirParser.parse(file);
        byte[] artifact = PortableArtifact.serialize(source, target);

        BytecodeHeader header = BytecodeHeader.read(artifact);
        assertEquals("StableHLO_v" + target, header.producer());
        assertEquals(PortableArtifact.bytecodeVersion(target), header.bytecodeVersion());
        assertSameProgram(source, PortableArtifact.deserialize(artifact));
    }

    static Stream<Arguments> stablehloPrograms() {
        List<Arguments> out = new ArrayList<>();
        for (String dir : List.of("text/testdata", "text/syntax")) {
            for (Path file : Samples.files(dir, ".mlir")) {
                List<String> chunks = MlirParser.splitInputFile(Samples.read(file));
                for (int i = 0; i < chunks.size(); i++) {
                    if (!Samples.isNegativeOrEmpty(chunks.get(i))) {
                        out.add(
                                Arguments.of(
                                        dir + "/" + file.getFileName() + "#" + i, chunks.get(i)));
                    }
                }
            }
        }
        return out.stream();
    }

    /**
     * Every sample round-trips for the newest version and for 1.0.0, or is rejected with a {@link
     * VhloException} (other dialects, features newer than the target).
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("stablehloPrograms")
    void roundTripsOrIsRejectedCleanly(String name, String text) {
        Program program = MlirParser.parse(text, name);
        for (StableHloVersion target :
                List.of(StableHloVersion.CURRENT, new StableHloVersion(1, 0, 0))) {
            byte[] artifact;
            try {
                artifact = PortableArtifact.serialize(program, target);
            } catch (VhloException e) {
                continue; // not expressible in VHLO / at this version
            } catch (com.qxotic.jota.stablehlo.bytecode.BytecodeException e) {
                assertTrue(e.getMessage().contains("no VHLO encoding"), e.getMessage());
                continue;
            }
            assertSameProgram(program, PortableArtifact.deserialize(artifact));
        }
    }

    /** Re-encoding the raw VHLO of every bundled artifact gives back the same VHLO program. */
    @ParameterizedTest(name = "{0}")
    @MethodSource(
            "com.qxotic.jota.stablehlo.portable.PortableArtifactDeserializeTest#validArtifacts")
    void rewritesRawVhloLosslessly(Path file) throws IOException {
        MlirBytecodeReader.Result original = MlirBytecodeReader.read(Files.readAllBytes(file));
        byte[] rewritten =
                MlirBytecodeWriter.write(
                        original.program(), original.bytecodeVersion(), original.producer());
        MlirBytecodeReader.Result back = MlirBytecodeReader.read(rewritten);
        assertEquals(original.bytecodeVersion(), back.bytecodeVersion());
        assertEquals(original.producer(), back.producer());
        assertTrue(StructuralEquality.diff(original.program(), back.program()).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(strings = {"0.9.0", "0.11.0", "0.13.0", "0.14.0", "1.0.0", "1.27.0"})
    void everyBytecodeVersionRoundTrips(String version) {
        Program p =
                MlirParser.parse(
                        """
                        func.func @main(%a: tensor<2x3xf32>, %b: tensor<3xf32>) -> tensor<2x3xf32> {
                          %0 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
                          %1 = stablehlo.add %a, %0 : tensor<2x3xf32>
                          %c = stablehlo.constant dense<0.0> : tensor<f32>
                          %2 = stablehlo.reduce(%1 init: %c) applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
                          %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
                          %4 = stablehlo.compare GT, %1, %3, FLOAT : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
                          %5 = stablehlo.select %4, %1, %3 : tensor<2x3xi1>, tensor<2x3xf32>
                          return %5 : tensor<2x3xf32>
                        }
                        """);
        StableHloVersion target = StableHloVersion.parse(version);
        byte[] artifact = PortableArtifact.serialize(p, target);
        assertEquals(
                PortableArtifact.bytecodeVersion(target),
                BytecodeHeader.read(artifact).bytecodeVersion());
        assertSameProgram(p, PortableArtifact.deserialize(artifact));
    }

    @Test
    void bytecodeVersionFollowsStableHlo() {
        assertEquals(0, PortableArtifact.bytecodeVersion(StableHloVersion.parse("0.9.0")));
        assertEquals(1, PortableArtifact.bytecodeVersion(StableHloVersion.parse("0.11.0")));
        assertEquals(3, PortableArtifact.bytecodeVersion(StableHloVersion.parse("0.13.0")));
        assertEquals(4, PortableArtifact.bytecodeVersion(StableHloVersion.parse("0.14.0")));
        assertEquals(6, PortableArtifact.bytecodeVersion(StableHloVersion.parse("0.15.0")));
        assertEquals(6, PortableArtifact.bytecodeVersion(StableHloVersion.CURRENT));
    }

    @Test
    void rejectsFeaturesNewerThanTheTarget() {
        Program tan =
                MlirParser.parse(
                        "func.func @f(%x: tensor<f32>) -> tensor<f32> {\n"
                                + "  %y = stablehlo.tan %x : tensor<f32>\n"
                                + "  return %y : tensor<f32>\n}");
        VhloException e =
                org.junit.jupiter.api.Assertions.assertThrows(
                        VhloException.class,
                        () -> PortableArtifact.serialize(tan, new StableHloVersion(1, 0, 0)));
        assertTrue(e.getMessage().contains("not available"), e.getMessage());

        Program gather =
                MlirParser.parse(
                        """
                        func.func @f(%x: tensor<2x3xf32>, %i: tensor<2x1xi64>) -> tensor<2xf32> {
                          %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<2x3xf32>, tensor<2x1xi64>) -> tensor<2xf32>
                          return %g : tensor<2xf32>
                        }
                        """);
        // batching dims exist since gather_v2 (1.1.0)
        org.junit.jupiter.api.Assertions.assertThrows(
                VhloException.class,
                () -> PortableArtifact.serialize(gather, new StableHloVersion(1, 0, 0)));
        assertSameProgram(
                gather,
                PortableArtifact.deserialize(
                        PortableArtifact.serialize(gather, new StableHloVersion(1, 1, 0))));
    }

    /** WEAKORDER (StableHLO 1.22) serializes for 1.22 and is refused for older targets. */
    @org.junit.jupiter.api.Test
    void weakOrderNeedsVersion122() {
        com.qxotic.jota.stablehlo.ir.Program program =
                com.qxotic.jota.stablehlo.text.MlirParser.parse(
                        "func.func @main(%a: tensor<2xf32>) -> tensor<2xi1> {\n"
                                + "  %0 = stablehlo.compare LT, %a, %a, WEAKORDER : (tensor<2xf32>,"
                                + " tensor<2xf32>) -> tensor<2xi1>\n"
                                + "  return %0 : tensor<2xi1>\n"
                                + "}",
                        "weak");
        byte[] bytes = PortableArtifact.serialize(program, new StableHloVersion(1, 22, 0));
        org.junit.jupiter.api.Assertions.assertTrue(
                com.qxotic.jota.stablehlo.ir.StructuralEquality.diff(
                                program, PortableArtifact.deserialize(bytes))
                        .isEmpty());
        org.junit.jupiter.api.Assertions.assertThrows(
                com.qxotic.jota.stablehlo.vhlo.VhloException.class,
                () -> PortableArtifact.serialize(program, new StableHloVersion(1, 21, 0)));
    }
}
