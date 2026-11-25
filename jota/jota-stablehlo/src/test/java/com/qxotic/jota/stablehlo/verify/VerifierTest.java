package com.qxotic.jota.stablehlo.verify;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.portable.PortableArtifact;
import com.qxotic.jota.stablehlo.text.MlirParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** The verifier on the bundled samples (no diagnostic) and on hand-written invalid programs. */
class VerifierTest {

    static Stream<Path> textSamples() throws Exception {
        try (Stream<Path> s = Files.walk(Samples.root().resolve("text"))) {
            return s.filter(p -> p.toString().endsWith(".mlir")).sorted().toList().stream();
        }
    }

    @ParameterizedTest
    @MethodSource("textSamples")
    void textSamplesVerifyClean(Path file) {
        List<String> diagnostics = new ArrayList<>();
        for (String chunk : MlirParser.splitInputFile(Samples.read(file))) {
            if (Samples.isNegativeOrEmpty(chunk)) {
                continue;
            }
            Verifier.verify(MlirParser.parse(chunk, file.toString()))
                    .forEach(d -> diagnostics.add(d.toString()));
        }
        assertEquals(List.of(), diagnostics);
    }

    static Stream<Path> portableSamples() {
        return Samples.files("portable", ".mlir.bc").stream()
                .filter(p -> !p.getFileName().toString().startsWith("invalid"));
    }

    @ParameterizedTest
    @MethodSource("portableSamples")
    void portableArtifactsVerifyClean(Path file) throws Exception {
        Program program = PortableArtifact.deserialize(Files.readAllBytes(file));
        assertEquals(List.of(), Verifier.verify(program));
    }

    static Stream<Arguments> invalid() {
        return Stream.of(
                Arguments.of(
                        "layout",
                        "stablehlo.infeed",
                        "func.func @f(%t: !stablehlo.token) -> tensor<3xi32> {\n"
                                + "  %0:2 = \"stablehlo.infeed\"(%t) {layout = [[0], [0]]} :"
                                + " (!stablehlo.token) -> (tensor<3xi32>, !stablehlo.token)\n"
                                + "  return %0#0 : tensor<3xi32>\n}"),
                Arguments.of(
                        "type",
                        "stablehlo.reshape",
                        "func.func @f(%a: tensor<?x?xf32, #stablehlo.bounds<5, 5>>) ->"
                            + " tensor<?x?xf32, #stablehlo.bounds<5, 5>> {\n"
                            + "  %0 = stablehlo.reshape %a : (tensor<?x?xf32, #stablehlo.bounds<5,"
                            + " 5>>) -> tensor<?x?xf32, #stablehlo.bounds<5, 5>>\n"
                            + "  return %0 : tensor<?x?xf32, #stablehlo.bounds<5, 5>>\n"
                            + "}"),
                Arguments.of(
                        "attr",
                        "stablehlo.dot",
                        "func.func @f(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {\n"
                            + "  %0 = \"stablehlo.dot\"(%a, %a) {precision_config = [\"FOO\"]} :"
                            + " (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>\n"
                            + "  return %0 : tensor<2x2xi32>\n"
                            + "}"),
                Arguments.of(
                        "replica_groups",
                        "stablehlo.all_gather",
                        "func.func @f(%a: tensor<16x8xf32>) -> tensor<16x16xf32> {\n"
                                + "  %0 = \"stablehlo.all_gather\"(%a) {all_gather_dim = 1 : i64,"
                                + " replica_groups = #stablehlo.replica_group_mesh_axes<mesh ="
                                + " @missing, axes = [#stablehlo.axis_ref<name = \"x\">]>} :"
                                + " (tensor<16x8xf32>) -> tensor<16x16xf32>\n"
                                + "  return %0 : tensor<16x16xf32>\n}"),
                Arguments.of(
                        "I8",
                        "stablehlo.dot_general",
                        "func.func @f(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
                            + "  %0 = stablehlo.dot_general %a, %a, contracting_dims = [1] x [0],"
                            + " algorithm = <lhs_precision_type = f32, rhs_precision_type = f32,"
                            + " accumulation_type = f32, lhs_component_count = 3,"
                            + " rhs_component_count = 1, num_primitive_operations = 1,"
                            + " allow_imprecise_accumulation = false> : (tensor<2x2xf32>,"
                            + " tensor<2x2xf32>) -> tensor<2x2xf32>\n"
                            + "  return %0 : tensor<2x2xf32>\n"
                            + "}"),
                Arguments.of(
                        "custom_call",
                        "stablehlo.custom_call",
                        "func.func @f(%a: memref<4xf32>) {\n"
                                + "  %0 = stablehlo.custom_call @FanOut(%a) : (memref<4xf32>) ->"
                                + " memref<4xf32>\n"
                                + "  return\n}"),
                Arguments.of(
                        "window",
                        "stablehlo.dynamic_conv",
                        "func.func @f(%a: tensor<1x8x8x207xf32>, %k: tensor<0x3x207x16xf32>, %p:"
                                + " tensor<2x2xi64>) -> tensor<1x8x8x16xf32> {\n"
                                + "  %0 = \"stablehlo.dynamic_conv\"(%a, %k, %p) {dimension_numbers"
                                + " = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,"
                                + " feature_group_count = 1 : i64, batch_group_count = 1 : i64} :"
                                + " (tensor<1x8x8x207xf32>, tensor<0x3x207x16xf32>,"
                                + " tensor<2x2xi64>) -> tensor<1x8x8x16xf32>\n"
                                + "  return %0 : tensor<1x8x8x16xf32>\n}"),
                Arguments.of(
                        "C1",
                        "stablehlo.add",
                        "func.func @f(%a: tensor<2xf32>, %b: tensor<3xf32>) -> tensor<2xf32> {\n"
                                + "  %0 = stablehlo.add %a, %b : (tensor<2xf32>, tensor<3xf32>) ->"
                                + " tensor<2xf32>\n"
                                + "  return %0 : tensor<2xf32>\n"
                                + "}"),
                Arguments.of(
                        "type",
                        "stablehlo.or",
                        "func.func @f(%a: tensor<2xf32>) -> tensor<2xf32> {\n"
                                + "  %0 = stablehlo.or %a, %a : tensor<2xf32>\n"
                                + "  return %0 : tensor<2xf32>\n}"),
                Arguments.of(
                        "C5",
                        "stablehlo.broadcast_in_dim",
                        "func.func @f(%a: tensor<3xf32>) -> tensor<2x4xf32> {\n"
                            + "  %0 = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<3xf32>)"
                            + " -> tensor<2x4xf32>\n"
                            + "  return %0 : tensor<2x4xf32>\n"
                            + "}"),
                Arguments.of(
                        "C2",
                        "stablehlo.transpose",
                        "func.func @f(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {\n"
                            + "  %0 = stablehlo.transpose %a, dims = [0, 0] : (tensor<2x3xf32>) ->"
                            + " tensor<3x2xf32>\n"
                            + "  return %0 : tensor<3x2xf32>\n"
                            + "}"),
                Arguments.of(
                        "C2",
                        "stablehlo.reshape",
                        "func.func @f(%a: tensor<2x3xf32>) -> tensor<5xf32> {\n"
                            + "  %0 = stablehlo.reshape %a : (tensor<2x3xf32>) -> tensor<5xf32>\n"
                            + "  return %0 : tensor<5xf32>\n"
                            + "}"),
                Arguments.of(
                        "C10",
                        "stablehlo.dot_general",
                        "func.func @f(%a: tensor<2x3xf32>, %b: tensor<4x5xf32>) -> tensor<2x5xf32>"
                            + " {\n"
                            + "  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] :"
                            + " (tensor<2x3xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>\n"
                            + "  return %0 : tensor<2x5xf32>\n"
                            + "}"),
                Arguments.of(
                        "C7",
                        "stablehlo.reduce",
                        "func.func @f(%a: tensor<2x3xf32>, %i: tensor<f32>) -> tensor<3xf32> {\n"
                            + "  %0 = stablehlo.reduce(%a init: %i) applies stablehlo.add across"
                            + " dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) ->"
                            + " tensor<3xf32>\n"
                            + "  return %0 : tensor<3xf32>\n"
                            + "}"),
                Arguments.of(
                        "C5",
                        "stablehlo.slice",
                        "func.func @f(%a: tensor<8xf32>) -> tensor<3xf32> {\n"
                                + "  %0 = stablehlo.slice %a [0:8:2] : (tensor<8xf32>) ->"
                                + " tensor<3xf32>\n"
                                + "  return %0 : tensor<3xf32>\n"
                                + "}"),
                Arguments.of(
                        "func",
                        "func.return",
                        "func.func @f(%a: tensor<2xf32>) -> tensor<3xf32> {\n"
                                + "  return %a : tensor<2xf32>\n}"),
                Arguments.of(
                        "C1",
                        "stablehlo.composite",
                        "func.func @g() { return }\n"
                            + "func.func @f() {\n"
                            + "  stablehlo.composite \"nonamespace\" { decomposition = @g } : () ->"
                            + " ()\n"
                            + "  return\n"
                            + "}"),
                Arguments.of(
                        "C2",
                        "stablehlo.all_gather",
                        "func.func @f(%a: tensor<8x2xf32>) -> tensor<8x8xf32> {\n"
                                + "  %0 = \"stablehlo.all_gather\"(%a) {all_gather_dim = 1 : i64,"
                                + " replica_groups = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>}"
                                + " : (tensor<8x2xf32>) -> tensor<8x8xf32>\n"
                                + "  return %0 : tensor<8x8xf32>\n}"));
    }

    @ParameterizedTest(name = "{1} {0}")
    @MethodSource("invalid")
    void invalidProgramIsRejected(String constraint, String opName, String source) {
        MlirParser.Parsed parsed = MlirParser.parseWithLines(source, "test");
        List<Diagnostic> diagnostics = Verifier.verify(parsed.program());
        assertTrue(
                diagnostics.stream()
                        .anyMatch(
                                d ->
                                        d.constraint().equals(constraint)
                                                && d.operation().name().equals(opName)),
                "expected " + opName + " (" + constraint + "), got " + diagnostics);
        assertTrue(
                parsed.line(diagnostics.getFirst().operation()) > 0,
                "diagnostic has a source line");
    }
}
