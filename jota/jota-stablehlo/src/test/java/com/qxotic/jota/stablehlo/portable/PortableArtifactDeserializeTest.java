package com.qxotic.jota.stablehlo.portable;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.bytecode.BytecodeException;
import com.qxotic.jota.stablehlo.bytecode.MlirBytecodeReader;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.StructuralEquality;
import com.qxotic.jota.stablehlo.op.FuncOp;
import com.qxotic.jota.stablehlo.op.GenericOp;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.text.MlirPrinter;
import com.qxotic.jota.stablehlo.transform.CanonicalizeDefaults;
import com.qxotic.jota.stablehlo.vhlo.VhloOpSchemas;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;

/** Full deserialization of portable artifacts: MLIR bytecode → VHLO → StableHLO. */
class PortableArtifactDeserializeTest {

    static Stream<Path> validArtifacts() {
        return Samples.files("portable", ".mlir.bc").stream()
                .filter(p -> !p.getFileName().toString().startsWith("invalid_"));
    }

    static Stream<Path> artifactsWithSource() {
        return validArtifacts().filter(p -> Files.exists(twin(p)));
    }

    static Path twin(Path artifact) {
        return artifact.resolveSibling(
                artifact.getFileName().toString().replace(".mlir.bc", ".mlir"));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("validArtifacts")
    void deserializesToTypedStableHlo(Path file) throws IOException {
        Program program = PortableArtifact.deserialize(file);
        assertFalse(program.functions().isEmpty());
        if (file.getFileName().toString().startsWith("stablehlo_legalize_to_vhlo")) {
            assertTrue(program.functions().size() > 100); // one function per op / attribute
        }
        program.operation()
                .walk(
                        op -> {
                            assertFalse(op.name().startsWith("vhlo."), "left over " + op.name());
                            if (op.dialect().equals("stablehlo")) {
                                assertFalse(op.asOp() instanceof GenericOp, op.name());
                            }
                        });
        // the result is a valid program for the text tools too
        Program reparsed = MlirParser.parse(MlirPrinter.print(program));
        assertTrue(StructuralEquality.diff(program, reparsed).isEmpty());
    }

    /**
     * The artifact and the text it was serialized from denote the same program (modulo attributes
     * spelled with their default value, which serialization drops).
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("artifactsWithSource")
    void equalsTheSourceProgram(Path file) throws IOException {
        Program fromBytecode = CanonicalizeDefaults.apply(PortableArtifact.deserialize(file));
        Program fromText = CanonicalizeDefaults.apply(MlirParser.parse(twin(file)));
        Map<String, Operation> source = new HashMap<>();
        for (FuncOp f : fromText.functions()) {
            source.put(f.symName(), f.operation());
        }
        assertEquals(source.size(), fromBytecode.functions().size());
        for (FuncOp f : fromBytecode.functions()) {
            Operation expected = source.get(f.symName());
            if (expected == null) {
                fail("function @" + f.symName() + " is not in the source");
            }
            Optional<String> diff = StructuralEquality.diff(f.operation(), expected);
            if (diff.isPresent()) {
                fail("@" + f.symName() + ": " + diff.get());
            }
        }
    }

    @Test
    void readsBytecodeVersionZeroAndSix() throws IOException {
        byte[] v0 =
                Files.readAllBytes(
                        Samples.root()
                                .resolve("portable/stablehlo_legalize_to_vhlo.0_9_0.mlir.bc"));
        byte[] v6 =
                Files.readAllBytes(
                        Samples.root()
                                .resolve("portable/stablehlo_legalize_to_vhlo.1_27_0.mlir.bc"));
        MlirBytecodeReader.Result r0 = MlirBytecodeReader.read(v0);
        MlirBytecodeReader.Result r6 = MlirBytecodeReader.read(v6);
        assertEquals(0, r0.bytecodeVersion());
        assertEquals(6, r6.bytecodeVersion());
        // StableHLO records its version in the producer string, not as a dialect version
        assertEquals("StableHLO_v1.27.0", BytecodeHeader.read(v6).producer());
        long vhloOps =
                r6.program().operations().stream()
                        .filter(op -> op.name().equals("vhlo.func_v1"))
                        .count();
        assertTrue(vhloOps > 100);
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource({
        "invalid_vhlo_future.mlir.bc, outside the supported window",
        "invalid_vhlo_tensor_bad_blob_size.mlir.bc, invalid dense buffer size",
        "invalid_vhlo_tensor_dynamic_shape.mlir.bc, negative tensor dimension",
        "invalid_vhlo_tensor_non_shaped.mlir.bc, expected a ranked tensor type",
        "invalid_vhlo_tensor_overflow_shape.mlir.bc, element count overflows",
        "invalid_vhlo_tensor_unsupported_elem.mlir.bc, unsupported tensor type"
    })
    void rejectsInvalidArtifacts(String name, String message) {
        Path file = Samples.root().resolve("portable").resolve(name);
        RuntimeException e =
                assertThrows(RuntimeException.class, () -> PortableArtifact.deserialize(file));
        assertTrue(
                e instanceof BytecodeException || e instanceof IllegalArgumentException,
                e.toString());
        assertTrue(e.getMessage().contains(message), e.getMessage());
    }

    @Test
    void futureArtifactIsStillReadableAsRawVhlo() throws IOException {
        byte[] data =
                Files.readAllBytes(Samples.root().resolve("portable/invalid_vhlo_future.mlir.bc"));
        BytecodeException e =
                assertThrows(BytecodeException.class, () -> PortableArtifact.readVhlo(data));
        assertTrue(e.getMessage().contains("vhlo.constant_v99"), e.getMessage());
    }

    @Test
    void vhloSchemasCoverTheDialect() {
        assertEquals(149, VhloOpSchemas.size());
        assertEquals(
                java.util.List.of(
                        "arg_attrs", "function_type", "res_attrs", "sym_name", "sym_visibility"),
                VhloOpSchemas.get("func_v1").attributes());
    }
}
