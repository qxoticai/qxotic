package com.qxotic.jota.stablehlo.portable;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.text.MlirParser;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Reads the portable artifacts of {@code stablehlo/tests/vhlo}: one VHLO bytecode file per
 * historical StableHLO version (0.9.0 to current), invalid artifacts, and the text programs the
 * artifacts were serialized from.
 */
class PortableArtifactSampleReadTest {

    private static final Pattern VERSIONED_NAME =
            Pattern.compile("stablehlo_legalize_to_vhlo\\.(\\d+)_(\\d+)_(\\d+)\\.mlir\\.bc");

    /** {@code // CHECK: note: in bytecode version 6 produced by: StableHLO_v2.0.0}. */
    private static final Pattern PRODUCER_NOTE =
            Pattern.compile("in bytecode version (\\d+) produced by: (\\S+)");

    static Stream<Path> versionedArtifacts() {
        return Samples.files("portable", ".mlir.bc").stream()
                .filter(p -> VERSIONED_NAME.matcher(p.getFileName().toString()).matches());
    }

    static Stream<Path> invalidArtifacts() {
        return Samples.files("portable", ".mlir.bc").stream()
                .filter(p -> p.getFileName().toString().startsWith("invalid_"));
    }

    static Stream<Path> textTwins() {
        return Samples.files("portable", ".mlir").stream()
                .filter(p -> p.getFileName().toString().startsWith("stablehlo_legalize_to_vhlo."));
    }

    private static BytecodeHeader header(Path file) {
        try {
            return BytecodeHeader.read(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("versionedArtifacts")
    void producerVersionMatchesFileName(Path file) throws IOException {
        byte[] data = Files.readAllBytes(file);
        assertTrue(BytecodeHeader.isBytecode(data));
        BytecodeHeader header = BytecodeHeader.read(data);

        Matcher m = VERSIONED_NAME.matcher(file.getFileName().toString());
        assertTrue(m.matches());
        StableHloVersion expected =
                new StableHloVersion(
                        Integer.parseInt(m.group(1)),
                        Integer.parseInt(m.group(2)),
                        Integer.parseInt(m.group(3)));
        assertTrue(header.isPortableArtifact(), "producer " + header.producer());
        assertEquals(expected, header.stablehloVersion().orElseThrow());
        assertTrue(expected.isSupported(), expected + " outside the compatibility window");
        // MLIR bytecode format versions in use: 0 (StableHLO 0.9.0) up to 6
        assertTrue(header.bytecodeVersion() >= 0 && header.bytecodeVersion() <= 6);
        assertTrue(header.headerSize() < data.length);
    }

    @Test
    void oldestArtifactUsesBytecodeVersionZero() {
        BytecodeHeader header =
                header(Samples.root().resolve("portable/stablehlo_legalize_to_vhlo.0_9_0.mlir.bc"));
        assertEquals(0, header.bytecodeVersion());
        assertEquals("StableHLO_v0.9.0", header.producer());
    }

    @Test
    void coversEveryVersionFromMinimum() {
        List<StableHloVersion> versions =
                versionedArtifacts()
                        .map(p -> header(p).stablehloVersion().orElseThrow())
                        .sorted()
                        .toList();
        assertEquals(StableHloVersion.MINIMUM, versions.getFirst());
        // one artifact per StableHLO release since 0.9.0 (39 at the bundled revision)
        assertTrue(versions.size() >= 39, "found " + versions.size() + " versioned artifacts");
    }

    /** Invalid artifacts still have well-formed headers; their .mlir twin names the producer. */
    @ParameterizedTest(name = "{0}")
    @MethodSource("invalidArtifacts")
    void invalidArtifactHeaderMatchesExpectedDiagnostic(Path file) throws IOException {
        BytecodeHeader header = BytecodeHeader.read(file);
        assertTrue(header.isPortableArtifact());
        Path twin = file.resolveSibling(file.getFileName().toString().replace(".mlir.bc", ".mlir"));
        if (Files.exists(twin)) {
            Matcher note = PRODUCER_NOTE.matcher(Files.readString(twin));
            if (note.find()) {
                assertEquals(Long.parseLong(note.group(1)), header.bytecodeVersion());
                assertEquals(note.group(2), header.producer());
            }
        }
    }

    @Test
    void artifactFromTheFutureIsOutsideTheCompatibilityWindow() {
        BytecodeHeader header =
                header(Samples.root().resolve("portable/invalid_vhlo_future.mlir.bc"));
        StableHloVersion version = header.stablehloVersion().orElseThrow();
        assertEquals(StableHloVersion.parse("2.0.0"), version);
        assertFalse(version.isSupported());
    }

    @Test
    void rejectsNonBytecode() {
        byte[] text = "module {}".getBytes();
        assertFalse(BytecodeHeader.isBytecode(text));
        assertThrows(IllegalArgumentException.class, () -> BytecodeHeader.read(text));
    }

    /** The StableHLO text the versioned artifacts were serialized from. */
    @ParameterizedTest(name = "{0}")
    @MethodSource("textTwins")
    void sourceProgramOfArtifactParses(Path file) throws IOException {
        Program program = MlirParser.parse(file);
        assertTrue(program.functions().size() > 100, "functions: " + program.functions().size());
    }

    @Test
    void prefixVarIntDecoding() {
        // 1 byte: low bit set, value in the upper 7 bits
        assertEquals(6, BytecodeHeader.readPrefixVarInt(new byte[] {0x0D}, new int[] {0}));
        // 2 bytes: one trailing zero bit
        int[] cursor = {0};
        assertEquals(300, BytecodeHeader.readPrefixVarInt(new byte[] {(byte) 0xB2, 0x04}, cursor));
        assertEquals(2, cursor[0]);
        // 9 bytes: zero first byte, the full 64-bit little-endian value follows
        byte[] big = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        cursor[0] = 0;
        assertEquals(0x0807060504030201L, BytecodeHeader.readPrefixVarInt(big, cursor));
        assertEquals(9, cursor[0]);
    }
}
