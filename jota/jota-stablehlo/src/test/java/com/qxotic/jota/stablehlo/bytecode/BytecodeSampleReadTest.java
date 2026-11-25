package com.qxotic.jota.stablehlo.bytecode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.StructuralEquality;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.transform.CanonicalizeDefaults;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Reads the plain (non-portable) MLIR bytecode samples written by MLIR itself (jaxlib, see {@code
 * tools/gen-bytecode-samples.py}): {@code bytecode/<name>.mlirbc} at jaxlib's bytecode version and
 * {@code <name>.v<N>.mlirbc} at older ones. Each must read to the program of its text source {@code
 * text/testdata/<name>.mlir}.
 */
class BytecodeSampleReadTest {

    private static final Pattern NAME = Pattern.compile("(.*?)(?:\\.v(\\d+))?\\.mlirbc");

    static Stream<Path> samples() {
        return Samples.files("bytecode", ".mlirbc").stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("samples")
    void readsToTheProgramOfItsSource(Path file) throws IOException {
        Matcher m = NAME.matcher(file.getFileName().toString());
        assertTrue(m.matches(), file.toString());
        Path source = Samples.root().resolve("text/testdata/" + m.group(1) + ".mlir");

        MlirBytecodeReader.Result result = MlirBytecodeReader.read(Files.readAllBytes(file));
        assertTrue(result.producer().startsWith("MLIR"), result.producer());
        if (m.group(2) != null) {
            assertEquals(Long.parseLong(m.group(2)), result.bytecodeVersion());
        }
        Program expected = CanonicalizeDefaults.apply(MlirParser.parse(source));
        Program actual = CanonicalizeDefaults.apply(result.program());
        Optional<String> diff = StructuralEquality.diff(expected, actual);
        if (diff.isPresent()) {
            fail(diff.get());
        }
    }

    @Test
    void sampleSetCoversCurrentAndOldBytecodeVersions() {
        List<Path> files = samples().toList();
        assertTrue(files.size() > 100, "found " + files.size());
        assertTrue(files.stream().anyMatch(f -> f.toString().endsWith(".v1.mlirbc")));
        assertTrue(files.stream().anyMatch(f -> f.toString().endsWith(".v4.mlirbc")));
    }
}
