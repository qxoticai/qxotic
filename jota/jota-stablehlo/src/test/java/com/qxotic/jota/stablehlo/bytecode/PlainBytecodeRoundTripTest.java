package com.qxotic.jota.stablehlo.bytecode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.StructuralEquality;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.transform.CanonicalizeDefaults;
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
 * Every bundled text program written as plain StableHLO bytecode ({@link
 * MlirBytecodeWriter#writePlain}) at bytecode versions 1 (attribute dictionaries), 4 and 6 (native
 * properties) reads back to the same program.
 */
class PlainBytecodeRoundTripTest {

    static Stream<Arguments> programs() {
        List<Arguments> out = new ArrayList<>();
        for (String dir : List.of("text/testdata", "text/interpret", "text/syntax")) {
            for (Path file : Samples.files(dir, ".mlir")) {
                List<String> chunks = MlirParser.splitInputFile(Samples.read(file));
                for (int i = 0; i < chunks.size(); i++) {
                    if (!Samples.isNegativeOrEmpty(chunks.get(i))) {
                        String name = dir + "/" + file.getFileName() + "#" + i;
                        out.add(Arguments.of(name, chunks.get(i)));
                    }
                }
            }
        }
        return out.stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("programs")
    void roundTrips(String name, String text) {
        Program source = MlirParser.parse(text, name);
        for (long version : new long[] {1, 4, MlirBytecodeReader.VERSION_CURRENT}) {
            byte[] bytes =
                    MlirBytecodeWriter.writePlain(
                            source, version, StableHloVersion.CURRENT, "jota-stablehlo");
            MlirBytecodeReader.Result read = MlirBytecodeReader.read(bytes);
            assertEquals(version, read.bytecodeVersion());
            Optional<String> diff =
                    StructuralEquality.diff(
                            CanonicalizeDefaults.apply(source),
                            CanonicalizeDefaults.apply(read.program()));
            if (diff.isPresent()) {
                fail("bytecode v" + version + ": " + diff.get());
            }
        }
    }

    @Test
    void recordsTheStableHloVersionAndUsesItsLayouts() {
        Program p =
                MlirParser.parse(
                        """
                        func.func @main(%a: tensor<f32>, %b: tensor<f32>) -> tensor<i1> {
                          %0 = stablehlo.compare GT, %a, %b, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                          return %0 : tensor<i1>
                        }
                        """);
        StableHloVersion target = StableHloVersion.parse("1.25.3");
        MlirBytecodeReader.Result read =
                MlirBytecodeReader.read(MlirBytecodeWriter.writePlain(p, 6, target, "test"));
        byte[] v = read.dialectVersions().get("stablehlo");
        assertTrue(v != null && v.length == 3, "stablehlo dialect version written");
        assertEquals(List.of(1, 25, 3), List.of((int) v[0] >> 1, (int) v[1] >> 1, (int) v[2] >> 1));
        assertTrue(StructuralEquality.diff(p, read.program()).isEmpty());
    }
}
