package com.qxotic.jota.stablehlo.portable;

import com.qxotic.jota.stablehlo.bytecode.MlirBytecodeReader;
import com.qxotic.jota.stablehlo.bytecode.MlirBytecodeWriter;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.vhlo.StableHloToVhlo;
import com.qxotic.jota.stablehlo.vhlo.VhloToStableHlo;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * StableHLO portable artifacts ({@code stablehlo/dialect/Serialization.h}): MLIR bytecode of a VHLO
 * program whose producer string is {@code StableHLO_vX.Y.Z}.
 *
 * <p>{@link #deserialize} checks the version window, decodes the bytecode and legalizes VHLO to
 * StableHLO, like {@code stablehlo::deserializePortableArtifact}. {@link #serialize} legalizes
 * StableHLO to VHLO ops valid at the target version and writes the bytecode format that version
 * expects, like {@code stablehlo::serializePortableArtifact}.
 */
public final class PortableArtifact {

    private PortableArtifact() {}

    /** The StableHLO version an artifact was serialized for. */
    public static StableHloVersion version(byte[] data) {
        return BytecodeHeader.read(data)
                .stablehloVersion()
                .orElseThrow(
                        () ->
                                new IllegalArgumentException(
                                        "not a StableHLO portable artifact"
                                                + " (producer '"
                                                + BytecodeHeader.read(data).producer()
                                                + "')"));
    }

    public static Program deserialize(Path file) throws IOException {
        return deserialize(Files.readAllBytes(file));
    }

    /** Reads a portable artifact into a StableHLO program. */
    public static Program deserialize(byte[] data) {
        StableHloVersion version = version(data);
        if (!version.isSupported()) {
            throw new IllegalArgumentException(
                    "portable artifact version "
                            + version
                            + " is outside the supported window "
                            + "["
                            + StableHloVersion.MINIMUM
                            + ", "
                            + StableHloVersion.CURRENT
                            + "]");
        }
        return VhloToStableHlo.convert(readVhlo(data));
    }

    /** The VHLO program of an artifact, before legalization (op names {@code vhlo.xxx_vN}). */
    public static Program readVhlo(byte[] data) {
        return MlirBytecodeReader.read(data).program();
    }

    /** Serializes for the newest supported version. */
    public static byte[] serialize(Program program) {
        return serialize(program, StableHloVersion.CURRENT);
    }

    /**
     * Serializes {@code program} as a portable artifact readable by StableHLO {@code target} and
     * any later version.
     */
    public static byte[] serialize(Program program, StableHloVersion target) {
        Program vhlo = StableHloToVhlo.convert(program, target);
        return MlirBytecodeWriter.write(vhlo, bytecodeVersion(target), "StableHLO_v" + target);
    }

    public static void serialize(Program program, StableHloVersion target, Path file)
            throws IOException {
        Files.write(file, serialize(program, target));
    }

    /** MLIR bytecode format version used for a target ({@code Version::getBytecodeVersion}). */
    public static int bytecodeVersion(StableHloVersion target) {
        if (!target.isSupported()) {
            throw new IllegalArgumentException("unsupported target version " + target);
        }
        if (target.compareTo(new StableHloVersion(0, 10, 0)) < 0) {
            return 0;
        }
        if (target.compareTo(new StableHloVersion(0, 12, 0)) < 0) {
            return 1;
        }
        if (target.compareTo(new StableHloVersion(0, 14, 0)) < 0) {
            return 3;
        }
        if (target.compareTo(new StableHloVersion(0, 15, 0)) < 0) {
            return 4;
        }
        return 6;
    }
}
