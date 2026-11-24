package com.qxotic.jota.stablehlo.portable;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Header of an MLIR bytecode file: magic {@code ML\xEFR}, bytecode format version (prefix varint),
 * producer string (null terminated).
 *
 * <p>StableHLO portable artifacts are MLIR bytecode of the VHLO dialect whose producer is {@code
 * StableHLO_vX.Y.Z}, the StableHLO version the artifact targets (see {@code
 * stablehlo/dialect/Serialization.h}).
 */
public record BytecodeHeader(long bytecodeVersion, String producer, int headerSize) {

    private static final byte[] MAGIC = {'M', 'L', (byte) 0xEF, 'R'};
    private static final Pattern STABLEHLO_PRODUCER =
            Pattern.compile("StableHLO_v(\\d+\\.\\d+\\.\\d+)");

    public static boolean isBytecode(byte[] data) {
        if (data.length < MAGIC.length) {
            return false;
        }
        for (int i = 0; i < MAGIC.length; i++) {
            if (data[i] != MAGIC[i]) {
                return false;
            }
        }
        return true;
    }

    public static BytecodeHeader read(Path file) throws IOException {
        return read(Files.readAllBytes(file));
    }

    public static BytecodeHeader read(byte[] data) {
        if (!isBytecode(data)) {
            throw new IllegalArgumentException("not an MLIR bytecode file (bad magic)");
        }
        int[] cursor = {MAGIC.length};
        long version = readPrefixVarInt(data, cursor);
        int start = cursor[0];
        int end = start;
        while (end < data.length && data[end] != 0) {
            end++;
        }
        if (end >= data.length) {
            throw new IllegalArgumentException("unterminated producer string");
        }
        String producer = new String(data, start, end - start, StandardCharsets.UTF_8);
        return new BytecodeHeader(version, producer, end + 1);
    }

    /**
     * MLIR "PrefixVarInt": the number of trailing zero bits of the first byte gives the number of
     * additional bytes; a zero first byte means 8 additional bytes holding the full value.
     */
    static long readPrefixVarInt(byte[] data, int[] cursor) {
        int p = cursor[0];
        int first = data[p] & 0xFF;
        if ((first & 1) != 0) {
            cursor[0] = p + 1;
            return first >>> 1;
        }
        if (first == 0) {
            long v = 0;
            for (int i = 8; i >= 1; i--) {
                v = (v << 8) | (data[p + i] & 0xFFL);
            }
            cursor[0] = p + 9;
            return v;
        }
        int extra = Integer.numberOfTrailingZeros(first);
        long v = 0;
        for (int i = extra; i >= 0; i--) {
            v = (v << 8) | (data[p + i] & 0xFFL);
        }
        cursor[0] = p + extra + 1;
        return v >>> (extra + 1);
    }

    /** The StableHLO version of a portable artifact ({@code 1.1.0}), if the producer says so. */
    public Optional<StableHloVersion> stablehloVersion() {
        Matcher m = STABLEHLO_PRODUCER.matcher(producer);
        return m.matches() ? Optional.of(StableHloVersion.parse(m.group(1))) : Optional.empty();
    }

    public boolean isPortableArtifact() {
        return stablehloVersion().isPresent();
    }
}
