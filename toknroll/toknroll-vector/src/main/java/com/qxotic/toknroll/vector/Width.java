package com.qxotic.toknroll.vector;

import jdk.incubator.vector.ByteVector;

/**
 * The vector width every kernel of this module uses: 128, 256 or 512 bits. By default the
 * platform's preferred width, capped at 512 (a split block is 64 bytes) and raised to 128 where the
 * platform prefers less; {@code -Dtoknroll.vector.width=<bits>} picks one of the three, as tests
 * and an image built for a narrower target do. A constant, so the JIT bakes the shapes in.
 */
final class Width {
    private Width() {}

    static final int BITS = bits(System.getProperty("toknroll.vector.width"));

    static int bits(String requested) {
        if (requested == null || requested.isBlank()) {
            return Math.max(128, Math.min(512, ByteVector.SPECIES_PREFERRED.vectorBitSize()));
        }
        int bits = Integer.parseInt(requested.strip());
        if (bits != 128 && bits != 256 && bits != 512) {
            throw new IllegalArgumentException(
                    "toknroll.vector.width must be 128, 256 or 512, not " + requested);
        }
        return bits;
    }
}
