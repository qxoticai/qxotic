package com.qxotic.jinfer.cache;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * The cache package's one digest idiom: SHA-256 folded to four longs (block keys, media
 * fingerprints, model seeds all speak it).
 */
final class Sha256 {

    private Sha256() {}

    static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** Finalizes {@code sha} and returns the 256-bit digest as 4 little-endian longs. */
    static long[] digestLongs(MessageDigest sha) {
        ByteBuffer out = ByteBuffer.wrap(sha.digest()).order(ByteOrder.LITTLE_ENDIAN);
        return new long[] {out.getLong(), out.getLong(), out.getLong(), out.getLong()};
    }
}
