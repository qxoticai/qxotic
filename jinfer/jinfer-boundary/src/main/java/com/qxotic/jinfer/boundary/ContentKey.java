package com.qxotic.jinfer.boundary;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * Stable identity of source content, for caching anything derived from it. Opaque: compared for
 * equality, never parsed. Canonical form is {@code "sha256:<hex>"} of the SOURCE bytes (decoded
 * data drifts, the source does not); caller-assigned ids are equally valid.
 */
public record ContentKey(String value) {

    public ContentKey {
        if (value == null || value.isBlank())
            throw new IllegalArgumentException("empty content key");
    }

    private static final String SHA256_PREFIX = "sha256:";

    public static ContentKey sha256(byte[] source) {
        try {
            return new ContentKey(
                    SHA256_PREFIX
                            + HexFormat.of()
                                    .formatHex(
                                            MessageDigest.getInstance("SHA-256").digest(source)));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /**
     * The raw 32 digest bytes of a {@link #sha256(byte[])} key - the one place the canonical form
     * is read back, so machinery that needs the digest itself (cache key chains, file headers)
     * never parses the string. Caller-assigned ids have no digest and fail loudly.
     */
    public byte[] digestBytes() {
        if (!value.startsWith(SHA256_PREFIX)) {
            throw new IllegalStateException("not a sha256 content key: " + value);
        }
        return HexFormat.of().parseHex(value.substring(SHA256_PREFIX.length()));
    }

    @Override
    public String toString() {
        return value;
    }
}
