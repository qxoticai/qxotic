package com.qxotic.jinfer;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * Stable identity of source content, for caching anything derived from it. Compared for equality
 * and never PARSED, but printable on purpose: a key may be {@code "sha256:<hex>"} of the source
 * bytes (decoded data drifts, the source does not), a caller-assigned id, or a canonical
 * description of what the identity is made of - which is what a cache seed uses, so a mismatch can
 * say what differs instead of showing a bare digest.
 */
public record ContentKey(String value) {

    public ContentKey {
        if (value == null || value.isBlank())
            throw new IllegalArgumentException("empty content key");
    }

    private static final String SHA256_PREFIX = "sha256:";

    /** The canonical key for {@code source}: {@code "sha256:<hex>"} of its bytes. */
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
     * The 32 digest bytes this key stands for, for machinery that needs a fixed-width identity
     * (cache key chains, file headers) without parsing the string: the hex a {@link
     * #sha256(byte[])} key already carries, and SHA-256 of the value itself for any other key. A
     * printable key therefore verifies by hand - {@code printf %s "<value>" | sha256sum} is the
     * digest a cache artifact stores.
     */
    public byte[] digestBytes() {
        if (!value.startsWith(SHA256_PREFIX)) {
            return sha256(value.getBytes(StandardCharsets.UTF_8)).digestBytes();
        }
        return HexFormat.of().parseHex(value.substring(SHA256_PREFIX.length()));
    }

    @Override
    public String toString() {
        return value;
    }
}
