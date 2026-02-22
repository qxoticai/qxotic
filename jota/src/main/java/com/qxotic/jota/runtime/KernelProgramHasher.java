package com.qxotic.jota.runtime;

import com.qxotic.jota.tensor.KernelCacheKey;
import com.qxotic.jota.tensor.KernelProgram;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Map;

public final class KernelProgramHasher {

    private KernelProgramHasher() {}

    public static KernelCacheKey keyFor(KernelProgram program) {
        MessageDigest digest = sha256();
        update(digest, program.kind().name());
        update(digest, program.language());
        update(digest, program.entryPoint());
        for (Map.Entry<String, String> entry : program.options().entrySet()) {
            update(digest, entry.getKey());
            update(digest, entry.getValue());
        }
        Object payload = program.payload();
        if (payload instanceof String source) {
            update(digest, source);
        } else if (payload instanceof byte[] bytes) {
            digest.update(bytes);
        } else {
            update(digest, payload.toString());
        }
        String hash = toHex(digest.digest());
        return KernelCacheKey.of(hash);
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("Missing SHA-256 MessageDigest", e);
        }
    }

    private static void update(MessageDigest digest, String value) {
        digest.update(value.getBytes(StandardCharsets.UTF_8));
    }

    private static String toHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
