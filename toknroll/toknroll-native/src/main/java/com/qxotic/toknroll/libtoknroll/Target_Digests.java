package com.qxotic.toknroll.libtoknroll;

import com.oracle.svm.core.annotate.Substitute;
import com.oracle.svm.core.annotate.TargetClass;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;

/** The hub's SHA-256, from Sha256 instead of MessageDigest.getInstance: no security providers. */
@TargetClass(className = "com.qxotic.jinfer.hub.Fetch", onlyWith = Lean.class)
final class Target_com_qxotic_jinfer_hub_Fetch {
    @Substitute
    private static MessageDigest sha256Digest() {
        return new Sha256();
    }
}

/** The tiktoken cache's key, the same digest. */
@TargetClass(className = "com.qxotic.toknroll.hf.RepositoryArtifactCache", onlyWith = Lean.class)
final class Target_com_qxotic_toknroll_hf_RepositoryArtifactCache {
    @Substitute
    private static String sha256(String input) {
        return HexFormat.of()
                .formatHex(new Sha256().digest(input.getBytes(StandardCharsets.UTF_8)));
    }
}
