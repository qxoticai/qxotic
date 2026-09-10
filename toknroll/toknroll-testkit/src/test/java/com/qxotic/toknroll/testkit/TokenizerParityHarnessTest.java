package com.qxotic.toknroll.testkit;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TokenizerParityHarnessTest {

    @Test
    void usesGeneratedChunkOffsets(@TempDir Path dir) throws Exception {
        String hash = "8e0375adfc1f4563"; // sha256("2:3")
        Files.writeString(
                dir.resolve("chunks.json"),
                "[{\"offset\":2,\"size\":3,\"hash\":\"" + hash + "\",\"text\":\"cde\"}]");
        Path golden = dir.resolve("golden.json");
        Files.writeString(
                golden,
                "[{\"chunk_hash\":\"" + hash + "\",\"tokens\":[99,100,101],\"token_count\":3}]");

        TokenizerParityHarness.runParity(
                "sample",
                golden,
                "abcdef".getBytes(StandardCharsets.UTF_8),
                Integer.MAX_VALUE,
                1,
                text -> text.chars().toArray(),
                String::length,
                tokens -> new String(tokens, 0, tokens.length),
                token -> Character.toString((char) token));
    }
}
