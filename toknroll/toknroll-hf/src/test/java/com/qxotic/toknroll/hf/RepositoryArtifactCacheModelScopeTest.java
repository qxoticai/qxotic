package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.fetch.Transport;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("network")
@Tag("local-external")
class RepositoryArtifactCacheModelScopeTest {

    @Test
    void fetchModelScope_deepSeekV4TokenizerJson_defaultsToMasterBranch() throws Exception {
        RepositoryArtifactCache cache = RepositoryArtifactCache.create(Transport.system());

        Path tokenizerJson =
                cache.fetchModelScope(
                        "deepseek-ai", "DeepSeek-V4-Pro", null, "tokenizer.json", false);

        assertTrue(Files.exists(tokenizerJson), "Expected tokenizer.json to be downloaded");
        String text = Files.readString(tokenizerJson, StandardCharsets.UTF_8);
        assertTrue(text.contains("\"model\""), "Expected tokenizer.json to contain model key");
        assertTrue(
                tokenizerJson.toString().contains("/master/"),
                "ModelScope cache path should use master revision");
    }
}
