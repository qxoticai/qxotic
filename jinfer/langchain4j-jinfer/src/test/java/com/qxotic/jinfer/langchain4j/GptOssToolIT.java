package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against gpt-oss (Harmony): declarations in the developer block's
 * TypeScript namespace, {@code commentary to=functions.*} calls parsed structurally, and {@code
 * REQUIRED} forcing via the {@code <|channel|>} seed + name pin + forced header epilogue.
 */
class GptOssToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.gptossModel", ModelFixture.GPTOSS_20B_Q8.path().toString()));
    }
}
