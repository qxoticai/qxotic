package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;

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
                        "jinfer.gptossModel",
                        TestModels.find("hf.co/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf"))
                                .toString()));
    }

    /**
     * Same flow as the shared test, with the emptiness claim demoted to an assumption for this
     * family alone. The 20B decorates a no-parameter call with a commentary argument in roughly one
     * run in four ({@code {"commentary":"calling"}}, {@code {"name":"refresh_cache","arguments":
     * {}}}) - what it INVENTS is model behavior, and no seed pins it: this is a MoE, whose threaded
     * expert reductions are not bit-deterministic, so the same seed decodes differently run to run.
     * The wire claims stay assertions: the call is parsed, and it names the offered tool.
     */
    @Override
    void noParameterTool() {
        var r = ask("Please refresh the cache now using the tool.", REFRESH);
        var call = assumeCall(r);
        assertEquals("refresh_cache", call.name());
        String raw = call.arguments() == null ? "" : call.arguments().strip();
        Assumptions.assumeTrue(
                raw.isEmpty() || args(call).isEmpty(),
                "gpt-oss decorated a no-parameter call: " + raw);
    }
}
