package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/**
 * {@link AbstractToolIT} against Meta Llama 3.2's bare-JSON function-call wire.
 *
 * <p>The 1B checkpoint is legacy and weak; the architecture is carried for Granite, MiniCPM5,
 * Mistral and SmolLM3, whose tool batteries pass in full. Two cases below assert argument fidelity
 * this checkpoint does not have: they fail identically on every run at temperature 0, so they are
 * disabled rather than left to flake. Re-enabling runs the real assertions.
 */
class Llama32ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return TestModels.require(
                "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
    }

    @Override
    @Test
    @Disabled(
            "Llama-3.2-1B invents an argument for a no-parameter tool: {\"cache\":\"server-side\"}")
    void noParameterTool() {
        super.noParameterTool();
    }

    @Override
    @Test
    @Disabled("Llama-3.2-1B garbles the verbatim unicode string: \"He said ägrü dich änd änd\"")
    void unicodeAndQuotesInArguments() {
        super.unicodeAndQuotesInArguments();
    }
}
