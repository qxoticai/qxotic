package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;

/**
 * {@link AbstractToolIT} against SmolLM3 (ChatML with the metadata header): python-repr tool
 * signatures in the system turn, {@code <tool_call>} JSON spans, tool results as user turns. {@code
 * REQUIRED} is marker seeding only (no pin hook).
 */
class SmolLm3ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.smollm3Model", ModelFixture.SMOLLM3_Q4.path().toString()));
    }

    /**
     * Capability-gated for this family alone. SmolLM3's own template renders a tool result as a
     * BARE user turn - nothing marks it as a result - and the 3B, handed a raw JSON blob that way,
     * opens a fresh turn instead of reading it (it used to write the user's next question itself;
     * the turn-header stop token in {@link com.qxotic.jinfer.models.llama.Llama#stopTokens} now
     * ends the reply there, so the visible symptom is an empty answer).
     *
     * <p>Not a wire bug: {@code SmolLm3Oracle} pins this exact conversation token-identical to the
     * model's own Jinja render, tool-result turn included, and every other family passes the shared
     * test. The assertion stays strict where it holds.
     */
    @Override
    void jsonToolResult() {
        Assumptions.assumeTrue(false, "SmolLM3 does not read a tool result (see javadoc)");
    }

    /**
     * Same limitation as {@link #jsonToolResult}, reached through the multi-turn loop: given "18C,
     * sunny" back, the 3B answers by re-emitting the call as TEXT ({@code {"name": "get_weather",
     * ...}}) instead of reporting the temperature. Result-reading is the capability this family
     * lacks; the wire around it is asserted by the tests that stay strict here.
     */
    @Override
    void multiTurnToolLoop() {
        Assumptions.assumeTrue(false, "SmolLM3 does not read a tool result (see javadoc)");
    }
}
