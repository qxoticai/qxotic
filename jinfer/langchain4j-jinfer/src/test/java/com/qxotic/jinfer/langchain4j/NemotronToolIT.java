package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against Nemotron (Cascade 2, {@code nemotron_h_moe}): fully native tool
 * codec - XML declarations in the system turn, {@code <tool_call>} spans with the XML function
 * payload shared with Qwen 3.5, tool results folded into user turns. {@code REQUIRED} is marker
 * seeding only (no pin hook).
 */
class NemotronToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.nemotronModel", ModelFixture.NEMOTRON_30B_Q8.path().toString()));
    }
}
