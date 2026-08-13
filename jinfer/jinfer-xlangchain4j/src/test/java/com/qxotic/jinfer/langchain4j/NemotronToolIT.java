package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
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
                        "jinfer.nemotronModel",
                        TestModels.find(
                                        "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf"))
                                .toString()));
    }
}
