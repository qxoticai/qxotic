package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against Ministral 3 (Mistral v13 wire): {@code [AVAILABLE_TOOLS]} JSON
 * declarations, {@code [TOOL_CALLS]name[ARGS]{json}} calls (no close marker - spans chain to the
 * next call or {@code </s>}), {@code [TOOL_RESULTS]} results. {@code REQUIRED} is marker seeding
 * only (no pin hook).
 */
class MinistralToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.ministralModel",
                        TestModels.find(
                                        "hf.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q8_0.gguf"))
                                .toString()));
    }
}
