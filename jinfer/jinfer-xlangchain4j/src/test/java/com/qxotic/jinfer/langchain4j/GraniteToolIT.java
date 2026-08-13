package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against Granite 4.1: native tool codec - the tools message (whole-envelope
 * JSON signatures) joins the system turn, calls are {@code <tool_call>} JSON spans, consecutive
 * results fold into one user turn. {@code REQUIRED} is marker seeding only (no pin hook).
 */
class GraniteToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.graniteModel",
                        TestModels.find(
                                        "hf.co/ibm-granite/granite-4.1-3b-GGUF/granite-4.1-3b-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/ibm-granite/granite-4.1-3b-GGUF/granite-4.1-3b-Q8_0.gguf"))
                                .toString()));
    }
}
