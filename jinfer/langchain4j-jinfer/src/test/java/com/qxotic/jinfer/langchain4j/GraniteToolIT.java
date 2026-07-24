package com.qxotic.jinfer.langchain4j;

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
                        "/home/mukel/Desktop/playground/models/ibm-granite/"
                                + "granite-4.1-3b-Q8_0.gguf"));
    }
}
