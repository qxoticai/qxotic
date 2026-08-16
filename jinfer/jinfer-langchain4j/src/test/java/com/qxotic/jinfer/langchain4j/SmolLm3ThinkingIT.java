package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractThinkingIT} against SmolLM3 - the family that motivated it: its /think scaffold
 * does not open the think span, and the checkpoint closes the turn on its first token unless the
 * engine prompt-opens it (see {@code JinjaChatTemplate}); under a forced cap close it then
 * fabricated a turn header until the cap started closing on a paragraph break.
 */
class SmolLm3ThinkingIT extends AbstractThinkingIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/ggml-org/SmolLM3-3B-GGUF/SmolLM3-Q4_K_M.gguf");
    }
}
