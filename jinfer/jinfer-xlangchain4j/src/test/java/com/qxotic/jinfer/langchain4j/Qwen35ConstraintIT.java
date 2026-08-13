package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractConstraintIT} against Qwen3.5 (prompt-opened think span). */
class Qwen35ConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.qwen35Model",
                        TestModels.find("hf.co/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf"))
                                .toString()));
    }
}
