package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractConstraintIT} against gpt-oss (Harmony channels: analysis free, final bound). */
class GptOssConstraintIT extends AbstractConstraintIT {

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
}
