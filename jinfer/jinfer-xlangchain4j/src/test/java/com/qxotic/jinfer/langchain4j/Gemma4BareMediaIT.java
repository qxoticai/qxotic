package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractMediaIT} against Gemma 4 E2B loaded BARE - a vision-capable checkpoint WITHOUT its
 * mmproj companion. The fail lane here is the one no other cell reaches: the family's own native
 * codec walks the conversation, meets the media part, and punts from {@code requireSupported}
 * ("media on a text-only load") - the engine then refuses with the attach-the-companion recipe.
 * Distinct from {@link Lfm2MediaIT}, where a genuinely text-only family's template refuses the same
 * content through its own path.
 */
class Gemma4BareMediaIT extends AbstractMediaIT {

    private static final String MODEL_REF =
            "hf.co/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";

    @Override
    Path modelPath() {
        return TestModels.require(MODEL_REF);
    }
}
