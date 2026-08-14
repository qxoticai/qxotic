package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * {@link AbstractMediaIT} against LFM2.5 loaded BARE: the family carries a vision sidecar in x, but
 * a text-only load (no companion attached) still refuses media content loudly - the battery's
 * fourth cell. Plus the load-time gate: an INCOMPATIBLE companion (the Gemma E2B projector, width
 * 1536, against the 350M backbone, width 1024) is refused at LOAD, not discovered mid-request.
 */
class Lfm2MediaIT extends AbstractMediaIT {

    private static final String MODEL_REF = "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf";
    private static final String MMPROJ_REF = "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf";

    @Override
    Path modelPath() {
        return TestModels.require(MODEL_REF);
    }

    @Test
    void attachingAnIncompatibleCompanionIsRefusedAtLoad() {
        // resolve BEFORE assertThrows: an abort inside the lambda would surface as a failure
        Path model = TestModels.require(MODEL_REF);
        Path mmproj = TestModels.require(MMPROJ_REF);
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                JinferChatModel.builder()
                                        .modelPath(model)
                                        .companion("media", mmproj)
                                        .build());
        assertTrue(e.getMessage().contains("does not match model width"), e.getMessage());
    }
}
