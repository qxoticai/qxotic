package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractMediaIT} against LFM3-VL-2B loaded BARE, without its mmproj: the fail lane, where
 * LFM3's template meets an image or a clip with no tower to read it and refuses with the remedy.
 */
class Lfm3BareMediaIT extends AbstractMediaIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/LiquidAI/LFM3-VL-2B-GGUF/LFM3-VL-2B-Q8_0.gguf");
    }
}
