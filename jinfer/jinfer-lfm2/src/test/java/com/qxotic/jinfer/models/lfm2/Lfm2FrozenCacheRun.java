// Frozen multi-prompt cache (use case B) validation on LFM2.5 via the shared testkit scenario.
//   java ... com.qxotic.jinfer.models.lfm2.Lfm2FrozenCacheRun [model.gguf]
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.testkit.FrozenScenario;
import com.qxotic.jinfer.testkit.Harness;
import java.nio.file.Path;

public final class Lfm2FrozenCacheRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        new FrozenScenario<>(new Harness<>(Lfm2.loadModel(path, 4096), path, 4096))
                .run("Lfm2FrozenCacheRun");
    }
}
