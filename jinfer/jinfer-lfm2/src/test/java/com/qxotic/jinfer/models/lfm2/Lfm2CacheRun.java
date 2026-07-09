// PromptCache validation + benchmark on LFM2.5 (hybrid conv+attention+MoE) via the shared
// testkit scenario.   java ... com.qxotic.jinfer.models.lfm2.Lfm2CacheRun [model.gguf]
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import java.nio.file.Path;

public final class Lfm2CacheRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        Harness<Lfm2.State> h =
                new Harness<>(
                        Lfm2.loadModel(path, 4096),
                        path,
                        4096,
                        false); // MoE: threaded decode is not byte-deterministic
        new CacheScenario<>(h, CacheScenario.Config.of("You are a concise assistant.", 120))
                .run("Lfm2CacheRun");
    }
}
