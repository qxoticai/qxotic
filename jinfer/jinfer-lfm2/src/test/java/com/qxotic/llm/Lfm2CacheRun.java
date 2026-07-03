// PromptCache validation + benchmark on LFM2.5 (hybrid conv+attention+MoE) via the shared
// testkit scenario.   java ... com.qxotic.llm.Lfm2CacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;

import java.nio.file.Path;

public final class Lfm2CacheRun {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        Harness<Lfm2.State> h = new Harness<>(Lfm2.loadModel(path, 4096), path, 4096);
        new CacheScenario<>(h, new CacheScenario.Config(
                "You are a concise assistant.", null, null, 0, 120, false)).run("Lfm2CacheRun");
    }
}
