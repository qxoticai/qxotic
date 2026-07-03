// PromptCache validation + benchmark on Qwen3.5 2B (hybrid gated-delta-net + periodic full
// attention) via the shared testkit scenario. The SSM checkpoint (conv ring + delta-net S matrix)
// is the interesting part; the byte-identity gate is the hard one.
//   java ... com.qxotic.llm.Qwen35CacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;

import java.nio.file.Path;

public final class Qwen35CacheRun {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf");
        Harness<Qwen35.State> h = new Harness<>(Qwen35.loadModel(path, 8192), path, 8192, false);   // jam threaded FFN-gemm is not run-deterministic on this arch (see Qwen35PrefillCheck)
        StringBuilder story = new StringBuilder("Summarize the following notes.\n");
        for (int i = 0; i < 90; i++) {
            story.append("Entry ").append(i).append(": the expedition logged river depth, canopy density, ")
                 .append("and soil acidity at station ").append(i)
                 .append("; readings were nominal and the weather held clear through the afternoon.\n");
        }
        new CacheScenario<>(h, CacheScenario.Config.of("You are a concise assistant.", 200,
                new CacheScenario.LongCase(story.toString(), "How many entries were there? One number.", 1500)))
                .run("Qwen35CacheRun");
    }
}
