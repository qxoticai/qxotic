// PromptCache validation + benchmark on Llama 1B (uniform full attention - the degenerate codec)
// via the shared testkit scenario.   java ... com.qxotic.llm.LlamaCacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;

import java.nio.file.Path;

public final class LlamaCacheRun {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/unsloth/Llama-3.2-1B-Instruct-Q8_0.gguf");
        Harness<Llama.State> h = new Harness<>(Llama.loadModel(path, 8192), path, 8192);
        StringBuilder story = new StringBuilder("Summarize the following notes.\n");
        for (int i = 0; i < 90; i++) {
            story.append("Entry ").append(i).append(": the expedition logged river depth, canopy density, ")
                 .append("and soil acidity at station ").append(i)
                 .append("; readings were nominal and the weather held clear through the afternoon.\n");
        }
        new CacheScenario<>(h, new CacheScenario.Config(
                "You are a concise assistant.", story.toString(),
                "How many entries were there? One number.", 1500, 60, false)).run("LlamaCacheRun");
    }
}
