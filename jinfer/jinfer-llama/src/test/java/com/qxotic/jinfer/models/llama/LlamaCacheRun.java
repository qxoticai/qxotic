// PromptCache validation + benchmark on Llama 1B (uniform full attention - the degenerate codec)
// via the shared testkit scenario.   java ... com.qxotic.jinfer.models.llama.LlamaCacheRun
// [model.gguf]
package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.Stories;
import java.nio.file.Path;

public final class LlamaCacheRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/unsloth/Llama-3.2-1B-Instruct-Q8_0.gguf");
        Harness<Llama.State> h = new Harness<>(Llama.loadModel(path, 8192).loaded(), path, 8192);
        new CacheScenario<>(
                        h,
                        CacheScenario.Config.of(
                                "You are a concise assistant.",
                                60,
                                new CacheScenario.LongCase(
                                        Stories.expeditionLog(),
                                        "How many entries were there? One number.",
                                        1500)))
                .run("LlamaCacheRun");
    }
}
