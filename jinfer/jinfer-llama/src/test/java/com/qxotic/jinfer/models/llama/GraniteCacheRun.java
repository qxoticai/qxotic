// PromptCache validation + benchmark on Granite 4.1 3B (uniform full attention - the degenerate
// codec) via the shared testkit scenario.   java ... com.qxotic.jinfer.models.llama.GraniteCacheRun
// [model.gguf]
package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.Stories;
import java.nio.file.Path;

public final class GraniteCacheRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/ibm-granite/granite-4.1-3b-Q8_0.gguf");
        Harness<Granite.State> h =
                new Harness<>(Granite.loadModel(path, 8192).loaded(), path, 8192);
        new CacheScenario<>(
                        h,
                        CacheScenario.Config.of(
                                "You are a concise assistant.",
                                60,
                                new CacheScenario.LongCase(
                                        Stories.expeditionLog(),
                                        "How many entries were there? One number.",
                                        1500)))
                .run("GraniteCacheRun");
    }
}
