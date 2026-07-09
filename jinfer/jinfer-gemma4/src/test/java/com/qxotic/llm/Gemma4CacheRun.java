// PromptCache validation + benchmark on Gemma 4 E2B (interleaved SWA rings + full attention +
// shared KV tail) via the shared testkit scenario. The long history exceeds the 512-token window
// so the SWA ring checkpoints actually wrap - the case where ring-slot restore bugs show.
//   java ... com.qxotic.llm.Gemma4CacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.Stories;
import java.nio.file.Path;

public final class Gemma4CacheRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        Harness<Gemma4.State> h = new Harness<>(Gemma4.loadModel(path, 8192), path, 8192);
        new CacheScenario<>(
                        h,
                        CacheScenario.Config.of(
                                null,
                                60,
                                new CacheScenario.LongCase(
                                        Stories.pelican(),
                                        "What was the codeword at the start? One word.",
                                        1024)))
                .run("Gemma4CacheRun");
    }
}
