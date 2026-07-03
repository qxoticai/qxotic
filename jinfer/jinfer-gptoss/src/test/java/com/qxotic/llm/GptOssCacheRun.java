// PromptCache validation + benchmark on gpt-oss-20b (alternating SWA/full attention + sinks +
// all-MoE) via the shared testkit scenario. The long history wraps the 128-token window ~10x -
// the hardest ring-restore case. Harmony replies are logged tail-only (analysis channel first).
//   java -Xmx24g ... com.qxotic.llm.GptOssCacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.Stories;

import java.nio.file.Path;

public final class GptOssCacheRun {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/unsloth/gpt-oss-20b-Q8_0.gguf");
        Harness<GptOss.State> h = new Harness<>(GptOss.loadModel(path, 8192), path, 8192, false);   // all-MoE: not byte-deterministic
        new CacheScenario<>(h, CacheScenario.Config.of(null, 200,
                new CacheScenario.LongCase(Stories.pelican(), "What was the codeword at the start? One word.", 1280))
                .logTailOnly()).run("GptOssCacheRun");
    }
}
