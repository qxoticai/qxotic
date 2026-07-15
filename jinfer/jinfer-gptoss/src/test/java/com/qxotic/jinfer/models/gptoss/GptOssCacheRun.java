// PromptCache validation + benchmark on gpt-oss-20b (alternating SWA/full attention + sinks +
// all-MoE) via the shared testkit scenario. The long history wraps the 128-token window ~10x -
// the hardest ring-restore case. Harmony replies are logged tail-only (analysis channel first).
//   java -Xmx24g ... com.qxotic.jinfer.models.gptoss.GptOssCacheRun [model.gguf]
package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.Stories;
import java.nio.file.Path;

public final class GptOssCacheRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/unsloth/gpt-oss-20b-Q8_0.gguf");
        GptOss m = GptOss.loadModel(path, 8192);
        Harness<GptOss.State> h =
                new Harness<>(
                        m.loaded(),
                        m.turnTemplate().orElseThrow(),
                        path,
                        8192,
                        false); // all-MoE: not byte-deterministic
        new CacheScenario<>(
                        h,
                        CacheScenario.Config.of(
                                        null,
                                        200,
                                        new CacheScenario.LongCase(
                                                Stories.pelican(),
                                                "What was the codeword at the start? One word.",
                                                1280))
                                .logTailOnly())
                .run("GptOssCacheRun");
    }
}
