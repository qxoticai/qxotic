// PromptCache validation + benchmark on Nemotron-H (hybrid Mamba2+attention+MoE) via the shared
// testkit scenario. MoE: threaded decode is not byte-deterministic, so the hard gate is
// restored-vs-live byte identity. The ~47.6MB SSM checkpoint per block dictates small decode
// budgets and a large cache budget (-Djinfer.promptCacheMB, default 8192).
//   java ... com.qxotic.llm.NemotronHCacheRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.Stories;

import java.nio.file.Path;

public final class NemotronHCacheRun {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0]
                : "/home/mukel/Desktop/playground/models/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf");
        Harness<NemotronH.State> h = new Harness<>(NemotronH.loadModel(path, 4096), path, 4096, false);
        new CacheScenario<>(h, CacheScenario.Config.of("You are a concise assistant.", 32,
                new CacheScenario.LongCase(Stories.pelican(),
                        "What was the secret codeword in the story? One word.", 700)))
                .run("NemotronHCacheRun");
    }
}
