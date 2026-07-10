// PromptCache validation + benchmark on Qwen3.5 2B (hybrid gated-delta-net + periodic full
// attention) via the shared testkit scenario. The SSM checkpoint (conv ring + delta-net S matrix)
// is the interesting part; the byte-identity gate is the hard one.
//   java ... com.qxotic.jinfer.models.qwen35.Qwen35CacheRun [model.gguf]
package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.Stories;
import java.nio.file.Path;

public final class Qwen35CacheRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf");
        Harness<Qwen35.State> h =
                new Harness<>(
                        Qwen35.loadModel(path, 8192).loaded(),
                        path,
                        8192,
                        false); // jam threaded FFN-gemm is not run-deterministic on this arch (see
        // Qwen35PrefillCheck)
        new CacheScenario<>(
                        h,
                        CacheScenario.Config.of(
                                "You are a concise assistant.",
                                200,
                                new CacheScenario.LongCase(
                                        Stories.expeditionLog(),
                                        "How many entries were there? One number.",
                                        1500)))
                .run("Qwen35CacheRun");
    }
}
