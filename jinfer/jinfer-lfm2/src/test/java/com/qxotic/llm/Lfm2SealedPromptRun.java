// SealedPrompt (use case A) validation on LFM2.5 via the shared testkit scenario.
//   java ... com.qxotic.llm.Lfm2SealedPromptRun [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.SealedScenario;
import java.nio.file.Path;

public final class Lfm2SealedPromptRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        new SealedScenario<>(new Harness<>(Lfm2.loadModel(path, 8192), path, 8192))
                .run("Lfm2SealedPromptRun");
    }
}
