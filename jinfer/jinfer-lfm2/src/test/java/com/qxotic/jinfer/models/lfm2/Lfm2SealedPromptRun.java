// SealedPrompt (use case A) validation on LFM2.5 via the shared testkit scenario.
//   java ... com.qxotic.jinfer.models.lfm2.Lfm2SealedPromptRun [model.gguf]
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.llm.*;
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
        Lfm2 m = Lfm2.loadModel(path, 8192);
        new SealedScenario<>(new Harness<>(m.loaded(), m.turnTemplate().orElseThrow(), path, 8192))
                .run("Lfm2SealedPromptRun");
    }
}
