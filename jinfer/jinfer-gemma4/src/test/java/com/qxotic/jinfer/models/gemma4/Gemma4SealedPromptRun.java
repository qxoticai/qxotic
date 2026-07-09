// SealedPrompt (use case A) validation on Gemma 4 E2B via the shared testkit scenario - the
// sealed span includes the SWA ring checkpoints, proving the single-span format handles windowed
// layers.   java ... com.qxotic.jinfer.models.gemma4.Gemma4SealedPromptRun [model.gguf]
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.SealedScenario;
import java.nio.file.Path;

public final class Gemma4SealedPromptRun {
    public static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        new SealedScenario<>(new Harness<>(Gemma4.loadModel(path, 8192), path, 8192))
                .run("Gemma4SealedPromptRun");
    }
}
