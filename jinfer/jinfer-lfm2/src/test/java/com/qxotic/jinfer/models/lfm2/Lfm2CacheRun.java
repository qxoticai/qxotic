// PromptCache validation + benchmark on LFM2.5 (hybrid conv+attention+MoE) via the shared
// testkit scenario.   java ... com.qxotic.jinfer.models.lfm2.Lfm2CacheRun [model.gguf]
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class Lfm2CacheRun {
    @Test
    @Tag("driver")
    void run() throws Exception {
        main(testArgs());
    }

    private static String[] testArgs() {
        String argv = System.getProperty("jinfer.args", "");
        return argv.isBlank() ? new String[0] : argv.trim().split("\\s+");
    }

    private static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        Lfm2 m = Lfm2.loadModel(path, 4096);
        Harness<Lfm2.State> h =
                new Harness<>(
                        m.loaded(),
                        m.turnTemplate().orElseThrow(),
                        path,
                        4096,
                        false); // MoE: threaded decode is not byte-deterministic
        new CacheScenario<>(h, CacheScenario.Config.of("You are a concise assistant.", 120))
                .run("Lfm2CacheRun");
    }
}
