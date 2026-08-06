// PromptCache validation + benchmark on Qwen3.5 (gated-delta-net hybrid, COARSE codec) via the
// shared testkit scenario - the byte-level gate (Harness.statesEqual) the adapter IT's
// text-equality cannot provide, plus the >512-token chunked-ingest cases.
//   java ... com.qxotic.jinfer.models.qwen35.Qwen35CacheRun [model.gguf]
package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class Qwen35CacheRun {
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
                Path.of(args.length > 0 ? args[0] : ModelFixture.QWEN35_2B_Q8.path().toString());
        Qwen35 m = Qwen35.loadModel(path, java.lang.foreign.Arena.ofAuto());
        Harness<Qwen35.State> h =
                new Harness<>(
                        m.loaded(),
                        m.turnTemplate().orElseThrow(),
                        path,
                        4096,
                        true); // the 2B is dense: decode is byte-deterministic
        new CacheScenario<>(h, CacheScenario.Config.of("You are a concise assistant.", 120))
                .run("Qwen35CacheRun");
    }
}
