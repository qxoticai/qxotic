// PromptCache validation + benchmark on NemotronH (Mamba2/attention hybrid, COARSE codec) via the
// shared testkit scenario - the byte-level gate (Harness.statesEqual) the adapter IT's
// text-equality cannot provide, plus the >512-token chunked-ingest cases.
//   java ... com.qxotic.jinfer.models.nemotronh.NemotronHCacheRun [model.gguf]
package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class NemotronHCacheRun {
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
                Path.of(args.length > 0 ? args[0] : ModelFixture.NEMOTRON_30B_Q8.path().toString());
        NemotronH m = NemotronH.loadModel(path, Arena.ofAuto());
        // deterministicDecode=false: restore==live is byte-exact (gated strictly below), but
        // cross-CHUNKING states drift an ulp (chunked ingest vs one-shot prefill - generic to the
        // hybrids, same probe result on Qwen3.5) and this model's greedy trajectory hits an
        // argmax tie inside 120 tokens, so resumed-vs-replayed reply text is not a law here
        Harness<NemotronH.State> h =
                new Harness<>(m.loaded(), m.turnTemplate().orElseThrow(), path, 4096, false);
        new CacheScenario<>(h, CacheScenario.Config.of("You are a concise assistant.", 120))
                .run("NemotronHCacheRun");
    }
}
