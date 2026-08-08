// FrozenBlocks TTFT benchmark: for a ~10k-token prompt on each common model, compare the cold
// full-prefill time-to-first-token against a frozen-artifact restore (open + resume + tail
// re-ingest + first argmax). Run:
//   mvn test -pl jinfer-bench -Dtest=FrozenTtftBench -Dsurefire.excludedGroups=
package com.qxotic.jinfer.bench;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.BlockTree;
import com.qxotic.jinfer.cache.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.models.gemma4.Gemma4;
import com.qxotic.jinfer.models.gptoss.GptOss;
import com.qxotic.jinfer.models.lfm2.Lfm2;
import com.qxotic.jinfer.models.llama.Granite;
import com.qxotic.jinfer.models.llama.Llama;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.IntSequence;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class FrozenTtftBench {

    static final int PROMPT_TOKENS = 10_000;
    static final int CTX = 12_288;
    static final long BUDGET = 32L << 30;

    @Test
    @Tag("bench")
    void run() throws Exception {
        bench(
                "LFM2.5-8B-A1B-Q8_0",
                ModelFixture.LFM25_8B_Q8.path(),
                p -> Lfm2.loadModel(p, Arena.ofAuto()).loaded());
        bench(
                "gemma-4-E2B-it-Q8_0",
                ModelFixture.GEMMA4_E2B_Q8.path(),
                p -> Gemma4.loadModel(p, Arena.ofAuto()).loaded());
        bench(
                "gpt-oss-20b-Q8_0",
                ModelFixture.GPTOSS_20B_Q8.path(),
                p -> GptOss.loadModel(p, Arena.ofAuto()).loaded());
        bench(
                "Llama-3.2-1B-Instruct-Q8_0",
                ModelFixture.LLAMA32_1B_Q8.path(),
                p -> Llama.loadModel(p, Arena.ofAuto()).loaded());
        bench(
                "granite-4.1-3b-Q8_0",
                ModelFixture.GRANITE_41_3B_Q8.path(),
                p -> Granite.loadModel(p, Arena.ofAuto()).loaded());
    }

    interface Loader {
        LoadedModel<?> load(Path gguf) throws Exception;
    }

    void bench(String name, Path gguf, Loader loader) throws Exception {
        if (!Files.exists(gguf)) {
            System.out.printf("%-28s SKIP (model not found)%n", name);
            return;
        }
        benchTyped(name, gguf, loader.load(gguf));
    }

    <S extends RuntimeState> void benchTyped(String name, Path gguf, LoadedModel<S> m)
            throws Exception {
        // ~10k-token prompt from repeated story text
        String story = TtftBench.story();
        StringBuilder text = new StringBuilder();
        while (m.tokenizer().encode(text.toString()).length() < PROMPT_TOKENS + 64) {
            text.append(story).append('\n');
        }
        IntSequence all = m.tokenizer().encode(text.toString());
        int[] prompt = all.subSequence(0, PROMPT_TOKENS).toArray();

        // COLD: plain full prefill + first argmax (no cache anywhere); skippable when the
        // baseline is already known (-Djinfer.skipCold=true)
        boolean skipCold = Boolean.getBoolean("jinfer.skipCold");
        int coldTok = -1;
        double coldMs = Double.NaN;
        if (!skipCold) {
            S cold = m.model().newState(CTX, 512);
            long t0 = System.nanoTime();
            for (Batch b : Batch.prepare(List.of(Batch.prefill(prompt)), 512)) {
                m.model().ingest(cold, b);
            }
            coldTok = m.model().logits(cold).argmax();
            coldMs = (System.nanoTime() - t0) / 1e6;
        }

        // COMPILE low-level ON PURPOSE: the bench compares against the cold prefill at byte
        // level, so the artifact must be chunked exactly like the cold ingest (uniform 512
        // chunks), with the final token split as its own block so the N-1 serve below re-ingests
        // exactly one token. The facade's define() covers production; this pins the byte shape.
        byte[] seed = m.seed();
        Path artifact = Files.createTempFile("frozen-" + name, ".jkv");
        artifact.toFile().deleteOnExit();
        long tFreeze = System.nanoTime();
        BlockTree<S> build = new BlockTree<>(m.codec(), CacheStore.inMemory(), BUDGET, seed);
        CachedSession<S> compile =
                CachedSession.start(m.model(), build, m.model().newState(CTX, 512));
        compile.ingest(List.of(Batch.prefill(Arrays.copyOf(prompt, prompt.length - 1))));
        compile.ingest(List.of(Batch.prefill(new int[] {prompt[prompt.length - 1]})));
        build.freeze(artifact);
        double freezeMs = (System.nanoTime() - tFreeze) / 1e6;

        // FROZEN: open + resume (to the deepest boundary below the tip) + tail + first argmax
        long t1 = System.nanoTime();
        FrozenBlocks frozen = FrozenBlocks.open(artifact, seed);
        double openMs = (System.nanoTime() - t1) / 1e6;
        S state = m.model().newState(CTX, 512);
        long t2 = System.nanoTime();
        BlockTree<S> graft = new BlockTree<>(m.codec(), CacheStore.inMemory(), 0, seed, frozen);
        CachedSession<S> w =
                CachedSession.resume(
                        m.model(), graft, state, List.of(Batch.prefill(prompt)), prompt.length - 1);
        double resumeMs = (System.nanoTime() - t2) / 1e6;
        int restored = w.position();
        long t3 = System.nanoTime();
        int[] tail = Arrays.copyOfRange(prompt, restored, prompt.length);
        w.ingest(List.of(Batch.prefill(tail)));
        int frozenTok = m.model().logits(state).argmax();
        double tailMs = (System.nanoTime() - t3) / 1e6;
        double frozenTotal = openMs + resumeMs + tailMs;

        System.out.printf(
                "%-28s cold %,8.0f ms   frozen %,7.1f ms (open %.2f + restore %.1f [%d pos] +"
                        + " tail %.1f [%d pos])   artifact %d MB (freeze %,.0f ms)   token %s%n",
                name,
                coldMs,
                frozenTotal,
                openMs,
                resumeMs,
                restored,
                tailMs,
                prompt.length - restored,
                Files.size(artifact) >> 20,
                freezeMs,
                skipCold || coldTok == frozenTok ? "OK" : "DIFFERS");
    }
}
