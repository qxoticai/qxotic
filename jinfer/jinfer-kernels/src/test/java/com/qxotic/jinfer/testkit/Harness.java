// Shared driver core for the per-model validation harnesses (cache/sealed/frozen/oracle): one
// model + its S1 seams (turnTemplate/kvCodec) + the check/finish protocol every run uses. The
// scenarios (CacheScenario, SealedScenario, FrozenScenario, OracleScenario) are parameterized by
// this; per-model entry points shrink to wiring.
package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.GgufTokenizer;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.KvCodec;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Harness<S extends RuntimeState> {

    public final LanguageModel<?, ?, S> model;
    public final TurnTemplate template;
    public final KvCodec<S> codec;
    public final GgufTokenizer tokenizer;
    public final Set<Integer> stops;
    public final Path path;
    public final byte[] seed;
    public final int ctx;
    /** Whether this model's greedy decode is byte-deterministic under load (dense models: yes;
     *  threaded-MoE reductions: no) - declared once per model, scenarios gate reply-text equality
     *  on it and always gate resume-state byte-identity. */
    public final boolean deterministicDecode;
    private final Checks checks = new Checks();

    public Harness(LanguageModel<?, ?, S> model, Path path, int ctx) {
        this(model, path, ctx, true);
    }

    public Harness(LanguageModel<?, ?, S> model, Path path, int ctx, boolean deterministicDecode) {
        this.deterministicDecode = deterministicDecode;
        this.model = model;
        this.template = model.turnTemplate().orElseThrow();
        this.codec = model.kvCodec().orElseThrow();
        this.tokenizer = model.tokenizer();
        this.stops = model.stopTokens();
        this.path = path;
        this.seed = PromptCache.modelSeed(path);
        this.ctx = ctx;
    }

    public S newState() {
        return model.newState(ctx, 512);
    }

    /** A decoded reply: the text plus how many tokens produced it (for tok/s). */
    public record Reply(String text, int tokens) {}

    /** Open the assistant turn, greedy-decode (each step a single-token block), close the turn. */
    public Reply decode(CachedSession<S> s, int maxTokens) {
        s.ingest(template.generationPrompt(true));
        Reply reply = greedy(s.state(), s::step, maxTokens);
        s.ingest(template.closeTurn());
        return reply;
    }

    /** Ingest the user turn + generation prompt on a cache-less state, greedy-decode the reply. */
    public String serve(S state, Message user, int maxTokens) {
        ingest(state, template.encodeTurn(user));
        ingest(state, template.generationPrompt(true));
        return greedy(state, tok -> model.ingest(state, Batch.step(tok)), maxTokens).text();
    }

    /** The shared greedy loop: argmax, feed each token through {@code step}, stop on the model's
     *  stop set or the budget. */
    private Reply greedy(S state, java.util.function.IntConsumer step, int maxTokens) {
        StringBuilder out = new StringBuilder();
        int tok = model.logits(state).argmax();
        int n = 0;
        for (; n < maxTokens && !stops.contains(tok); n++) {
            out.append(tokenizer.decode(tok));
            step.accept(tok);
            tok = model.logits(state).argmax();
        }
        return new Reply(out.toString(), Math.max(n, 1));
    }

    /** Plain chunked ingest (no cache); returns the ingested fingerprints. */
    public long[] ingest(S state, List<Batch> batches) {
        List<Batch> prepared = Batch.prepare(batches, state.batchCapacity());
        for (Batch b : prepared) model.ingest(state, b);
        int[] ids = Batch.tokenIds(prepared);
        long[] fp = new long[ids.length];
        for (int i = 0; i < fp.length; i++) fp[i] = ids[i];
        return fp;
    }

    /** Resume-state equality through the codec (model-agnostic): serialize both states' resume
     *  state for {@code [0,positions)} and byte-compare - exactly the bytes a cache block holds.
     *  This is the sound cache-identity gate; reply-text equality is load-sensitive for MoE
     *  models (threaded reductions are not byte-deterministic, near-tie greedy picks can flip). */
    public boolean statesEqual(S a, S b, int positions) {
        long rowBytes = codec.rowBytes(positions);
        long bytes = rowBytes + codec.checkpointBytes();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment sa = arena.allocate(bytes, 64);
            MemorySegment sb = arena.allocate(bytes, 64);
            codec.saveRows(a, 0, positions, sa);
            codec.saveCheckpoint(a, positions, sa.asSlice(rowBytes));
            codec.saveRows(b, 0, positions, sb);
            codec.saveCheckpoint(b, positions, sb.asSlice(rowBytes));
            return MemorySegment.mismatch(sa, 0, bytes, sb, 0, bytes) == -1;
        }
    }

    public void check(boolean ok, String what) {
        checks.check(ok, what);
    }

    /** Prints the verdict and exits non-zero on any failed check. */
    public void finish(String name) {
        checks.finish(name, "all checks passed");
    }

    // ---- shared shapes ----

    @SafeVarargs
    public static List<Batch> concat(List<Batch>... groups) {
        List<Batch> out = new ArrayList<>();
        for (List<Batch> g : groups) out.addAll(g);
        return out;
    }

    public static long[] flatten(List<Batch> batches) {
        int[] ids = Batch.tokenIds(batches);
        long[] fp = new long[ids.length];
        for (int i = 0; i < fp.length; i++) fp[i] = ids[i];
        return fp;
    }

    public static long[] concatFp(long[] a, long[] b) {
        long[] out = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    public static int[] toInts(long[] fp) {
        int[] ids = new int[fp.length];
        for (int i = 0; i < ids.length; i++) ids[i] = (int) fp[i];
        return ids;
    }

    /** Long replies (e.g. Harmony analysis channels): show just the final stretch for logs. */
    public static String tail(String reply) {
        String r = reply.strip().replace("\n", " ");
        return r.length() > 160 ? "..." + r.substring(r.length() - 160) : r;
    }
}
