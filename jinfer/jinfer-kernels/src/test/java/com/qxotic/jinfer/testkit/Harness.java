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
    public int lastReplyTokens;
    private int failures;

    public Harness(LanguageModel<?, ?, S> model, Path path, int ctx) {
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

    /** Open the assistant turn, greedy-decode (each step a single-token block), close the turn. */
    public String decode(CachedSession<S> s, int maxTokens) {
        s.ingest(template.generationPrompt(true));
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s.state()).argmax();
        int n = 0;
        for (; n < maxTokens && !stops.contains(tok); n++) {
            out.append(tokenizer.decode(tok));
            s.step(tok);
            tok = model.logits(s.state()).argmax();
        }
        lastReplyTokens = Math.max(n, 1);
        s.ingest(template.closeTurn());
        return out.toString();
    }

    /** Plain chunked ingest (no cache); returns the ingested fingerprints. */
    public long[] ingest(S state, List<Batch> batches) {
        List<Long> fp = new ArrayList<>();
        for (Batch b : Batch.prepare(batches, state.batchCapacity())) {
            model.ingest(state, b);
            for (int id : ((Batch.Input.Tokens) b.input()).ids()) fp.add((long) id);
        }
        long[] a = new long[fp.size()];
        for (int i = 0; i < a.length; i++) a[i] = fp.get(i);
        return a;
    }

    /** Ingest the user turn + generation prompt on a cache-less state, greedy-decode the reply. */
    public String serve(S state, Message user, int maxTokens) {
        ingest(state, template.encodeTurn(user));
        ingest(state, template.generationPrompt(true));
        StringBuilder out = new StringBuilder();
        int tok = model.logits(state).argmax();
        for (int n = 0; n < maxTokens && !stops.contains(tok); n++) {
            out.append(tokenizer.decode(tok));
            model.ingest(state, Batch.step(tok));
            tok = model.logits(state).argmax();
        }
        return out.toString();
    }

    public void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok:   " + what);
        } else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }

    /** Prints the verdict and exits non-zero on any failed check. */
    public void finish(String name) {
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println(name + ": all checks passed");
    }

    // ---- shared shapes ----

    @SafeVarargs
    public static List<Batch> concat(List<Batch>... groups) {
        List<Batch> out = new ArrayList<>();
        for (List<Batch> g : groups) out.addAll(g);
        return out;
    }

    public static long[] flatten(List<Batch> batches) {
        List<Integer> ids = new ArrayList<>();
        for (Batch b : batches) for (int id : ((Batch.Input.Tokens) b.input()).ids()) ids.add(id);
        long[] fp = new long[ids.size()];
        for (int i = 0; i < fp.length; i++) fp[i] = ids.get(i);
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
