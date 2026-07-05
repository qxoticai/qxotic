package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Greedy self-speculative decode over the Gemma 4 MTP draft head (Stage 3). Per iteration: chain
 * {@code depth} greedy drafts through {@link Gemma4MtpDecoder} (no backbone forwards), then verify
 * them in ONE backbone ALL-outputs batch and accept the longest matching prefix; the rejected tail
 * is rolled back with {@link com.qxotic.jinfer.RuntimeState#resumeAt} (stale KV rows are overwritten
 * by the next append, sound under the causal mask).
 *
 * <p>Every emitted token is the backbone's own greedy argmax read from a verified row, so the output
 * is token-identical to plain greedy BY CONSTRUCTION; the draft only decides how many backbone
 * forwards it takes. The verify batch always starts with the exact next token (known from the
 * previous row's logits), so each iteration commits at least one correct token.
 *
 * <p>Single-threaded, like the decoder it drives. The caller has already ingested the prompt
 * (token batches - the row->token map {@code state.lastTokens} seeds the first draft).
 */
public final class Gemma4Speculative {

    /** Emitted tokens (stop-truncated, exclusive) + all committed tokens (what the KV actually holds,
     *  for {@link com.qxotic.jinfer.cache.CachedSession#adopt}), and draft statistics. */
    public record Result(List<Integer> tokens, List<Integer> committed, int drafted, int accepted, int forwards) {}

    /** Diagnostic hook: for every emitted token, the top-2 of the verify row that produced it —
     *  the spec-side half of a near-tie analysis (the lockstep oracle supplies the other half). */
    public interface TopRecorder {
        void onEmit(int token, int top1, float top1Logit, int top2, float top2Logit);
    }

    static boolean DEBUG = Boolean.getBoolean("jinfer.mtpDebug");

    private Gemma4Speculative() {}

    public static Result generate(Gemma4 model, Gemma4.State s, int maxTokens, Set<Integer> stops, int depth) {
        return generate(model, s, maxTokens, stops, depth, null);
    }

    public static Result generate(Gemma4 model, Gemma4.State s, int maxTokens, Set<Integer> stops, int depth,
                                  TopRecorder recorder) {
        Gemma4MtpDecoder decoder = model.mtpDecoder();
        if (decoder == null) throw new IllegalStateException("MTP sidecar not loaded - use loadModel(gguf, ctx, mtpSidecar)");
        int dim = model.config().embeddingLength();
        int vocab = model.config().vocabularySize();

        List<Integer> emitted = new ArrayList<>();
        List<Integer> committed = new ArrayList<>();
        int drafted = 0, acceptedTotal = 0, forwards = 0;

        // seed from the last ingested row: its token, its hidden, and the exact next token
        int lastRow = s.lastChunkLen - 1;
        int tLast = s.lastTokens[lastRow];
        F32FloatTensor h = F32FloatTensor.allocate(dim);
        s.residual.copyTo((long) lastRow * dim, h, 0, dim);
        FloatTensor promptLogits = model.logits(s, s.outputCount - 1);
        int next = promptLogits.argmax(0, vocab);
        float[] pending = recorder != null ? top2(promptLogits, vocab) : null;   // stats of the row that produced `next`
        float[][] rowStats = recorder != null ? new float[depth + 1][] : null;

        int[] cand = new int[depth + 1];
        while (emitted.size() < maxTokens) {
            // draft chain: warm-up pairs (h, tLast) at tLast's position; heads chain greedily from `next`
            int pos = s.position() - 1;
            decoder.draft(s, h, 0, tLast, pos);
            cand[0] = next;
            int dTok = next;
            for (int i = 1; i <= depth; i++) {
                FloatTensor dl = decoder.draft(s, decoder.chainedHidden(), 0, dTok, pos);
                dTok = dl.argmax(0, vocab);
                cand[i] = dTok;
            }
            drafted += depth;

            // verify all candidates in one backbone forward (ALL outputs), then walk the rows greedily
            int basePos = s.position();
            model.ingest(s, Batch.score(cand));
            forwards++;
            int accepted = 0;                 // drafts confirmed beyond cand[0]
            int nextAfter = -1;
            while (accepted < depth) {
                FloatTensor rl = model.logits(s, accepted);
                if (rowStats != null) rowStats[accepted] = top2(rl, vocab);
                int trueNext = rl.argmax(0, vocab);
                if (cand[accepted + 1] == trueNext) accepted++;
                else { nextAfter = trueNext; break; }
            }
            if (nextAfter < 0) {
                FloatTensor rl = model.logits(s, accepted);
                if (rowStats != null) rowStats[accepted] = top2(rl, vocab);
                nextAfter = rl.argmax(0, vocab);
            }
            acceptedTotal += accepted;
            if (DEBUG) {
                // re-derive each accepted token's predecessor-row argmax from THIS score batch
                System.out.printf("  iter base=%d cand=%s accepted=%d nextAfter=%d%n",
                        basePos, java.util.Arrays.toString(cand), accepted, nextAfter);
                for (int i = 0; i < accepted; i++) {
                    int tn = model.logits(s, i).argmax(0, vocab);
                    if (tn != cand[i + 1]) System.out.printf("    !! row %d argmax=%d but cand[%d]=%d (accepted anyway?)%n", i, tn, i + 1, cand[i + 1]);
                }
            }

            // extract the next iteration's seed BEFORE rollback (residual/logits are chunk scratch)
            s.residual.copyTo((long) accepted * dim, h, 0, dim);
            tLast = cand[accepted];
            next = nextAfter;

            // stop handling first, so the KV keep-count and `committed` stay in exact lockstep:
            // keep everything up to and including a stop (it was verified), else the accepted prefix
            int stopIdx = -1;
            for (int i = 0; i <= accepted && stopIdx < 0; i++) {
                if (stops.contains(cand[i])) stopIdx = i;
            }
            int keep = stopIdx >= 0 ? stopIdx + 1 : accepted + 1;
            s.resumeAt(basePos + keep);                  // keep cand[0..keep), drop the rest
            for (int i = 0; i < keep; i++) committed.add(cand[i]);
            for (int i = 0; i < keep && (stopIdx < 0 || i < stopIdx); i++) {
                emitted.add(cand[i]);
                if (recorder != null) {
                    float[] st = i == 0 ? pending : rowStats[i - 1];   // row that produced cand[i]
                    recorder.onEmit(cand[i], (int) st[0], st[1], (int) st[2], st[3]);
                }
                if (emitted.size() >= maxTokens) return new Result(emitted, committed, drafted, acceptedTotal, forwards);
            }
            if (recorder != null) pending = rowStats[accepted];        // the row that produced `next` (= nextAfter)
            if (stopIdx >= 0) return new Result(emitted, committed, drafted, acceptedTotal, forwards);
        }
        return new Result(emitted, committed, drafted, acceptedTotal, forwards);
    }

    /** argmax + runner-up of a logits row: {top1, top1Logit, top2, top2Logit}. */
    static float[] top2(FloatTensor logits, int vocab) {
        int i1 = -1, i2 = -1;
        float l1 = Float.NEGATIVE_INFINITY, l2 = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            float v = logits.getFloat(i);
            if (v > l1) { i2 = i1; l2 = l1; i1 = i; l1 = v; }
            else if (v > l2) { i2 = i; l2 = v; }
        }
        return new float[]{i1, l1, i2, l2};
    }
}
