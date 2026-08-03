package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.telemetry.DecodeEvent;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Set;

/**
 * The generation loop: prefill, then sample and decode until a stop token, the completion budget or
 * the wall-clock deadline.
 *
 * <p>It knows TOKENS and nothing else. Reply structure - think spans, tool-call spans, UTF-8
 * assembly - is the chat layer's {@code ReplyParser}; string-level stops are {@code TextStops};
 * billing policy is the server's. All of them observe this loop through the token sink rather than
 * living inside it.
 */
public final class Generator {

    private Generator() {}

    /** See {@link LanguageModel#stateFor} - the policy's home; kept here for its many callers. */
    public static <S extends RuntimeState> S stateFor(LanguageModel<?, ?, S> model, int promptLen) {
        return model.stateFor(promptLen);
    }

    /**
     * The generated-token stream: sees EVERY sampled token in order, the trailing stop token
     * included, before the loop acts on it. Return false to abort the pass (the aborting token is
     * recorded but not ingested; finishReason "abort").
     */
    @FunctionalInterface
    public interface TokenSink {
        boolean onToken(int token);
    }

    /**
     * The outcome of one generation pass - only what the loop alone knows. {@code tokens} are the
     * generated tokens excluding the trailing stop token, reported in {@code stopToken} (-1 when
     * not ended by a stop token). {@code finishReason}: "stop" for a stop token, "length" for the
     * budget or deadline, "abort" when the sink ended the pass. Durations are exact {@link
     * System#nanoTime} deltas; consumers convert at their display edge.
     */
    public record GenerationResult(
            IntSequence tokens,
            int stopToken,
            String finishReason,
            long promptNanos,
            long predictedNanos) {

        public int completionTokens() {
            return tokens.length();
        }
    }

    /**
     * As {@link #generate(LanguageModel, RuntimeState, List, Sampler, int, long, Set, TokenSink)}
     * for a plain token prompt.
     */
    public static <S extends RuntimeState> GenerationResult generate(
            LanguageModel<?, ?, S> model,
            S state,
            IntSequence promptTokens,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Set<Integer> stopTokens,
            TokenSink sink) {
        List<Batch> prompt =
                promptTokens.isEmpty() ? List.of() : List.of(Batch.prefill(promptTokens.toArray()));
        return generate(model, state, prompt, sampler, maxTokens, timeoutNanos, stopTokens, sink);
    }

    /**
     * One generation pass: ingest {@code prompt} (token and media-embedding batches) at the state's
     * cursor, then decode until a stop token, an aborting sink, the wall-clock deadline ({@code
     * timeoutNanos} as a duration; 0 = none), or the completion budget ({@code maxTokens}; negative
     * = as much as the context allows). The state is caller-owned; a fresh state generates from the
     * prompt, a resumed state continues from its position (an empty prompt samples directly from
     * the retained logits). Generations on a shared model are serialized.
     */
    public static <S extends RuntimeState> GenerationResult generate(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Set<Integer> stopTokens,
            TokenSink sink) {
        return generate(
                model, state, prompt, sampler, maxTokens, timeoutNanos, stopTokens, sink, null);
    }

    /**
     * As above with a step-time hook: {@code afterIngest} fires right after each decode token is
     * INGESTED - the state's frontier includes it, which is what per-position cache accounting
     * needs (a commit must save at the frontier: ring rows alias and residues move). Never fired
     * for the final sampled token, which the loop does not ingest. An explicit parameter rather
     * than a sink default method, so no wrapper can silently drop it.
     */
    public static <S extends RuntimeState> GenerationResult generate(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Set<Integer> stopTokens,
            TokenSink sink,
            java.util.function.IntConsumer afterIngest) {
        int contextLength = model.config().contextLength();
        int promptCount = Batch.positions(prompt);
        int promptPositions = state.position() + promptCount;
        require(
                promptPositions <= contextLength,
                "Prompt exceeds context length (%d tokens used, %d available)",
                promptPositions,
                contextLength);
        int actualMaxTokens =
                maxTokens < 0
                        ? contextLength - promptPositions
                        : Math.min(maxTokens, contextLength - promptPositions);

        long deadlineNanos = timeoutNanos != 0 ? System.nanoTime() + timeoutNanos : Long.MAX_VALUE;
        try {
            return generationPass(
                    model,
                    state,
                    prompt,
                    sampler,
                    stopTokens,
                    sink,
                    afterIngest,
                    actualMaxTokens,
                    deadlineNanos);
        } finally {
            // Weights live in an automatic-arena mapping and kernels read them via raw addresses
            // (FloatTensor.GLOBAL_SEGMENT), which the GC cannot see: this fence pins the model -
            // and through it the mapping - for the whole pass, so the Cleaner can never unmap
            // under a running kernel even if the caller drops its reference mid-call.
            java.lang.ref.Reference.reachabilityFence(model);
        }
    }

    private static <S extends RuntimeState> GenerationResult generationPass(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Sampler sampler,
            Set<Integer> stopTokens,
            TokenSink sink,
            java.util.function.IntConsumer afterIngest,
            int actualMaxTokens,
            long deadlineNanos) {
        BaseState base = (BaseState) state;
        base.enter(); // the single-mutator contract, held across the whole generation
        try {
            return guardedPass(
                    model,
                    state,
                    prompt,
                    sampler,
                    stopTokens,
                    sink,
                    afterIngest,
                    actualMaxTokens,
                    deadlineNanos);
        } finally {
            base.exit();
        }
    }

    private static <S extends RuntimeState> GenerationResult guardedPass(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Sampler sampler,
            Set<Integer> stopTokens,
            TokenSink sink,
            java.util.function.IntConsumer afterIngest,
            int actualMaxTokens,
            long deadlineNanos) {
        boolean hasDeadline = deadlineNanos != Long.MAX_VALUE;
        long startNanos = System.nanoTime();
        long[] prefillDoneNanos = {0};
        boolean[] aborted = {false};
        boolean[] deadlineHit = {false};
        IntSequence responseTokens;
        TokenSink guarded =
                token -> {
                    boolean keepGoing = sink == null || sink.onToken(token);
                    if (!keepGoing) aborted[0] = true;
                    if (System.nanoTime() >= deadlineNanos) {
                        deadlineHit[0] = true;
                        return false;
                    }
                    return keepGoing;
                };
        synchronized (model) { // generations on a shared model are strictly serialized
            responseTokens =
                    decodeLoop(
                            model,
                            state,
                            prompt,
                            stopTokens,
                            actualMaxTokens,
                            sampler,
                            guarded,
                            afterIngest,
                            prefillDoneNanos);
        }
        long endNanos = System.nanoTime();
        long boundary = prefillDoneNanos[0] != 0 ? prefillDoneNanos[0] : endNanos;

        int stopToken = -1;
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            stopToken = responseTokens.getLast();
            responseTokens = responseTokens.subSequence(0, responseTokens.length() - 1);
        }
        String finishReason =
                stopToken >= 0
                        ? "stop"
                        : deadlineHit[0] || responseTokens.length() >= actualMaxTokens
                                ? "length"
                                : aborted[0] ? "abort" : "stop";
        return new GenerationResult(
                responseTokens,
                stopToken,
                finishReason,
                boundary - startNanos,
                endNanos - boundary);
    }

    /**
     * Prefill the prompt (prepared to the state's batch capacity), then decode one token at a time
     * via logits(state) + sampler + Batch.step until a stop token, an aborting sink, or the budget.
     * {@code onTokenGenerated} returning false aborts (the aborting token is recorded but not
     * ingested).
     */
    private static <S extends RuntimeState> IntSequence decodeLoop(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Set<Integer> stopTokens,
            int maxNewTokens,
            Sampler sampler,
            TokenSink onTokenGenerated,
            java.util.function.IntConsumer afterIngest,
            long[] prefillDoneNanos) {
        int vocab = model.config().vocabularySize();
        int contextLength = model.config().contextLength();
        for (Batch batch : Batch.prepare(prompt, state.batchCapacity())) {
            model.ingest(state, batch); // the port chunks internally + runs the decode pool
        }
        IntSequence.Builder generated = IntSequence.newBuilder();
        // Resolved ONCE, not per token: this is the hot loop, and allocating a per-token event
        // only to find it disabled would let telemetry perturb the thing it measures. A recording
        // started mid-generation is picked up by the next generation, which is soon enough.
        boolean traceTokens = new DecodeEvent().isEnabled();
        while (generated.size() < maxNewTokens) {
            DecodeEvent decode = traceTokens ? new DecodeEvent() : null;
            if (decode != null) decode.begin();
            try {
                FloatTensor logits =
                        model.logits(state); // last retained row; ports run this on the decode pool
                if (prefillDoneNanos[0] == 0)
                    prefillDoneNanos[0] = System.nanoTime(); // time-to-first-token boundary
                int nextToken = sampler.sampleToken(logits);
                if (nextToken < 0 || nextToken >= vocab) {
                    throw new IllegalArgumentException(
                            "sampler returned token id "
                                    + nextToken
                                    + " out of range [0, "
                                    + vocab
                                    + ")");
                }
                generated.add(nextToken);
                boolean keepGoing = onTokenGenerated == null || onTokenGenerated.onToken(nextToken);
                if (stopTokens.contains(nextToken) || !keepGoing) break;
                if (generated.size() >= maxNewTokens || state.position() >= contextLength) break;
                model.ingest(state, Batch.step(nextToken));
                if (afterIngest != null) afterIngest.accept(nextToken);
            } finally {
                if (decode != null) {
                    decode.end();
                    decode.commit();
                }
            }
        }
        return generated.build();
    }

    static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }
}
