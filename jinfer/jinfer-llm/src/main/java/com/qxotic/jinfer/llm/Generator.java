// New-API generation driver: prefill via Batch.prepare (the port chunks internally), decode via
// logits(state) + Sampler + Batch.step. Tokens in, tokens out - the loop knows sampling, token
// stops, the completion budget and the wall-clock deadline, and NOTHING else: no text, no
// reasoning, no tool calls, no billing. Reply STRUCTURE (think spans, tool-call spans, UTF-8
// assembly) is the chat layer's ReplyParser; string-level stop handling is TextStops; billing
// policy (BOS discounts, cache restores) is the server's; all are driven by the caller through
// the TokenSink. NOTE: prompt caching / prefix resume is not (yet) supported on this path - the
// cache path decodes from a resumed state via generate() with an empty prompt.
package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.*;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Set;

public final class Generator {

    private Generator() {}

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
        int contextLength = model.config().contextLength();
        int promptCount = prompt.stream().mapToInt(Batch::count).sum();
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

        boolean hasDeadline = timeoutNanos != 0;
        long deadlineNanos = hasDeadline ? System.nanoTime() + timeoutNanos : Long.MAX_VALUE;

        long startNanos = System.nanoTime();
        long[] prefillDoneNanos = {0};
        boolean[] aborted = {false};
        boolean[] deadlineHit = {false};
        IntSequence responseTokens;
        synchronized (model) { // generations on a shared model are strictly serialized
            responseTokens =
                    decodeLoop(
                            model,
                            state,
                            prompt,
                            stopTokens,
                            actualMaxTokens,
                            sampler,
                            token -> {
                                boolean keepGoing = sink == null || sink.onToken(token);
                                if (!keepGoing) aborted[0] = true;
                                if (System.nanoTime() >= deadlineNanos) {
                                    deadlineHit[0] = true;
                                    return false;
                                }
                                return keepGoing;
                            },
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
            long[] prefillDoneNanos) {
        int vocab = model.config().vocabularySize();
        int contextLength = model.config().contextLength();
        for (Batch batch : Batch.prepare(prompt, state.batchCapacity())) {
            model.ingest(state, batch); // the port chunks internally + runs the decode pool
        }
        IntSequence.Builder generated = IntSequence.newBuilder();
        while (generated.size() < maxNewTokens) {
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
        }
        return generated.build();
    }

    static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }
}
