// New-API generation driver: prefill via Batch.prepare (the port chunks internally), decode via
// logits(state) + Sampler + Batch.step. Tokens in, tokens out - the loop knows sampling, token
// stops, the completion budget and the wall-clock deadline, and NOTHING else: no text, no
// reasoning, no tool calls. Reply STRUCTURE (think spans, tool-call spans, UTF-8 assembly) is the
// chat layer's ReplyDecoder; string-level stop handling is TextStops; both are driven by the
// caller through the TokenSink. NOTE: prompt caching / prefix resume is not (yet) supported on
// this path - the cache path decodes from a resumed state via generate() with an empty prompt.
package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.*;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Map;
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
     * Sampling and limit parameters for one generation pass. The sampler is supplied by the caller
     * (see {@link #configuredSampler}) so interactive frontends can keep one RNG across turns.
     * {@code maxTokens} is the completion budget in generated tokens; negative means "as much as
     * the context allows". {@code timeoutNanos} is a wall-clock decode deadline as a DURATION
     * (converted to an absolute deadline at entry); 0 means no deadline. {@code stopTokens} end
     * generation (the stop token is reported but excluded from the result tokens).
     */
    public record Params(
            Sampler sampler, int maxTokens, long timeoutNanos, Set<Integer> stopTokens) {}

    /**
     * The outcome of one generation pass. {@code tokens} are the generated tokens excluding the
     * trailing stop token, reported in {@code stopToken} (-1 when not ended by a stop token).
     * {@code promptTokens} is the raw prompt size ingested by this pass (billing policy - BOS
     * discounts, cache restores - is the caller's, via {@link #withUsage}). {@code finishReason}:
     * "stop" for a stop token, "length" for the budget or deadline, "abort" when the sink ended the
     * pass.
     */
    public record GenerationResult(
            IntSequence tokens,
            int stopToken,
            int promptTokens,
            int completionTokens,
            int cachedTokens,
            String finishReason,
            double promptMillis,
            double predictedMillis) {

        /**
         * The same result with restamped billing counts (the prompt-cache path bills the whole
         * prompt, of which {@code cachedTokens} were restored instead of prefilled).
         */
        public GenerationResult withUsage(int promptTokens, int cachedTokens) {
            return new GenerationResult(
                    tokens,
                    stopToken,
                    promptTokens,
                    completionTokens,
                    cachedTokens,
                    finishReason,
                    promptMillis,
                    predictedMillis);
        }
    }

    /**
     * Sampler for a generation request: temperature / top-p (greedy at temp 0). Structure-aware
     * wrappers (think-marker bans, think budgets, grammars) are the caller's to layer on top.
     */
    public static Sampler configuredSampler(
            LanguageModel<?, ?, ?> model, float temperature, float topp, long seed) {
        require(
                Float.isFinite(temperature) && 0 <= temperature,
                "Invalid argument: temperature must be a finite non-negative number");
        require(
                Float.isFinite(topp) && 0 <= topp && topp <= 1,
                "Invalid argument: top_p must be within [0, 1]");
        return Sampler.select(model.config().vocabularySize(), temperature, topp, seed);
    }

    /**
     * One generation pass: ingest {@code prompt} at the state's cursor, then decode until a stop
     * token, an aborting sink, the wall-clock deadline, or the completion budget. The state is
     * caller-owned; a fresh state generates from the prompt, a resumed state continues from its
     * position (an empty prompt samples directly from the retained logits). Generations on a shared
     * model are serialized.
     */
    private static <S extends RuntimeState> GenerationResult generateBatches(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Params params,
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
                params.maxTokens() < 0
                        ? contextLength - promptPositions
                        : Math.min(params.maxTokens(), contextLength - promptPositions);

        boolean hasDeadline = params.timeoutNanos() != 0;
        long deadlineNanos =
                hasDeadline ? System.nanoTime() + params.timeoutNanos() : Long.MAX_VALUE;

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
                            params.stopTokens(),
                            actualMaxTokens,
                            params.sampler(),
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
        double promptMillis = (boundary - startNanos) / 1e6;
        double predictedMillis = (endNanos - boundary) / 1e6;

        int stopToken = -1;
        if (!responseTokens.isEmpty() && params.stopTokens().contains(responseTokens.getLast())) {
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
                promptCount,
                responseTokens.length(),
                0,
                finishReason,
                promptMillis,
                predictedMillis);
    }

    /** As {@link #generateBatches} for a plain token prompt. */
    public static <S extends RuntimeState> GenerationResult generate(
            LanguageModel<?, ?, S> model,
            S state,
            IntSequence promptTokens,
            Params params,
            TokenSink sink) {
        List<Batch> prompt =
                promptTokens.isEmpty() ? List.of() : List.of(Batch.prefill(promptTokens.toArray()));
        return generateBatches(model, state, prompt, params, sink);
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

    /** Prompt size as billed to the client: a leading BOS is template overhead, not user input. */
    public static int consumedPromptTokens(GgufTokenizer tokenizer, IntSequence promptTokens) {
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        int bos =
                specialTokens.getOrDefault(
                        "<bos>", specialTokens.getOrDefault("<|startoftext|>", 1));
        if (!promptTokens.isEmpty() && promptTokens.getFirst() == bos) {
            return promptTokens.length() - 1;
        }
        return promptTokens.length();
    }

    static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }
}
