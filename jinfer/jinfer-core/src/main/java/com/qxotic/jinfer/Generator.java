// New-API generation driver: the LanguageModel/Batch equivalent of Engine.generate (which drives the
// legacy ModelLegacy engine). Prefill via Batch.prefill (the port chunks internally), decode via
// logits(state) + Sampler + Batch.step. Token/text stops, streaming demux, wall-clock deadline, timing
// and GenerationResult all match Engine's semantics - the tokenizer-based helpers are reused as-is.
// NOTE: prompt caching (Engine's GenerationHooks / prefix resume) is not yet ported to this path.
package com.qxotic.jinfer;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

public final class Generator {

    private Generator() {
    }

    /** Sampler for a generation request: temperature / top-p (greedy at temp 0), optionally banning the
     *  {@code <think>}/{@code </think>} markers when reasoning is disabled. Mirrors Engine.configuredSampler
     *  but reads vocabulary + special tokens off the new-API model. */
    public static Sampler configuredSampler(LanguageModel<?, ?, ?> model, boolean think, float temperature, float topp, long seed) {
        Engine.require(Float.isFinite(temperature) && 0 <= temperature, "Invalid argument: temperature must be a finite non-negative number");
        Engine.require(Float.isFinite(topp) && 0 <= topp && topp <= 1, "Invalid argument: top_p must be within [0, 1]");
        Sampler sampler = Sampler.select(model.config().vocabularySize(), temperature, topp, seed);
        if (!think) {
            LFMTokenizer tokenizer = model.tokenizer();
            Integer thinkStart = tokenizer.getSpecialTokens().get("<think>");
            Integer thinkEnd = tokenizer.getSpecialTokens().get("</think>");
            java.util.Set<Integer> banned = new java.util.HashSet<>();
            if (thinkStart != null) banned.add(thinkStart);
            if (thinkEnd != null) banned.add(thinkEnd);
            sampler = Sampler.banning(sampler, banned);
        }
        return sampler;
    }

    /**
     * One generation pass on the new API: ingest {@code promptTokens} at the state's cursor, then decode
     * until a stop token, a text stop, the wall-clock deadline, or the completion budget. The state is
     * caller-owned; a fresh state generates from the prompt, a resumed state continues from its position
     * (an empty prompt samples directly from the retained logits). Generations on a shared model are
     * serialized. Mirrors {@link Engine#generate} and returns the same {@link Engine.GenerationResult}.
     */
    public static <S extends RuntimeState> Engine.GenerationResult generate(
            LanguageModel<?, ?, S> model, S state, List<Integer> promptTokens,
            Engine.Params params, Engine.Listener listener) {
        LFMTokenizer tokenizer = model.tokenizer();
        int contextLength = model.config().contextLength();
        int consumedPromptTokens = Engine.consumedPromptTokens(tokenizer, promptTokens);
        int promptPositions = state.position() + promptTokens.size();   // new API ingests the prompt verbatim
        Engine.require(promptPositions <= contextLength,
                "Prompt exceeds context length (%d tokens used, %d available)", promptPositions, contextLength);
        int actualMaxTokens = params.maxTokens() < 0 ? contextLength - promptPositions
                : Math.min(params.maxTokens(), contextLength - promptPositions);

        // Streaming / stop-aware text sink (identical to Engine.generate).
        StringBuilder streamed = new StringBuilder();
        Engine.StopAwareTextConsumer stopAware = null;
        IntConsumer demux = null;
        if (listener.onContent() != null) {
            Consumer<String> onContent = listener.onContent();
            stopAware = new Engine.StopAwareTextConsumer(params.stops().textStops(), text -> {
                streamed.append(text);
                onContent.accept(text);
            });
            demux = Engine.streamingDemux(tokenizer, stopAware, listener.onReasoning(), listener.onToolCall(), params.inlineThink());
        } else if (!params.stops().textStops().isEmpty()) {
            stopAware = new Engine.StopAwareTextConsumer(params.stops().textStops(), text -> {});
            demux = Engine.streamingDemux(tokenizer, stopAware, null, null, params.inlineThink());
        }
        IntConsumer onToken = listener.onToken();
        IntConsumer textSink = demux;
        Engine.StopAwareTextConsumer stopTracker = stopAware;
        boolean hasDeadline = params.timeoutNanos() != 0;
        long deadlineNanos = hasDeadline ? System.nanoTime() + params.timeoutNanos() : Long.MAX_VALUE;
        boolean[] deadlineHit = {false};
        IntPredicate sink = onToken == null && textSink == null && !hasDeadline ? null : token -> {
            if (onToken != null) onToken.accept(token);
            if (textSink != null) textSink.accept(token);
            if (System.nanoTime() >= deadlineNanos) { deadlineHit[0] = true; return false; }
            return stopTracker == null || !stopTracker.stopped();
        };

        long startNanos = System.nanoTime();
        long[] prefillDoneNanos = {0};
        List<Integer> responseTokens;
        synchronized (model) { // generations on a shared model are strictly serialized
            responseTokens = decodeLoop(model, state, promptTokens, params.stops().tokenStops(),
                    actualMaxTokens, params.sampler(), sink, prefillDoneNanos);
        }
        long endNanos = System.nanoTime();
        long boundary = prefillDoneNanos[0] != 0 ? prefillDoneNanos[0] : endNanos;
        double promptMillis = (boundary - startNanos) / 1e6;
        double predictedMillis = (endNanos - boundary) / 1e6;

        if (stopAware != null) stopAware.flush();
        int stopToken = -1;
        if (!responseTokens.isEmpty() && params.stops().tokenStops().contains(responseTokens.getLast())) {
            stopToken = responseTokens.removeLast();
        }
        String text = listener.onContent() != null ? streamed.toString()
                : tokenizer.decode(Engine.visibleTokens(tokenizer, responseTokens, params.inlineThink()));
        Engine.StopResult stopResult = Engine.applyTextStops(text, params.stops().textStops());
        boolean textStopped = stopResult.stopped() || (stopAware != null && stopAware.stopped());
        String finishReason = stopToken >= 0 || textStopped ? "stop"
                : (deadlineHit[0] || responseTokens.size() >= actualMaxTokens ? "length" : "stop");
        String reasoning = listener.onContent() == null && !params.inlineThink()
                ? Engine.reasoningText(tokenizer, responseTokens) : null;
        return new Engine.GenerationResult(responseTokens, stopToken, stopResult.text(), reasoning, List.of(),
                consumedPromptTokens, responseTokens.size(), 0, finishReason, promptMillis, predictedMillis);
    }

    /** Prefill the prompt (one Batch.prefill; the port chunks it internally), then decode one token at a
     *  time via logits(state) + sampler + Batch.step until a stop token, an aborting sink, or the budget.
     *  {@code onTokenGenerated} returning false aborts (the aborting token is recorded but not ingested). */
    private static <S extends RuntimeState> List<Integer> decodeLoop(
            LanguageModel<?, ?, S> model, S state, List<Integer> promptTokens, Set<Integer> stopTokens,
            int maxNewTokens, Sampler sampler, IntPredicate onTokenGenerated, long[] prefillDoneNanos) {
        int vocab = model.config().vocabularySize();
        int contextLength = model.config().contextLength();
        if (!promptTokens.isEmpty()) {
            int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
            model.ingest(state, Batch.prefill(ids));   // the port chunks by batchCapacity + runs the decode pool
        }
        List<Integer> generated = new ArrayList<>();
        while (generated.size() < maxNewTokens) {
            FloatTensor logits = model.logits(state);   // last retained row; ports run this on the decode pool
            if (prefillDoneNanos[0] == 0) prefillDoneNanos[0] = System.nanoTime(); // time-to-first-token boundary
            int nextToken = sampler.sampleToken(logits);
            if (nextToken < 0 || nextToken >= vocab) {
                throw new IllegalArgumentException(
                        "sampler returned token id " + nextToken + " out of range [0, " + vocab + ")");
            }
            generated.add(nextToken);
            boolean keepGoing = onTokenGenerated == null || onTokenGenerated.test(nextToken);
            if (stopTokens.contains(nextToken) || !keepGoing) break;
            if (generated.size() >= maxNewTokens || state.position() >= contextLength) break;
            model.ingest(state, Batch.step(nextToken));
        }
        return generated;
    }
}
