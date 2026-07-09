// New-API generation driver: prefill via Batch.prefill (the port chunks internally), decode via
// logits(state) + Sampler + Batch.step. Owns the whole generation contract -
// Params/Listener/StopSpec/
// GenerationResult plus the tokenizer-based stop/stream/reasoning machinery (formerly shared with
// the
// now-removed legacy Engine). Token/text stops, streaming demux with think-span routing, wall-clock
// deadline, timing. NOTE: prompt caching / prefix resume is not (yet) supported on this path.
package com.qxotic.jinfer;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

public final class Generator {

    private Generator() {}

    /**
     * Sampling and limit parameters for one generation pass. The sampler is supplied by the caller
     * (see {@link #configuredSampler}) so interactive frontends can keep one RNG across turns.
     * {@code maxTokens} is the completion budget in generated tokens; negative means "as much as
     * the context allows". {@code timeoutNanos} is a wall-clock decode deadline as a DURATION
     * (converted to an absolute deadline at {@link #generate} entry); 0 means no deadline. {@code
     * inlineThink} keeps think spans inline in the text instead of routing them to the reasoning
     * channel.
     */
    record Params(
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            StopSpec stops,
            boolean inlineThink) {}

    /**
     * Token stops end generation (the stop token is reported but excluded from the result tokens);
     * text stops truncate the produced text and, when streaming, are held back so a configured stop
     * string is never emitted downstream.
     */
    record StopSpec(Set<Integer> tokenStops, List<String> textStops) {}

    /**
     * Streaming callbacks; fields are nullable. A non-null {@code onContent} switches to
     * incremental text decoding: deltas are UTF-8 safe and never contain a configured text stop.
     * {@code onReasoning} (only honored with {@code onContent}) receives think-span text. {@code
     * onToken} sees every sampled token - specials and the stop token included - before any text
     * for it is emitted.
     */
    record Listener(
            IntConsumer onToken,
            Consumer<String> onContent,
            Consumer<String> onReasoning,
            Consumer<String> onToolCall) {
        static final Listener NONE = new Listener(null, null, null, null);
    }

    /**
     * The outcome of one generation pass. {@code tokens} are the generated tokens excluding the
     * trailing stop token, reported in {@code stopToken} (-1 when not ended by a stop token).
     * {@code text} has think spans routed out (unless inlineThink) and text stops applied; {@code
     * reasoning} carries the think-span text for non-streaming passes. {@code toolCalls} is
     * attached by the chat layer after parsing - never by the generator.
     */
    record GenerationResult(
            List<Integer> tokens,
            int stopToken,
            String text,
            String reasoning,
            List<Map<String, Object>> toolCalls,
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
                    text,
                    reasoning,
                    toolCalls,
                    promptTokens,
                    completionTokens,
                    cachedTokens,
                    finishReason,
                    promptMillis,
                    predictedMillis);
        }

        /**
         * The chat-layer rewrite of a reply that parsed as tool calls; {@code content} is any text
         * the model produced before the first call marker.
         */
        GenerationResult asToolCalls(List<Map<String, Object>> calls, String content) {
            return new GenerationResult(
                    tokens,
                    stopToken,
                    content,
                    null,
                    calls,
                    promptTokens,
                    completionTokens,
                    cachedTokens,
                    "tool_calls",
                    promptMillis,
                    predictedMillis);
        }
    }

    /**
     * Sampler for a generation request: temperature / top-p (greedy at temp 0), optionally banning
     * the {@code <think>}/{@code </think>} markers when reasoning is disabled.
     */
    public static Sampler configuredSampler(
            LanguageModel<?, ?, ?> model, boolean think, float temperature, float topp, long seed) {
        require(
                Float.isFinite(temperature) && 0 <= temperature,
                "Invalid argument: temperature must be a finite non-negative number");
        require(
                Float.isFinite(topp) && 0 <= topp && topp <= 1,
                "Invalid argument: top_p must be within [0, 1]");
        Sampler sampler = Sampler.select(model.config().vocabularySize(), temperature, topp, seed);
        if (!think) {
            GgufTokenizer tokenizer = model.tokenizer();
            Integer thinkStart = tokenizer.getSpecialTokens().get("<think>");
            Integer thinkEnd = tokenizer.getSpecialTokens().get("</think>");
            Set<Integer> banned = new HashSet<>();
            if (thinkStart != null) banned.add(thinkStart);
            if (thinkEnd != null) banned.add(thinkEnd);
            sampler = Sampler.banning(sampler, banned);
        }
        return sampler;
    }

    /**
     * One generation pass on the new API: ingest {@code promptTokens} at the state's cursor, then
     * decode until a stop token, a text stop, the wall-clock deadline, or the completion budget.
     * The state is caller-owned; a fresh state generates from the prompt, a resumed state continues
     * from its position (an empty prompt samples directly from the retained logits). Generations on
     * a shared model are serialized.
     */
    public static <S extends RuntimeState> GenerationResult generate(
            LanguageModel<?, ?, S> model,
            S state,
            List<Integer> promptTokens,
            Params params,
            Listener listener) {
        GgufTokenizer tokenizer = model.tokenizer();
        int contextLength = model.config().contextLength();
        int consumedPromptTokens = consumedPromptTokens(tokenizer, promptTokens);
        int promptPositions =
                state.position() + promptTokens.size(); // new API ingests the prompt verbatim
        require(
                promptPositions <= contextLength,
                "Prompt exceeds context length (%d tokens used, %d available)",
                promptPositions,
                contextLength);
        int actualMaxTokens =
                params.maxTokens() < 0
                        ? contextLength - promptPositions
                        : Math.min(params.maxTokens(), contextLength - promptPositions);

        StringBuilder streamed = new StringBuilder();
        StopAwareTextConsumer stopAware = null;
        IntConsumer demux = null;
        if (listener.onContent() != null) {
            Consumer<String> onContent = listener.onContent();
            stopAware =
                    new StopAwareTextConsumer(
                            params.stops().textStops(),
                            text -> {
                                streamed.append(text);
                                onContent.accept(text);
                            });
            demux =
                    streamingDemux(
                            tokenizer,
                            stopAware,
                            listener.onReasoning(),
                            listener.onToolCall(),
                            params.inlineThink());
        } else if (!params.stops().textStops().isEmpty()) {
            stopAware = new StopAwareTextConsumer(params.stops().textStops(), text -> {});
            demux = streamingDemux(tokenizer, stopAware, null, null, params.inlineThink());
        }
        IntConsumer onToken = listener.onToken();
        IntConsumer textSink = demux;
        StopAwareTextConsumer stopTracker = stopAware;
        boolean hasDeadline = params.timeoutNanos() != 0;
        long deadlineNanos =
                hasDeadline ? System.nanoTime() + params.timeoutNanos() : Long.MAX_VALUE;
        boolean[] deadlineHit = {false};
        IntPredicate sink =
                onToken == null && textSink == null && !hasDeadline
                        ? null
                        : token -> {
                            if (onToken != null) onToken.accept(token);
                            if (textSink != null) textSink.accept(token);
                            if (System.nanoTime() >= deadlineNanos) {
                                deadlineHit[0] = true;
                                return false;
                            }
                            return stopTracker == null || !stopTracker.stopped();
                        };

        long startNanos = System.nanoTime();
        long[] prefillDoneNanos = {0};
        List<Integer> responseTokens;
        synchronized (model) { // generations on a shared model are strictly serialized
            responseTokens =
                    decodeLoop(
                            model,
                            state,
                            promptTokens,
                            params.stops().tokenStops(),
                            actualMaxTokens,
                            params.sampler(),
                            sink,
                            prefillDoneNanos);
        }
        long endNanos = System.nanoTime();
        long boundary = prefillDoneNanos[0] != 0 ? prefillDoneNanos[0] : endNanos;
        double promptMillis = (boundary - startNanos) / 1e6;
        double predictedMillis = (endNanos - boundary) / 1e6;

        if (stopAware != null) stopAware.flush();
        int stopToken = -1;
        if (!responseTokens.isEmpty()
                && params.stops().tokenStops().contains(responseTokens.getLast())) {
            stopToken = responseTokens.removeLast();
        }
        String text =
                listener.onContent() != null
                        ? streamed.toString()
                        : tokenizer.decode(
                                visibleTokens(tokenizer, responseTokens, params.inlineThink()));
        StopResult stopResult = applyTextStops(text, params.stops().textStops());
        boolean textStopped = stopResult.stopped() || (stopAware != null && stopAware.stopped());
        String finishReason =
                stopToken >= 0 || textStopped
                        ? "stop"
                        : (deadlineHit[0] || responseTokens.size() >= actualMaxTokens
                                ? "length"
                                : "stop");
        String reasoning =
                listener.onContent() == null && !params.inlineThink()
                        ? reasoningText(tokenizer, responseTokens)
                        : null;
        return new GenerationResult(
                responseTokens,
                stopToken,
                stopResult.text(),
                reasoning,
                List.of(),
                consumedPromptTokens,
                responseTokens.size(),
                0,
                finishReason,
                promptMillis,
                predictedMillis);
    }

    /**
     * Prefill the prompt (one Batch.prefill; the port chunks it internally), then decode one token
     * at a time via logits(state) + sampler + Batch.step until a stop token, an aborting sink, or
     * the budget. {@code onTokenGenerated} returning false aborts (the aborting token is recorded
     * but not ingested).
     */
    private static <S extends RuntimeState> List<Integer> decodeLoop(
            LanguageModel<?, ?, S> model,
            S state,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxNewTokens,
            Sampler sampler,
            IntPredicate onTokenGenerated,
            long[] prefillDoneNanos) {
        int vocab = model.config().vocabularySize();
        int contextLength = model.config().contextLength();
        if (!promptTokens.isEmpty()) {
            int[] ids = promptTokens.stream().mapToInt(Integer::intValue).toArray();
            model.ingest(
                    state,
                    Batch.prefill(ids)); // the port chunks by batchCapacity + runs the decode pool
        }
        List<Integer> generated = new ArrayList<>();
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
            boolean keepGoing = onTokenGenerated == null || onTokenGenerated.test(nextToken);
            if (stopTokens.contains(nextToken) || !keepGoing) break;
            if (generated.size() >= maxNewTokens || state.position() >= contextLength) break;
            model.ingest(state, Batch.step(nextToken));
        }
        return generated;
    }

    // ---- tokenizer-based stop / stream / reasoning machinery (model-agnostic) ----

    /** Prompt size as billed to the client: a leading BOS is template overhead, not user input. */
    static int consumedPromptTokens(GgufTokenizer tokenizer, List<Integer> promptTokens) {
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        int bos =
                specialTokens.getOrDefault(
                        "<bos>", specialTokens.getOrDefault("<|startoftext|>", 1));
        if (!promptTokens.isEmpty() && promptTokens.getFirst() == bos) {
            return promptTokens.size() - 1;
        }
        return promptTokens.size();
    }

    /**
     * Caps the think span: once {@code budget} tokens have been sampled inside {@code <think>}, the
     * close marker is forced so the remaining completion budget always goes to content (thinking
     * models otherwise starve the answer under tight max_tokens). Cumulative across spans; the
     * forced token consumes no RNG draw. Negative = uncapped.
     */
    static Sampler withThinkBudget(Sampler inner, GgufTokenizer tokenizer, int budget) {
        Integer open = tokenizer.getSpecialTokens().get("<think>");
        Integer close = tokenizer.getSpecialTokens().get("</think>");
        if (budget < 0 || open == null || close == null) {
            return inner;
        }
        int openToken = open, closeToken = close;
        return new Sampler() {
            boolean inThink;
            int thought;

            @Override
            public int sampleToken(FloatTensor logits) {
                if (inThink && thought >= budget) {
                    inThink = false;
                    return closeToken;
                }
                int token = inner.sampleToken(logits);
                if (token == openToken) inThink = true;
                else if (token == closeToken) inThink = false;
                else if (inThink) thought++;
                return token;
            }
        };
    }

    /**
     * Routes each generated token to the content, reasoning, or tool-call channel. Think markers
     * flip the think flag (inline mode emits them literally). Tool-call spans are buffered
     * separately so an in-progress call never leaks into the text stream. Other specials are
     * dropped. Everything else is UTF-8 decoded incrementally.
     */
    static IntConsumer streamingDemux(
            GgufTokenizer tokenizer,
            Consumer<String> onText,
            Consumer<String> onReasoning,
            Consumer<String> onToolCall,
            boolean inlineThink) {
        Integer thinkOpen = tokenizer.getSpecialTokens().get("<think>");
        Integer thinkClose = tokenizer.getSpecialTokens().get("</think>");
        Integer tcOpen = tokenizer.getSpecialTokens().get("<|tool_call_start|>");
        Integer tcClose = tokenizer.getSpecialTokens().get("<|tool_call_end|>");
        boolean[] inThink = {false};
        boolean[] inToolCall = {false};
        Utf8TokenDecoder textDecoder = new Utf8TokenDecoder(onText);
        Utf8TokenDecoder reasoningDecoder =
                onReasoning != null && !inlineThink ? new Utf8TokenDecoder(onReasoning) : null;
        Utf8TokenDecoder toolCallDecoder =
                onToolCall != null && tcOpen != null && tcClose != null
                        ? new Utf8TokenDecoder(onToolCall)
                        : null;
        return token -> {
            if (tokenizer.isSpecialToken(token)) {
                if (thinkOpen != null && token == thinkOpen) {
                    inThink[0] = true;
                    if (inlineThink) textDecoder.accept(tokenizer.decodeTokenBytes(token));
                }
                if (thinkClose != null && token == thinkClose) {
                    inThink[0] = false;
                    if (inlineThink) textDecoder.accept(tokenizer.decodeTokenBytes(token));
                    textDecoder.flushPending();
                    if (reasoningDecoder != null) reasoningDecoder.flushPending();
                }
                if (tcOpen != null && token == tcOpen) {
                    inToolCall[0] = true;
                    textDecoder.flushPending();
                }
                if (tcClose != null && token == tcClose) {
                    inToolCall[0] = false;
                    if (toolCallDecoder != null) toolCallDecoder.flushPending();
                }
                return;
            }
            if (inToolCall[0] && toolCallDecoder != null) {
                toolCallDecoder.accept(tokenizer.decodeTokenBytes(token));
                return;
            }
            if (inThink[0] && !inlineThink) {
                if (reasoningDecoder != null) {
                    reasoningDecoder.accept(tokenizer.decodeTokenBytes(token));
                }
                return;
            }
            textDecoder.accept(tokenizer.decodeTokenBytes(token));
        };
    }

    static List<Integer> visibleTokens(
            GgufTokenizer tokenizer, List<Integer> tokens, boolean think) {
        if (think) {
            return tokens;
        }
        return stripThoughtTokens(tokenizer, tokens);
    }

    /** The think-span text of a response (between <think> markers, terminated or not), or null. */
    static String reasoningText(GgufTokenizer tokenizer, List<Integer> tokens) {
        Integer thinkOpen = tokenizer.getSpecialTokens().get("<think>");
        Integer thinkClose = tokenizer.getSpecialTokens().get("</think>");
        if (thinkOpen == null || thinkClose == null) return null;
        List<Integer> thinking = new ArrayList<>();
        boolean inThink = false;
        for (int token : tokens) {
            if (token == thinkOpen) inThink = true;
            else if (token == thinkClose) inThink = false;
            else if (inThink && !tokenizer.isSpecialToken(token)) thinking.add(token);
        }
        return thinking.isEmpty() ? null : tokenizer.decode(thinking);
    }

    private static List<Integer> stripThoughtTokens(GgufTokenizer tokenizer, List<Integer> tokens) {
        Integer thinkOpen = tokenizer.getSpecialTokens().get("<think>");
        Integer thinkClose = tokenizer.getSpecialTokens().get("</think>");
        if (thinkOpen == null || thinkClose == null) {
            return tokens;
        }
        List<Integer> result = new ArrayList<>();
        boolean inThink = false;
        for (int token : tokens) {
            if (token == thinkOpen) {
                inThink = true;
                continue;
            }
            if (token == thinkClose) {
                inThink = false;
                continue;
            }
            if (!inThink) {
                result.add(token);
            }
        }
        // an unterminated <think> (max_tokens cut generation mid-think) is still thinking: hide it
        // -
        // API clients must never see raw think markup in content
        return result;
    }

    record StopResult(String text, boolean stopped) {}

    static StopResult applyTextStops(String text, List<String> stops) {
        int cut = -1;
        for (String stop : stops) {
            int index = text.indexOf(stop);
            if (index >= 0 && (cut < 0 || index < cut)) cut = index;
        }
        return cut >= 0
                ? new StopResult(text.substring(0, cut), true)
                : new StopResult(text, false);
    }

    /**
     * Forwards text downstream while holding back any suffix that could grow into a configured stop
     * string; once a stop string appears the text before it is emitted and the consumer goes
     * silent.
     */
    static final class StopAwareTextConsumer implements Consumer<String> {
        private final List<String> stops;
        private final Consumer<String> downstream;
        private final StringBuilder pending = new StringBuilder();
        private boolean stopped;

        StopAwareTextConsumer(List<String> stops, Consumer<String> downstream) {
            this.stops = stops;
            this.downstream = downstream;
        }

        @Override
        public void accept(String text) {
            if (stopped || text.isEmpty()) return;
            pending.append(text);
            StopResult stopResult = applyTextStops(pending.toString(), stops);
            if (stopResult.stopped()) {
                emit(stopResult.text());
                pending.setLength(0);
                stopped = true;
                return;
            }
            int keep = longestStopPrefixSuffix(pending, stops);
            int emitLength = pending.length() - keep;
            if (emitLength > 0) {
                emit(pending.substring(0, emitLength));
                pending.delete(0, emitLength);
            }
        }

        private void emit(String text) {
            if (!text.isEmpty()) downstream.accept(text);
        }

        void flush() {
            if (!stopped && !pending.isEmpty()) {
                emit(pending.toString());
                pending.setLength(0);
            }
        }

        boolean stopped() {
            return stopped;
        }

        private static int longestStopPrefixSuffix(StringBuilder text, List<String> stops) {
            int keep = 0;
            String current = text.toString();
            for (String stop : stops) {
                int max = Math.min(stop.length() - 1, current.length());
                for (int len = max; len > keep; len--) {
                    if (current.endsWith(stop.substring(0, len))) {
                        keep = len;
                        break;
                    }
                }
            }
            return keep;
        }
    }

    /**
     * Buffers token bytes until they form valid UTF-8, so multi-byte sequences split across tokens
     * never reach the consumer as replacement characters.
     */
    static final class Utf8TokenDecoder {
        private final Consumer<String> downstream;
        private final ByteArrayOutputStream pending = new ByteArrayOutputStream();
        private final CharsetDecoder decoder =
                StandardCharsets.UTF_8
                        .newDecoder()
                        .onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT);

        Utf8TokenDecoder(Consumer<String> downstream) {
            this.downstream = downstream;
        }

        void accept(byte[] bytes) {
            pending.writeBytes(bytes);
            decodePending();
        }

        /**
         * Emits any accumulated incomplete bytes as-is, resetting the pending buffer. Called at
         * channel transitions (e.g. think->content) to prevent stale bytes from polluting the next
         * channel.
         */
        void flushPending() {
            if (pending.size() > 0) {
                downstream.accept(pending.toString(StandardCharsets.UTF_8));
                pending.reset();
            }
        }

        private void decodePending() {
            try {
                decoder.reset();
                CharBuffer chars = decoder.decode(ByteBuffer.wrap(pending.toByteArray()));
                if (!chars.isEmpty()) downstream.accept(chars.toString());
                pending.reset();
            } catch (CharacterCodingException ignored) {
                // Wait for a later token to complete a split UTF-8 sequence.
            }
        }
    }

    static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }
}
