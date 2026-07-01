package com.qxotic.jinfer;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/**
 * The inference-side seam between the generation loop ({@link #decodeLoop}) and any frontend
 * (CLI, HTTP server, future embedders). One entry point — {@link #generate} — runs a complete
 * generation pass: sampler-driven decode with stop handling (token stops, and text stops with
 * emission holdback), think-span routing to a separate reasoning channel, UTF-8-safe incremental
 * text decoding for streaming consumers, prefill/decode timing, and cached-prefix accounting.
 * Transports own request parsing, response encoding and scheduling; the engine owns everything
 * about how tokens become a result.
 */
final class Engine {

    private Engine() {
    }

    /** Sampling and limit parameters for one generation pass. The sampler is supplied by the
     *  caller (see {@link #configuredSampler}) so interactive frontends can keep one RNG across
     *  turns. {@code maxTokens} is the completion budget in generated tokens; negative means
     *  "as much as the context allows". {@code timeoutNanos} is a wall-clock decode deadline as a
     *  DURATION (converted to an absolute deadline at {@link #generate} entry, so it measures
     *  execution rather than queue-wait); 0 means no deadline. {@code inlineThink} keeps think
     *  spans inline in the text instead of routing them to the reasoning channel. */
    record Params(Sampler sampler, int maxTokens, long timeoutNanos, StopSpec stops, boolean inlineThink) {}

    /** Token stops end generation (the stop token is reported but excluded from the result
     *  tokens); text stops truncate the produced text and, when streaming, are held back so a
     *  configured stop string is never emitted downstream. */
    record StopSpec(Set<Integer> tokenStops, List<String> textStops) {}

    /** Streaming callbacks; fields are nullable. A non-null {@code onContent} switches the
     *  engine to incremental text decoding: deltas are UTF-8 safe and never contain a configured
     *  text stop. {@code onReasoning} (only honored together with {@code onContent}) receives
     *  think-span text. {@code onToken} sees every sampled token — specials and the stop token
     *  included — before any text for it is emitted. */
    record Listener(IntConsumer onToken, Consumer<String> onContent, Consumer<String> onReasoning,
                    Consumer<String> onToolCall) {
        static final Listener NONE = new Listener(null, null, null, null);
    }

    /**
     * The outcome of one generation pass. {@code tokens} are the generated tokens excluding the
     * trailing stop token, which is reported in {@code stopToken} (-1 when generation was not
     * ended by a stop token). {@code text} has think spans routed out (unless inlineThink) and
     * text stops applied; {@code reasoning} carries the think-span text for non-streaming passes.
     * {@code toolCalls} is attached by the chat layer after parsing — never by the engine.
     */
    record GenerationResult(List<Integer> tokens, int stopToken, String text, String reasoning,
                            List<Map<String, Object>> toolCalls,
                            int promptTokens, int completionTokens, int cachedTokens, String finishReason,
                            double promptMillis, double predictedMillis) {

        /** The chat-layer rewrite of a reply that parsed as tool calls; {@code content} is any
         *  text the model produced before the first call marker. */
        GenerationResult asToolCalls(List<Map<String, Object>> calls, String content) {
            return new GenerationResult(tokens, stopToken, content, null, calls, promptTokens, completionTokens,
                    cachedTokens, "tool_calls", promptMillis, predictedMillis);
        }
    }

    /**
     * One full generation pass over {@link #decodeLoop}: ingest the prompt (resuming any
     * cached prefix via {@code hooks}), then decode until a stop token, a text stop, or the
     * completion budget. Timing uses the prefill/decode boundary reported by the loop
     * ({@code afterPrefill}); cached-prefix counts are captured from the hooks' resume position.
     */
    static GenerationResult generate(ModelLegacy model, InferenceState state, int startPosition, List<Integer> promptTokens,
                                     Params params, Listener listener, GenerationHooks hooks) {
        LFMTokenizer tokenizer = model.tokenizer();
        int contextLength = model.contextLength();
        int consumedPromptTokens = consumedPromptTokens(tokenizer, promptTokens); // client-facing usage counts
        // generation limits use REAL stream positions: a resumed session sits deeper in the
        // context than the re-encoded prompt suggests (thinking/surrogate tokens in the stream)
        int promptPositions = prefillPositions(state, startPosition, promptTokens);
        require(promptPositions <= contextLength, "Prompt exceeds context length (%d tokens used, %d available)", promptPositions, contextLength);
        int actualMaxTokens = params.maxTokens() < 0 ? contextLength - promptPositions
                : Math.min(params.maxTokens(), contextLength - promptPositions);
        int totalTokenLimit = promptPositions + actualMaxTokens;

        StringBuilder streamed = new StringBuilder();
        StopAwareTextConsumer stopAware = null;
        IntConsumer demux = null;
        if (listener.onContent() != null) {
            Consumer<String> onContent = listener.onContent();
            stopAware = new StopAwareTextConsumer(params.stops().textStops(), text -> {
                streamed.append(text);
                onContent.accept(text);
            });
            demux = streamingDemux(tokenizer, stopAware, listener.onReasoning(), listener.onToolCall(), params.inlineThink());
        } else if (!params.stops().textStops().isEmpty()) {
            // non-streaming: track text stops on a silent content-only decode so generation
            // aborts when one matches instead of burning the rest of the completion budget
            stopAware = new StopAwareTextConsumer(params.stops().textStops(), text -> {});
            demux = streamingDemux(tokenizer, stopAware, null, null, params.inlineThink());
        }
        IntConsumer onToken = listener.onToken();
        IntConsumer textSink = demux;
        StopAwareTextConsumer stopTracker = stopAware;
        // wall-clock decode deadline: duration -> absolute now (we already run on the worker)
        boolean hasDeadline = params.timeoutNanos() != 0;
        long deadlineNanos = hasDeadline ? System.nanoTime() + params.timeoutNanos() : Long.MAX_VALUE;
        boolean[] deadlineHit = {false};
        IntPredicate sink = onToken == null && textSink == null && !hasDeadline ? null : token -> {
            if (onToken != null) onToken.accept(token);
            if (textSink != null) textSink.accept(token);
            if (System.nanoTime() >= deadlineNanos) { deadlineHit[0] = true; return false; }
            return stopTracker == null || !stopTracker.stopped();
        };

        long[] prefillDoneNanos = {0};
        int[] resumed = {0};
        GenerationHooks wrapped = new GenerationHooks() {
            @Override
            public int resumePosition(int[] stream, int prefillLength) {
                int cached = hooks.resumePosition(stream, prefillLength);
                resumed[0] = cached;
                return cached;
            }

            @Override
            public int clampChunk(int position, int chunkLength) {
                return hooks.clampChunk(position, chunkLength);
            }

            @Override
            public void afterIngest(int[] stream, int position) {
                hooks.afterIngest(stream, position);
            }

            @Override
            public void afterPrefill() {
                prefillDoneNanos[0] = System.nanoTime();
                hooks.afterPrefill();
            }
        };

        long startNanos = System.nanoTime();
        List<Integer> responseTokens;
        synchronized (model) { // generations on a shared model are strictly serialized
            responseTokens = decodeLoop(model, state, startPosition, promptTokens,
                    params.stops().tokenStops(), totalTokenLimit, params.sampler(), sink, wrapped);
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
                : tokenizer.decode(visibleTokens(tokenizer, responseTokens, params.inlineThink()));
        StopResult stopResult = applyTextStops(text, params.stops().textStops());
        boolean textStopped = stopResult.stopped() || (stopAware != null && stopAware.stopped());
        String finishReason = stopToken >= 0 || textStopped ? "stop"
                : (deadlineHit[0] || responseTokens.size() >= actualMaxTokens ? "length" : "stop"); // deadline abort = truncated
        String reasoning = listener.onContent() == null && !params.inlineThink() ? reasoningText(tokenizer, responseTokens) : null; // streaming already delivered reasoning deltas
        int cachedTokens = Math.min(startPosition > 0 ? startPosition : resumed[0], consumedPromptTokens);
        return new GenerationResult(responseTokens, stopToken, stopResult.text(), reasoning, List.of(),
                consumedPromptTokens, responseTokens.size(), cachedTokens, finishReason, promptMillis, predictedMillis);
    }

    /**
     * The effective token stream prefill ingests: the not-yet-ingested {@code latestToken}
     * (BOS for a fresh state) followed by the prompt, deduplicating a leading BOS. Position i of
     * the result is the token ingested at position {@code startPosition + i} — the canonical key
     * for prefix caching. A negative {@code latestToken} means "no prior token" (e.g. a fresh state
     * of a model with add_bos=false): the prompt is ingested verbatim with nothing prepended.
     */
    static int[] buildPrefillTokens(int latestToken, int startPosition, List<Integer> promptTokens) {
        if (latestToken < 0) {
            return promptTokens.stream().mapToInt(Integer::intValue).toArray();
        }
        int skip = startPosition == 0 && !promptTokens.isEmpty() && promptTokens.getFirst() == latestToken ? 1 : 0;
        int[] prefillTokens = new int[1 + promptTokens.size() - skip];
        prefillTokens[0] = latestToken;
        for (int i = 1; i < prefillTokens.length; i++) {
            prefillTokens[i] = promptTokens.get(skip + i - 1);
        }
        return prefillTokens;
    }

    /** Number of context positions occupied after prefill ingests the effective stream. */
    static int prefillPositions(InferenceState state, int startPosition, List<Integer> promptTokens) {
        if (promptTokens.isEmpty()) {
            return startPosition;
        }
        return startPosition + buildPrefillTokens(state.latestToken(), startPosition, promptTokens).length;
    }

    /**
     * The generation loop — prefill and decode are one operation: ingest the pending span of the
     * token stream. The prompt is pending up front (chunked by {@link ModelLegacy#batchCapacity()});
     * decode appends one sampled token at a time and ingests it through the identical path.
     * {@code maxTokens} is a total-position limit; stop tokens are recorded but never ingested; an
     * empty prompt samples directly from the current logits (multi-turn continuation).
     */
    static List<Integer> decodeLoop(ModelLegacy model, InferenceState state, int startPosition, List<Integer> promptTokens,
                                    Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                    IntPredicate onTokenGenerated, GenerationHooks hooks) {
        int contextLength = model.contextLength();
        int vocabularySize = model.vocabularySize();
        int capacity = model.batchCapacity();
        if (maxTokens < 0 || contextLength < maxTokens) {
            maxTokens = contextLength;
        }
        int[] prefill = promptTokens.isEmpty() ? new int[0] : buildPrefillTokens(state.latestToken(), startPosition, promptTokens);
        int[] stream = Arrays.copyOf(prefill, Math.max(Math.max(maxTokens - startPosition, prefill.length), 1));
        int length = prefill.length;
        int position = length > 0 ? hooks.resumePosition(stream, length) : 0;
        boolean prefilling = position < length;
        List<Integer> generatedTokens = new ArrayList<>();
        while (true) {
            if (position == length) {                        // nothing pending: extend the stream
                if (startPosition + position >= maxTokens) {
                    break;
                }
                FloatTensor logits = Parallel.onDecodePool(() -> model.computeLogits(state));
                if (prefilling) {
                    prefilling = false;
                    hooks.afterPrefill();
                }
                int nextToken = sampler.sampleToken(logits);
                if (nextToken < 0 || nextToken >= vocabularySize) {
                    throw new IllegalArgumentException(
                        "sampler returned token id " + nextToken + " out of range [0, " + vocabularySize + ")");
                }
                generatedTokens.add(nextToken);
                // a false return aborts generation (e.g. a text stop matched downstream); like
                // a stop token, the aborting token is recorded but never ingested
                boolean keepGoing = onTokenGenerated == null || onTokenGenerated.test(nextToken);
                state.latestToken(nextToken);
                if (stopTokens.contains(nextToken) || !keepGoing) {
                    break;
                }
                stream[length++] = nextToken;
            }
            int chunk = Math.min(length - position, capacity);
            // never ingest past the kv-cache capacity: cache writes are unchecked (UNSAFE) and a
            // context overflow segfaults instead of failing gracefully
            chunk = Math.min(chunk, contextLength - (startPosition + position));
            if (chunk <= 0) {
                break;
            }
            chunk = hooks.clampChunk(position, chunk);
            final int p = position, c = chunk;
            if (c == 1) {   // decode step: physical-core-width pool (bandwidth bound); prefill keeps the common pool
                Parallel.onDecodePool(() -> { model.ingest(state, stream, p, startPosition + p, c); return null; });
            } else {
                model.ingest(state, stream, position, startPosition + position, chunk);
            }
            position += chunk;
            hooks.afterIngest(stream, position);
        }
        return generatedTokens;
    }

    /** Prompt size as billed to the client: a leading BOS is template overhead, not user input. */
    static int consumedPromptTokens(LFMTokenizer tokenizer, List<Integer> promptTokens) {
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        int bos = specialTokens.getOrDefault("<bos>", specialTokens.getOrDefault("<|startoftext|>", 1));
        if (!promptTokens.isEmpty() && promptTokens.getFirst() == bos) {
            return promptTokens.size() - 1;
        }
        return promptTokens.size();
    }

    /** {@link Sampler#select} plus the think-token ban when thinking is disabled. */
    static Sampler configuredSampler(ModelLegacy model, boolean think, float temperature, float topp, long seed) {
        require(Float.isFinite(temperature) && 0 <= temperature, "Invalid argument: temperature must be a finite non-negative number");
        require(Float.isFinite(topp) && 0 <= topp && topp <= 1, "Invalid argument: top_p must be within [0, 1]");
        Sampler sampler = Sampler.select(model.vocabularySize(), temperature, topp, seed);
        if (!think) {
            Integer thinkStart = model.tokenizer().getSpecialTokens().get("<think>");
            Integer thinkEnd = model.tokenizer().getSpecialTokens().get("</think>");
            Set<Integer> banned = new HashSet<>();
            if (thinkStart != null) banned.add(thinkStart);
            if (thinkEnd != null) banned.add(thinkEnd);
            sampler = Sampler.banning(sampler, banned);
        }
        return sampler;
    }

    /**
     * Caps the think span: once {@code budget} tokens have been sampled inside {@code <think>},
     * the close marker is forced so the remaining completion budget always goes to content
     * (thinking models otherwise starve the answer under tight max_tokens). The budget is
     * cumulative across spans; the forced token consumes no RNG draw. Negative = uncapped.
     */
    static Sampler withThinkBudget(Sampler inner, LFMTokenizer tokenizer, int budget) {
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

    /** Routes each generated token to the content, reasoning, or tool-call channel.
     *  Think markers flip the think flag (inline mode emits them literally). Tool-call
     *  spans are buffered separately so an in-progress call never leaks into the text
     *  stream. Other specials are dropped. Everything else is UTF-8 decoded incrementally. */
    private static IntConsumer streamingDemux(LFMTokenizer tokenizer, Consumer<String> onText,
                                               Consumer<String> onReasoning, Consumer<String> onToolCall,
                                               boolean inlineThink) {
        Integer thinkOpen = tokenizer.getSpecialTokens().get("<think>");
        Integer thinkClose = tokenizer.getSpecialTokens().get("</think>");
        Integer tcOpen = tokenizer.getSpecialTokens().get("<|tool_call_start|>");
        Integer tcClose = tokenizer.getSpecialTokens().get("<|tool_call_end|>");
        boolean[] inThink = {false};
        boolean[] inToolCall = {false};
        Utf8TokenDecoder textDecoder = new Utf8TokenDecoder(onText);
        Utf8TokenDecoder reasoningDecoder = onReasoning != null && !inlineThink ? new Utf8TokenDecoder(onReasoning) : null;
        Utf8TokenDecoder toolCallDecoder = onToolCall != null && tcOpen != null && tcClose != null
                ? new Utf8TokenDecoder(onToolCall) : null;
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

    static List<Integer> visibleTokens(LFMTokenizer tokenizer, List<Integer> tokens, boolean think) {
        if (think) {
            return tokens;
        }
        return stripThoughtTokens(tokenizer, tokens);
    }

    /** The think-span text of a response (between <think> markers, terminated or not), or null. */
    static String reasoningText(LFMTokenizer tokenizer, List<Integer> tokens) {
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

    private static List<Integer> stripThoughtTokens(LFMTokenizer tokenizer, List<Integer> tokens) {
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
        // an unterminated <think> (max_tokens cut generation mid-think) is still thinking:
        // hide it — API clients must never see raw think markup in content
        return result;
    }

    record StopResult(String text, boolean stopped) {}

    static StopResult applyTextStops(String text, List<String> stops) {
        int cut = -1;
        for (String stop : stops) {
            int index = text.indexOf(stop);
            if (index >= 0 && (cut < 0 || index < cut)) cut = index;
        }
        return cut >= 0 ? new StopResult(text.substring(0, cut), true) : new StopResult(text, false);
    }

    /** Forwards text downstream while holding back any suffix that could grow into a configured
     *  stop string; once a stop string appears the text before it is emitted and the consumer
     *  goes silent. */
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

    /** Buffers token bytes until they form valid UTF-8, so multi-byte sequences split across
     *  tokens never reach the consumer as replacement characters. */
    static final class Utf8TokenDecoder {
        private final Consumer<String> downstream;
        private final ByteArrayOutputStream pending = new ByteArrayOutputStream();
        private final CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);

        Utf8TokenDecoder(Consumer<String> downstream) {
            this.downstream = downstream;
        }

        void accept(byte[] bytes) {
            pending.writeBytes(bytes);
            decodePending();
        }

        /** Emits any accumulated incomplete bytes as-is, resetting the pending buffer.
         *  Called at channel transitions (e.g. think→content) to prevent stale bytes
         *  from polluting the next channel's decoding. */
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

    private static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }
}
