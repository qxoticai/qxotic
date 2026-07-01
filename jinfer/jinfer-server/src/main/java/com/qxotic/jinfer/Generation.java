package com.qxotic.jinfer;

import com.qxotic.jinfer.Engine.GenerationResult;
import com.qxotic.jinfer.Engine.StopSpec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.function.IntConsumer;

/**
 * The inference service: turns a parsed request into a {@link GenerationResult}, with streaming
 * sinks for the live channels. Owns the model, the prompt cache, and the in-place chat-session
 * state, and internally handles sampler/grammar/think wiring, stop conditions, session resume,
 * tool-call seeding/parsing, and prompt-cache warming. Transport-agnostic — endpoint handlers and
 * SSE streaming live in {@link Server}; this is the domain core they drive.
 */
final class Generation {

    private final ModelLegacy model;
    private final LLMOptions options;
    private final Worker worker;
    private final PromptCache promptCache;   // null when caching is disabled / unsupported by the model
    private final Map<LFMTokenizer, int[]> newlineCache = Collections.synchronizedMap(new WeakHashMap<>());
    private ChatSession chatSession;         // single generation worker, so a plain field suffices

    Generation(ModelLegacy model, LLMOptions options, Worker worker) throws IOException {
        this.model = model;
        this.options = options;
        this.worker = worker;
        this.promptCache = createPromptCache(model);
    }

    /** The prompt cache, or null when disabled/unsupported (for /props and /metrics). */
    PromptCache cache() {
        return promptCache;
    }

    /** The prompt cache is an opt-in model capability (only models whose KV/conv layout the cache
     *  understands provide it — today LFM2.5); other models run the plain, un-cached path. */
    private static PromptCache createPromptCache(ModelLegacy model) throws IOException {
        PromptCacheSupport cacheSupport = RuntimeFlags.PROMPT_CACHE && model instanceof Llama llama ? llama.promptCacheSupport().orElse(null) : null;
        if (cacheSupport == null) return null;
        CacheStore store;
        String cacheFile = RuntimeFlags.PROMPT_CACHE_FILE;
        if (cacheFile != null) {
            store = CacheStore.mmap(Path.of(cacheFile),
                    RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES,
                    RuntimeFlags.PROMPT_CACHE_BLOCK_TOKENS, cacheSupport.kvBytesPerToken());
        } else {
            store = CacheStore.inMemory();
        }
        PromptCache cache = cacheSupport.create(store);
        System.out.printf("Prompt cache enabled: budget=%d MB %s%n",
                RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES >> 20,
                cacheFile != null ? "file=" + cacheFile : "(in memory)");
        return cache;
    }

    // ---- chat / completion -------------------------------------------------

    @SuppressWarnings("unchecked")
    GenerationResult chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        LFMTokenizer tokenizer = model.tokenizer();
        ChatContext chatContext = new ChatContext(
                messages,
                ToolUse.offered(request) ? Values.asArray(request.get("tools"), "tools") : null,
                request.get("tool_choice"),
                true,
                requestThink(request),
                request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs ? (Map<String, Object>) kwargs : null);
        List<Integer> promptTokens = new ArrayList<>(model.chatFormat().encode(chatContext));
        ToolUse.seedForced(tokenizer, request, promptTokens);
        if (System.getProperty("jinfer.debugPrompt") != null) {
            System.err.println("[prompt] " + tokenizer.decode(promptTokens));
        }
        // Incremental in-place session resume (ChatML delta encoding) is a capability of the same
        // models that support the prompt cache, but it works independently of whether the radix cache
        // is actually enabled; other models re-encode the full prompt each turn.
        boolean sessionResume = model instanceof Llama llama && llama.promptCacheSupport().isPresent();
        Ingestion resumed = sessionResume
                ? matchChatSession(request, messages, new LFMChatFormat(tokenizer), tokenizer) : null;
        Ingestion ingestion = resumed != null ? resumed : Ingestion.of(model.createNewState(), 0, promptTokens);
        GenerationResult result = generate(request, promptTokens, ingestion, model.stopTokens(), sinks);
        if (sessionResume) saveChatSession(request, messages, ingestion, result);
        return ToolUse.offered(request) ? ToolUse.parse(model, result, request) : result;
    }

    GenerationResult completion(Map<String, Object> request, String prompt, Sinks sinks) {
        List<Integer> promptTokens = options.rawPrompt() ? model.tokenizer().encodeWithSpecialTokens(prompt) : new ArrayList<>(model.tokenizer().encode(prompt));
        Ingestion ingestion = Ingestion.of(model.createNewState(), 0, promptTokens);
        return generate(request, promptTokens, ingestion, model.stopTokens(), sinks);
    }

    /** Request fields to {@link Engine.Params}/{@link Engine.Listener}, then one engine pass through
     *  {@link #runServerGeneration}. Streaming counters mirror Engine's final usage: generated tokens
     *  are counted unless they are the trailing stop token removed from the result. */
    private GenerationResult generate(Map<String, Object> request, List<Integer> promptTokens,
                                      Ingestion ingestion, Set<Integer> baseStopTokens, Sinks sinks) {
        OpenAiSchema.Usage usageCounts = sinks.usage();
        float temperature = Values.floatValue(request.get("temperature"), options.temperature());
        float topp = Values.floatValue(request.get("top_p"), options.topp());
        long seed = Values.longValue(request.get("seed"), options.seed());
        int maxTokens = Values.intValue(request.getOrDefault("max_tokens", request.get("max_completion_tokens")), options.maxTokens());
        // server-side completion-token ceiling: an unbounded (or oversized) request can never run
        // the worker past llama.serverMaxTokens; hitting it reports finish_reason "length"
        if (RuntimeFlags.SERVER_MAX_TOKENS > 0)
            maxTokens = maxTokens < 0 ? RuntimeFlags.SERVER_MAX_TOKENS : Math.min(maxTokens, RuntimeFlags.SERVER_MAX_TOKENS);
        // defense in depth: server requests were already checked by validateGenerationParams on the
        // handler thread; these guards keep the method safe for any future non-HTTP caller
        LLMOptions.require(Values.intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        LLMOptions.require(0 <= maxTokens, "Invalid argument: max_tokens must be non-negative");
        StopSpec stops = stopSpec(request.get("stop"), baseStopTokens);
        boolean think = requestThink(request);
        Sampler sampler = Engine.configuredSampler(model, think, temperature, topp, seed);
        if (think) {
            // thinking models starve the answer under tight budgets: cap the think span, by default
            // at half the completion budget (request reasoning_max_tokens overrides; -1 = uncapped)
            int reasoningBudget = Values.intValue(request.get("reasoning_max_tokens"),
                    maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1);
            sampler = Engine.withThinkBudget(sampler, model.tokenizer(), reasoningBudget);
        }
        Grammar.Cursor grammarCursor = buildGrammarCursor(model.tokenizer(), request);
        if (grammarCursor != null) {
            Map<String, Integer> specials = model.tokenizer().getSpecialTokens();
            int eosToken = specials.getOrDefault("<eos>", specials.getOrDefault("<|endoftext|>", 2));
            // For a reasoning request, hold the grammar until after the think span — constraining
            // from token 0 would suppress the model's reasoning and degrade the answer. The newline
            // the model emits between </think> and the answer is passed through, not consumed by the
            // grammar (so non-ws-tolerant grammars like choice/jsonCompact see a clean first token).
            Integer thinkClose = specials.get("</think>");
            int grammarGate = think && thinkClose != null ? thinkClose : -1;
            int[] skipNl = grammarGate >= 0 ? newlineTokens(model.tokenizer()) : null;
            sampler = Sampler.withGrammar(sampler, grammarCursor, eosToken, grammarGate, skipNl);
        }
        int consumedPromptTokens = Engine.consumedPromptTokens(model.tokenizer(), promptTokens); // client-facing usage counts
        int[] cachedOut = {Math.min(ingestion.startPosition(), consumedPromptTokens)};
        IntConsumer onToken = usageCounts == null ? null : token -> {
            usageCounts.cachedTokens = Math.min(cachedOut[0], consumedPromptTokens);
            if (!stops.tokenStops().contains(token)) usageCounts.completionTokens++;
        };
        if (usageCounts != null) usageCounts.promptTokens = consumedPromptTokens;
        Engine.Params params = new Engine.Params(sampler, maxTokens, RuntimeFlags.SERVER_REQUEST_TIMEOUT_NANOS, stops, inlineReasoning(request));
        Engine.Listener listener = new Engine.Listener(onToken, sinks.onText(), sinks.onReasoning(), sinks.onToolCall());
        GenerationResult result = runServerGeneration(ingestion, params, listener, cachedOut, false);
        Metrics.record(result);
        return result;
    }

    /**
     * Server generation driver over {@link Engine#generate}. The ingestion plan says where to start:
     * a fresh state at position 0 (with the prompt cache plugged in through {@link GenerationHooks}),
     * or a resumed chat session mid-context — in which case the cache is bypassed (its tree is keyed
     * by full-stream positions). cachedOut[0] tracks the resumed-prefix length for streaming usage.
     */
    private GenerationResult runServerGeneration(Ingestion ingestion, Engine.Params params,
                                                 Engine.Listener listener, int[] cachedOut, boolean warm) {
        InferenceState state = ingestion.state();
        if (promptCache == null || ingestion.startPosition() > 0) {
            return Engine.generate(model, state, ingestion.startPosition(), ingestion.tokens(), params, listener, GenerationHooks.NONE);
        }
        // The prompt cache owns its own resume/commit policy; we just drive Engine.generate with it
        // as the hooks, commit the final frontier on success, and release per-request state.
        PromptCache.CacheRun run = promptCache.beginGeneration(state, cachedOut, warm);
        try {
            GenerationResult result = Engine.generate(model, state, 0, ingestion.tokens(), params, listener, run);
            run.commitFinal();
            return result;
        } finally {
            run.cleanup();
        }
    }

    // ---- in-place chat session resume --------------------------------------

    /** What to feed the generation loop: which state, from which position, with which tokens. Fresh
     *  requests: new state, position 0, the full prompt. Resumed chat sessions: the live state
     *  mid-context with only the delta (turn close + new messages + generation prompt). */
    private record Ingestion(InferenceState state, int startPosition, List<Integer> tokens, int prefillPositions) {
        static Ingestion of(InferenceState state, int startPosition, List<Integer> tokens) {
            return new Ingestion(state, startPosition, tokens, Engine.prefillPositions(state, startPosition, tokens));
        }
    }

    /**
     * The live state of the most recent chat generation. A follow-up request resumes it IN PLACE
     * when its messages extend the session's conversation (same prefix, the assistant echo equal to
     * what we returned, same tools): only the delta is ingested — no re-prefill, no cache restore,
     * token-exact even with thinking/surrogate tokens in the stream, because history is never
     * re-encoded. Single generation worker thread, so a plain field suffices; an unrelated request
     * simply replaces the session.
     */
    private record ChatSession(InferenceState state, int position, List<String> messageKeys, String reply, String toolsKey) {}

    private static String messageKey(Object message) {
        Map<String, Object> map = Values.asObject(message, "message");
        return Values.stringValue(map.get("role"), "user") + "\u0000" + ChatFormats.chatMessageContent(map);
    }

    private static String toolsKey(Map<String, Object> request) {
        if (!ToolUse.offered(request)) return "";
        return JsonCodec.stringify(request.get("tools")) + "|" + JsonCodec.stringify(request.get("tool_choice"));
    }

    /** Delta tokens to resume the session with this request, or null when it does not extend it. */
    private Ingestion matchChatSession(Map<String, Object> request, List<Object> messages,
                                       LFMChatFormat chatFormat, LFMTokenizer tokenizer) {
        ChatSession s = chatSession;
        if (s == null || !toolsKey(request).equals(s.toolsKey())) return null;
        int n = s.messageKeys().size();
        if (messages.size() < n + 2) return null;
        for (int i = 0; i < n; i++) {
            if (!messageKey(messages.get(i)).equals(s.messageKeys().get(i))) return null;
        }
        Map<String, Object> echo = Values.asObject(messages.get(n), "message");
        if (!"assistant".equals(Values.stringValue(echo.get("role"), ""))
                || !ChatFormats.chatMessageContent(echo).strip().equals(s.reply().strip())) return null;
        chatSession = null; // taken: a failure mid-resume must not leave a stale session behind
        // the session's un-ingested stop token (state.latestToken) closes the assistant turn;
        // generate() prepends it, we add the template newline and the new turns
        List<Integer> delta = new ArrayList<>(tokenizer.encode("\n"));
        for (int i = n + 1; i < messages.size(); i++) {
            delta.addAll(ChatFormats.encodeChatTurn(chatFormat, messages.get(i)));
        }
        delta.addAll(chatFormat.encodeGenerationPrompt());
        if (!requestThink(request)) chatFormat.appendThinkSurrogate(delta);
        ToolUse.seedForced(tokenizer, request, delta);
        return Ingestion.of(s.state(), s.position(), delta);
    }

    /** Remember the live state for instant resume of the next turn; only clean stop-terminated text
     *  replies are resumable (tool calls and aborted/length-capped replies are not). */
    private void saveChatSession(Map<String, Object> request, List<Object> messages,
                                 Ingestion ingestion, GenerationResult result) {
        if (!"stop".equals(result.finishReason()) || !result.toolCalls().isEmpty() || result.text().isBlank()) {
            return;
        }
        int position = ingestion.prefillPositions() + result.completionTokens();
        List<String> keys = new ArrayList<>(messages.size());
        for (Object message : messages) keys.add(messageKey(message));
        chatSession = new ChatSession(ingestion.state(), position, keys, result.text(), toolsKey(request));
    }

    // ---- sampler / grammar / stop / think wiring ---------------------------

    /** Builds a grammar cursor from request params: {@code grammar} (GBNF string) or
     *  {@code response_format: {type: "json_object"}}. Returns null when no constraint. */
    private static Grammar.Cursor buildGrammarCursor(LFMTokenizer tokenizer, Map<String, Object> request) {
        if (!RuntimeFlags.GRAMMAR) return null;
        Object gbnf = request.get("grammar");
        if (gbnf instanceof String s && !s.isBlank()) {
            return Grammar.of(s, tokenizer).cursor();
        }
        Object fmt = request.get("response_format");
        if (fmt instanceof Map<?, ?> f && "json_object".equals(f.get("type"))) {
            return Grammar.json(tokenizer).cursor();
        }
        return null;
    }

    /** Token ids that decode to newlines only (LF/CR), per tokenizer (cached). The chat template
     *  emits {@code </think>\n} before the answer; these are passed through untouched so the
     *  boilerplate newline is not consumed by the grammar (no whitespace baked into the language). */
    private int[] newlineTokens(LFMTokenizer tok) {
        return newlineCache.computeIfAbsent(tok, t -> {
            List<Integer> ids = new ArrayList<>();
            for (int i = 0, n = t.vocabularySize(); i < n; i++) {
                byte[] b = t.decodeTokenBytes(i);
                if (b.length == 0) continue;
                boolean nl = true;
                for (byte x : b) { int c = x & 0xFF; if (c != '\n' && c != '\r') { nl = false; break; } }
                if (nl) ids.add(i);
            }
            int[] arr = new int[ids.size()];
            for (int i = 0; i < arr.length; i++) arr[i] = ids.get(i);
            return arr;
        });
    }

    /** User stop strings stay TEXT stops only: token stops end generation anywhere, including inside
     *  the think span, while text stops are matched against content alone. */
    private static StopSpec stopSpec(Object value, Set<Integer> baseStopTokens) {
        List<String> textStops = new ArrayList<>();
        if (value instanceof String s) {
            if (!s.isEmpty()) textStops.add(s);
        } else if (value instanceof List<?> values) {
            for (Object item : values) {
                String stop = Values.stringValue(item, "");
                if (!stop.isEmpty()) textStops.add(stop);
            }
        } else if (value != null) {
            throw new IllegalArgumentException("stop must be a string or an array of strings");
        }
        return new StopSpec(Collections.unmodifiableSet(baseStopTokens), List.copyOf(textStops));
    }

    /** Effective thinking switch for a server request: chat_template_kwargs.enable_thinking
     *  (llama.cpp convention) overrides the CLI --think flag. Forced tool calls never think — the
     *  call marker is seeded as the first assistant token. */
    private boolean requestThink(Map<String, Object> request) {
        if (ToolUse.forced(request) != null) {
            return false;
        }
        if (request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs
                && kwargs.get("enable_thinking") instanceof Boolean enabled) {
            return enabled;
        }
        return options.think();
    }

    /** llama.cpp-compatible reasoning_format: "none" = leave thinking inline in content (with literal
     *  <think> markers) instead of routing it to the reasoning_content channel — lets vanilla OpenAI
     *  clients that only render content show thinking live. */
    private static boolean inlineReasoning(Map<String, Object> request) {
        return "none".equals(Values.stringValue(request.get("reasoning_format"), null));
    }

    // ---- prompt-cache warming ----------------------------------------------

    /**
     * Pre-ingests --warm-prompt / -Dllama.promptCacheWarm files into the prompt cache with FULLY
     * DENSE retention and sticky (eviction-exempt) nodes: requests diverging at ANY position inside a
     * warmed prompt resume token-exact, with zero re-ingest. Each file is warmed in two stream forms —
     * chat-template system message and raw completion prompt. Runs on the generation worker (cache
     * blobs are confined to it) and blocks until done.
     */
    void warm() {
        List<String> files = new ArrayList<>(options.warmPrompts());
        if (RuntimeFlags.PROMPT_CACHE_WARM != null) {
            for (String f : RuntimeFlags.PROMPT_CACHE_WARM.split(",")) {
                if (!f.isBlank()) files.add(f.strip());
            }
        }
        if (files.isEmpty()) return;
        if (promptCache == null) {
            System.err.println("warm-prompt ignored: prompt cache is disabled");
            return;
        }
        worker.runToCompletion(() -> {
            try {
                for (String file : files) {
                    warmFile(file);
                }
            } catch (Exception e) {
                System.err.println("warm-prompt failed: " + e);
            }
        });
    }

    private void warmFile(String file) throws IOException {
        String text = Files.readString(Path.of(file));
        LFMTokenizer tokenizer = model.tokenizer();
        ChatContext warm = new ChatContext(
                List.of(Map.of("role", "system", "content", text)), null, null, true, false, null);
        List<Integer> chatForm = ChatFormats.forModel(tokenizer).encode(warm);
        List<Integer> rawForm = tokenizer.encode(text);
        Engine.Params params = new Engine.Params(Sampler.ARGMAX, 0, 0, new StopSpec(Set.of(), List.of()), false); // warm: no deadline
        for (List<Integer> tokens : List.of(chatForm, rawForm)) {
            long startNanos = System.nanoTime();
            Ingestion ingestion = Ingestion.of(model.createNewState(), 0, tokens);
            runServerGeneration(ingestion, params, new Engine.Listener(null, null, null, null), new int[1], true);
            System.out.printf("warm-prompt %s: %d tokens in %.1f s%n",
                    file, ingestion.prefillPositions(), (System.nanoTime() - startNanos) / 1e9);
        }
    }
}
