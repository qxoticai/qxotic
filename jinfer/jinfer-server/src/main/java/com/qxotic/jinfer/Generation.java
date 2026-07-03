package com.qxotic.jinfer;

import com.qxotic.jinfer.Generator.GenerationResult;
import com.qxotic.jinfer.Generator.StopSpec;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.KvCodec;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.function.IntConsumer;

/**
 * The inference service: turns a parsed request into a {@link GenerationResult}, with streaming
 * sinks for the live channels. Owns the model and drives generation through the new-API
 * {@link Generator}: sampler / grammar / think wiring, stop conditions, tool-call seeding/parsing.
 * Transport-agnostic - endpoint handlers and SSE streaming live in {@link Server}.
 *
 * <p>Prompt caching: when the model exposes a {@link TurnTemplate} and a {@link KvCodec} (the S1
 * capability seams) and the request is plain chat (no tools), the conversation is lowered through
 * the hand-written template and served through a per-model {@link PromptCache}, so a follow-up
 * request that echoes the prior turns resumes their KV instead of re-prefilling. Requests with
 * tools, and models without those seams, take the legacy whole-render path on a fresh state.
 */
final class Generation {

    private final LanguageModel<?, ?, ?> model;
    private final LLMOptions options;
    private final Worker worker;
    private final Map<GgufTokenizer, int[]> newlineCache = Collections.synchronizedMap(new WeakHashMap<>());
    // Per-model prompt cache for the TurnTemplate path (one model per server); null until first use
    // or when the model lacks the seams / caching is disabled. Guarded by the model monitor.
    private PromptCache<?> promptCache;
    private boolean promptCacheResolved;

    Generation(LanguageModel<?, ?, ?> model, LLMOptions options, Worker worker) {
        this.model = model;
        this.options = options;
        this.worker = worker;
    }

    // ---- chat / completion -------------------------------------------------

    @SuppressWarnings("unchecked")
    GenerationResult chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        GgufTokenizer tokenizer = model.tokenizer();
        // Fast path: hand-written TurnTemplate + per-model prompt cache, for plain chat only.
        if (!ToolUse.offered(request) && RuntimeFlags.PROMPT_CACHE
                && model.turnTemplate().isPresent() && cache() != null) {
            List<Message> conversation = toConversation(messages);
            if (conversation != null) {
                return chatCached(model, request, conversation, sinks);
            }
        }
        ChatContext chatContext = new ChatContext(
                messages,
                ToolUse.offered(request) ? Values.asArray(request.get("tools"), "tools") : null,
                true,
                requestThink(request),
                request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs ? (Map<String, Object>) kwargs : null);
        List<Integer> promptTokens = new ArrayList<>(ChatFormats.forModel(tokenizer).encode(chatContext));
        ToolUse.seedForced(tokenizer, request, promptTokens);
        if (System.getProperty("jinfer.debugPrompt") != null) {
            System.err.println("[prompt] " + tokenizer.decode(promptTokens));
        }
        GenerationResult result = generate(request, promptTokens, model.stopTokens(), sinks);
        return ToolUse.offered(request) ? ToolUse.parse(model, result, request) : result;
    }

    /** Maps OpenAI messages to {@link Message}s, or null when a message can't be represented on the
     *  TurnTemplate path (tool role / tool_calls / non-text content) - the caller falls back. */
    private List<Message> toConversation(List<Object> messages) {
        List<Message> out = new ArrayList<>(messages.size());
        for (Object raw : messages) {
            if (!(raw instanceof Map<?, ?> m)) return null;
            String role = Values.stringValue(m.get("role"), "user");
            if ("tool".equals(role) || m.get("tool_calls") != null || m.get("function_call") != null) return null;
            String content = Values.messageContent(m.get("content"));
            out.add(new Message(new Role(role), content));
        }
        return out;
    }

    /** The TurnTemplate + PromptCache chat path: lower the conversation to the model's own framing,
     *  resume the longest cached prefix into a fresh state (skipping that prefill), ingest only the
     *  delta (caching it for the next turn), then decode from the retained logits. Assistant history
     *  is re-tokenized from text - best-effort, but the framing is byte-exact with the model's Jinja
     *  template (validated by the per-model oracle), so a stable client echo reuses the whole prefix. */
    private <S extends RuntimeState> GenerationResult chatCached(
            LanguageModel<?, ?, S> m, Map<String, Object> request, List<Message> conversation, Sinks sinks) {
        TurnTemplate template = m.turnTemplate().orElseThrow();
        boolean think = requestThink(request);
        // Turn-aligned blocks: conversation-start, each turn, and the generation prompt are separate
        // groups, so a follow-up request that diverges after turn k still reuses turns 0..k-1 (blocks
        // match completely, so one giant block would be unusable the moment the conversation grows).
        List<List<Batch>> groups = new ArrayList<>();
        groups.add(template.conversationStart());
        for (Message msg : conversation) groups.add(template.encodeTurn(msg));
        groups.add(template.generationPrompt(think));

        long[] fingerprints = tokenFingerprints(groups.stream().flatMap(List::stream).toList());
        int total = fingerprints.length;
        @SuppressWarnings("unchecked")
        PromptCache<S> cache = (PromptCache<S>) cache();
        S state = m.newState(m.config().contextLength());
        CachedSession<S> session = CachedSession.resume(m, cache, state, fingerprints);
        int restored = session.position();                       // cache hit length (0 = cold), on a group boundary

        int pos = 0;
        for (List<Batch> group : groups) {                       // ingest+commit each group past the hit
            int len = tokenFingerprints(group).length;
            if (pos + len <= restored) { pos += len; continue; } // wholly cached
            session.ingest(group);
            pos += len;
        }
        if (restored == total) {                                 // whole prompt cached: re-ingest the last
            state.resumeAt(total - 1);                           // token so the logits are fresh (no commit)
            m.ingest(state, Batch.prefill(new int[]{(int) fingerprints[total - 1]}));
        }
        if (System.getProperty("jinfer.debugPrompt") != null) {
            System.err.printf("[prompt-cache] %d/%d positions restored (%s)%n", restored, total, cache.stats());
        }
        // decode from the retained logits: empty prompt continues at the cursor. Restamp the usage
        // counts on the result (the non-streaming source of truth): the whole prompt was billed,
        // of which `restored` came from the cache.
        GenerationResult r = generateFrom(m, state, request, sinks, restored);
        return new GenerationResult(r.tokens(), r.stopToken(), r.text(), r.reasoning(), r.toolCalls(),
                total, r.completionTokens(), restored, r.finishReason(), r.promptMillis(), r.predictedMillis());
    }

    /** The per-model prompt cache, built once (null if the model lacks a KvCodec). */
    private synchronized PromptCache<?> cache() {
        if (!promptCacheResolved) {
            promptCacheResolved = true;
            promptCache = buildCache(model, options.modelPath());
        }
        return promptCache;
    }

    private static <S extends RuntimeState> PromptCache<S> buildCache(LanguageModel<?, ?, S> m, java.nio.file.Path modelPath) {
        KvCodec<S> codec = m.kvCodec().orElse(null);
        if (codec == null) return null;
        return new PromptCache<>(codec, com.qxotic.jinfer.CacheStore.inMemory(),
                RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES, PromptCache.modelSeed(modelPath));
    }

    /** The token ids a batch list ingests, as fingerprints (token-only; the TurnTemplate path is text). */
    private static long[] tokenFingerprints(List<Batch> batches) {
        int n = 0;
        for (Batch b : batches) n += ((Batch.Input.Tokens) b.input()).ids().length;
        long[] fp = new long[n];
        int i = 0;
        for (Batch b : batches) for (int id : ((Batch.Input.Tokens) b.input()).ids()) fp[i++] = id;
        return fp;
    }

    GenerationResult completion(Map<String, Object> request, String prompt, Sinks sinks) {
        GgufTokenizer tokenizer = model.tokenizer();
        List<Integer> promptTokens = options.rawPrompt()
                ? new ArrayList<>(tokenizer.encodeWithSpecialTokens(prompt))
                : new ArrayList<>(tokenizer.encode(prompt));
        return generate(request, promptTokens, model.stopTokens(), sinks);
    }

    /** Request fields to {@link Generator.Params}/{@link Generator.Listener}, then one pass through the
     *  new-API {@link Generator}. Streaming counters mirror the final usage: generated tokens are
     *  counted unless they are the trailing stop token removed from the result. */
    private GenerationResult generate(Map<String, Object> request, List<Integer> promptTokens,
                                      Set<Integer> baseStopTokens, Sinks sinks) {
        return generate(request, promptTokens, baseStopTokens, sinks, null, 0);
    }

    /** Decode a request onto an already-resumed state (empty prompt, the state continues at its
     *  cursor); {@code cachedTokens} is the restored prefix length billed to the client. */
    private <S extends RuntimeState> GenerationResult generateFrom(
            LanguageModel<?, ?, S> m, S state, Map<String, Object> request, Sinks sinks, int cachedTokens) {
        return generate(request, List.of(), m.stopTokens(), sinks, state, cachedTokens);
    }

    private GenerationResult generate(Map<String, Object> request, List<Integer> promptTokens,
                                      Set<Integer> baseStopTokens, Sinks sinks, RuntimeState resumedState, int cachedTokens) {
        GgufTokenizer tokenizer = model.tokenizer();
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
        Sampler sampler = Generator.configuredSampler(model, think, temperature, topp, seed);
        if (think) {
            // thinking models starve the answer under tight budgets: cap the think span, by default
            // at half the completion budget (request reasoning_max_tokens overrides; -1 = uncapped)
            int reasoningBudget = Values.intValue(request.get("reasoning_max_tokens"),
                    maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1);
            sampler = Generator.withThinkBudget(sampler, tokenizer, reasoningBudget);
        }
        Grammar.Cursor grammarCursor = buildGrammarCursor(tokenizer, request);
        if (grammarCursor != null) {
            Map<String, Integer> specials = tokenizer.getSpecialTokens();
            int eosToken = specials.getOrDefault("<eos>", specials.getOrDefault("<|endoftext|>", 2));
            // For a reasoning request, hold the grammar until after the think span - constraining
            // from token 0 would suppress the model's reasoning and degrade the answer. The newline
            // the model emits between </think> and the answer is passed through, not consumed by the
            // grammar (so non-ws-tolerant grammars like choice/jsonCompact see a clean first token).
            Integer thinkClose = specials.get("</think>");
            int grammarGate = think && thinkClose != null ? thinkClose : -1;
            int[] skipNl = grammarGate >= 0 ? newlineTokens(tokenizer) : null;
            sampler = Sampler.withGrammar(sampler, grammarCursor, eosToken, grammarGate, skipNl);
        }
        // Billed prompt: the whole conversation. On the cached path the state is pre-resumed to the
        // full prompt (position == total), of which cachedTokens were restored from the cache.
        int billedPrompt = resumedState != null ? resumedState.position()
                : Generator.consumedPromptTokens(tokenizer, promptTokens);
        IntConsumer onToken = usageCounts == null ? null : token -> {
            usageCounts.cachedTokens = cachedTokens;
            if (!stops.tokenStops().contains(token)) usageCounts.completionTokens++;
        };
        if (usageCounts != null) usageCounts.promptTokens = billedPrompt;
        Generator.Params params = new Generator.Params(sampler, maxTokens, RuntimeFlags.SERVER_REQUEST_TIMEOUT_NANOS, stops, inlineReasoning(request));
        Generator.Listener listener = new Generator.Listener(onToken, sinks.onText(), sinks.onReasoning(), sinks.onToolCall());
        GenerationResult result = runGen(model, promptTokens, params, listener, resumedState);
        Metrics.record(result);
        return result;
    }

    /** One new-API generation pass: a fresh state prefills {@code promptTokens}, or a pre-resumed
     *  {@code state} (from the prompt cache) continues from its cursor with an empty prompt. */
    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> GenerationResult runGen(LanguageModel<?, ?, S> m, List<Integer> promptTokens,
                                                             Generator.Params params, Generator.Listener listener, RuntimeState resumedState) {
        S state = resumedState != null ? (S) resumedState
                : m.newState(m.config().contextLength(), Math.max(promptTokens.size(), 16));
        return Generator.generate(m, state, promptTokens, params, listener);
    }

    // ---- sampler / grammar / stop / think wiring ---------------------------

    /** Builds a grammar cursor from request params: {@code grammar} (GBNF string) or
     *  {@code response_format: {type: "json_object"}}. Returns null when no constraint. */
    private static Grammar.Cursor buildGrammarCursor(GgufTokenizer tokenizer, Map<String, Object> request) {
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
    private int[] newlineTokens(GgufTokenizer tok) {
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
     *  (llama.cpp convention) overrides the CLI --think flag. Forced tool calls never think - the
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
     *  <think> markers) instead of routing it to the reasoning_content channel - lets vanilla OpenAI
     *  clients that only render content show thinking live. */
    private static boolean inlineReasoning(Map<String, Object> request) {
        return "none".equals(Values.stringValue(request.get("reasoning_format"), null));
    }

    // ---- prompt-cache warming (disabled on the new-API path) ---------------

    /** Prompt-cache warming is not available on the new-API path; warns if warm prompts were requested. */
    void warm() {
        List<String> files = new ArrayList<>(options.warmPrompts());
        if (!files.isEmpty()) {
            System.err.println("warm-prompt ignored: prompt cache is not available on the new-API path");
        }
    }
}
