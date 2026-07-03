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
 * <p>Plain chat on a model with a {@link TurnTemplate} is lowered through the hand-written
 * template (injection-inert, oracle-validated framing); when the model also has a {@link KvCodec}
 * and caching is enabled, the conversation is served through a per-model {@link PromptCache}, so a
 * follow-up request that echoes the prior turns resumes their KV instead of re-prefilling.
 * Requests with tools, and models without a template, take the whole-render fallback.
 */
final class Generation {

    private final LanguageModel<?, ?, ?> model;
    private final LLMOptions options;
    private final Worker worker;
    private final Map<GgufTokenizer, int[]> newlineCache = Collections.synchronizedMap(new WeakHashMap<>());
    private final TurnTemplate template;         // memoized model framing, null when the model has none
    private final Set<Integer> stopTokens;       // memoized model stops
    private final PromptCache<?> promptCache;    // per-model cache; null without a KvCodec or when disabled

    Generation(LanguageModel<?, ?, ?> model, LLMOptions options, Worker worker) {
        this.model = model;
        this.options = options;
        this.worker = worker;
        this.template = model.turnTemplate().orElse(null);
        this.stopTokens = model.stopTokens();
        this.promptCache = RuntimeFlags.PROMPT_CACHE ? buildCache(model, options.modelPath()) : null;
    }

    private static <S extends RuntimeState> PromptCache<S> buildCache(LanguageModel<?, ?, S> m, java.nio.file.Path modelPath) {
        KvCodec<S> codec = m.kvCodec().orElse(null);
        if (codec == null) return null;
        return new PromptCache<>(codec, CacheStore.inMemory(),
                RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES, PromptCache.modelSeed(modelPath));
    }

    // ---- chat / completion -------------------------------------------------

    @SuppressWarnings("unchecked")
    GenerationResult chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        boolean tools = ToolUse.offered(request);
        // TurnTemplate path: the model's own framing, for plain chat - caching is a separate,
        // optional layer on top (the framing does not depend on it).
        if (!tools && template != null) {
            List<Message> conversation = toConversation(messages);
            if (conversation != null) {
                return chatTemplated(model, request, conversation, sinks);
            }
        }
        ChatContext chatContext = new ChatContext(
                messages,
                tools ? Values.asArray(request.get("tools"), "tools") : null,
                true,
                requestThink(request),
                request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs ? (Map<String, Object>) kwargs : null);
        List<Integer> promptTokens = new ArrayList<>(ChatFormat.encode(model.tokenizer(), chatContext));
        ToolUse.seedForced(model.tokenizer(), request, promptTokens);
        if (System.getProperty("jinfer.debugPrompt") != null) {
            System.err.println("[prompt] " + model.tokenizer().decode(promptTokens));
        }
        GenerationResult result = generate(request, promptTokens, sinks);
        return tools ? ToolUse.parse(model, result, request) : result;
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

    /** The TurnTemplate chat path: lower the conversation to the model's own framing and, when the
     *  prompt cache is available, resume the longest cached prefix into the state (skipping that
     *  prefill) and ingest only the delta, caching it for the next turn. Assistant history is
     *  re-tokenized from text - best-effort, but the framing is byte-exact with the model's Jinja
     *  template (validated by the per-model oracle), so a stable client echo reuses the whole prefix. */
    private <S extends RuntimeState> GenerationResult chatTemplated(
            LanguageModel<?, ?, S> m, Map<String, Object> request, List<Message> conversation, Sinks sinks) {
        boolean think = requestThink(request);
        // Turn-aligned blocks: conversation-start, each turn, and the generation prompt are separate
        // groups, so a follow-up request that diverges after turn k still reuses turns 0..k-1 (blocks
        // match completely, so one giant block would be unusable the moment the conversation grows).
        List<List<Batch>> groups = new ArrayList<>();
        groups.add(template.conversationStart());
        for (Message msg : conversation) groups.add(template.encodeTurn(msg));
        groups.add(template.generationPrompt(think));

        int[] groupLen = new int[groups.size()];
        int total = 0;
        for (int g = 0; g < groups.size(); g++) {
            groupLen[g] = groups.get(g).stream().mapToInt(Batch::count).sum();
            total += groupLen[g];
        }
        S state = m.newState(m.config().contextLength());
        int restored = 0;
        @SuppressWarnings("unchecked")
        PromptCache<S> cache = (PromptCache<S>) promptCache;
        if (cache != null) {
            long[] fingerprints = new long[total];
            int i = 0;
            for (List<Batch> group : groups) for (int id : Batch.tokenIds(group)) fingerprints[i++] = id;
            // Resume at most up to the final block (the generation prompt): a whole-prompt hit then
            // still re-ingests that block, leaving fresh logits at the cursor - no special case.
            CachedSession<S> session = CachedSession.resume(m, cache, state, fingerprints, total - groupLen[groupLen.length - 1]);
            restored = session.position();                       // cache hit length (0 = cold), on a group boundary
            int pos = 0;
            for (int g = 0; g < groups.size(); g++) {            // ingest+commit each group past the hit
                if (pos + groupLen[g] <= restored) { pos += groupLen[g]; continue; }
                session.ingest(groups.get(g));
                pos += groupLen[g];
            }
            if (System.getProperty("jinfer.debugPrompt") != null) {
                System.err.printf("[prompt-cache] %d/%d positions restored (%s)%n", restored, total, cache.stats());
            }
        } else {
            List<Batch> all = new ArrayList<>();                 // uncached: plain ingest, framing unchanged
            for (List<Batch> group : groups) all.addAll(group);
            for (Batch b : Batch.prepare(all, state.batchCapacity())) m.ingest(state, b);
        }
        // decode from the retained logits (empty prompt continues at the cursor); the whole prompt
        // was billed, of which `restored` came from the cache.
        return generateFrom(m, state, request, sinks, restored).withUsage(total, restored);
    }

    GenerationResult completion(Map<String, Object> request, String prompt, Sinks sinks) {
        GgufTokenizer tokenizer = model.tokenizer();
        List<Integer> promptTokens = options.rawPrompt()
                ? new ArrayList<>(tokenizer.encodeWithSpecialTokens(prompt))
                : new ArrayList<>(tokenizer.encode(prompt));
        return generate(request, promptTokens, sinks);
    }

    /** One pass through the new-API {@link Generator}: a fresh state prefills the prompt. */
    private GenerationResult generate(Map<String, Object> request, List<Integer> promptTokens, Sinks sinks) {
        return runGeneration(model, null, request, promptTokens, sinks, 0);
    }

    /** Decode a request onto an already-resumed state (empty prompt, the state continues at its
     *  cursor); {@code cachedTokens} is the restored prefix length billed to the client. */
    private <S extends RuntimeState> GenerationResult generateFrom(
            LanguageModel<?, ?, S> m, S state, Map<String, Object> request, Sinks sinks, int cachedTokens) {
        return runGeneration(m, state, request, List.of(), sinks, cachedTokens);
    }

    /** Request fields to {@link Generator.Params}/{@link Generator.Listener}, then one generation
     *  pass. Streaming counters mirror the final usage: generated tokens are counted unless they
     *  are the trailing stop token removed from the result. */
    private <S extends RuntimeState> GenerationResult runGeneration(
            LanguageModel<?, ?, S> m, S resumedState, Map<String, Object> request,
            List<Integer> promptTokens, Sinks sinks, int cachedTokens) {
        GgufTokenizer tokenizer = m.tokenizer();
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
        StopSpec stops = stopSpec(request.get("stop"), stopTokens);
        boolean think = requestThink(request);
        Sampler sampler = Generator.configuredSampler(m, think, temperature, topp, seed);
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
        S state = resumedState != null ? resumedState
                : m.newState(m.config().contextLength(), Math.max(promptTokens.size(), 16));
        GenerationResult result = Generator.generate(m, state, promptTokens, params, listener);
        Metrics.record(result);
        return result;
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

    // ---- prompt-cache warming ----------------------------------------------

    /** Warm prompts are not implemented on this path yet (the cache itself is live; warming would
     *  pre-ingest and commit the given prompts at startup). Warns so the flag is not silently lost. */
    void warm() {
        if (!options.warmPrompts().isEmpty()) {
            System.err.println("warm-prompt ignored: startup warming is not implemented yet");
        }
    }
}
