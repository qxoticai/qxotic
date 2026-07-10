package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.SessionPool;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.Generator.GenerationResult;
import com.qxotic.jinfer.llm.Generator.StopSpec;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.function.IntConsumer;

/**
 * The inference service: turns a parsed request into a {@link GenerationResult}, with streaming
 * sinks for the live channels. Owns the model and drives generation through the new-API {@link
 * Generator}: sampler / grammar / think wiring, stop conditions, tool-call seeding/parsing.
 * Transport-agnostic - endpoint handlers and SSE streaming live in {@link Server}.
 *
 * <p>Plain chat on a model with a {@link TurnTemplate} is lowered through the hand-written template
 * (injection-inert, oracle-validated framing); when the model also has a {@link StateCodec} and
 * caching is enabled, the conversation is served through a per-model {@link PromptCache}, so a
 * follow-up request that echoes the prior turns resumes their KV instead of re-prefilling. Requests
 * with tools, and models without a template, take the whole-render fallback.
 */
final class Generation {

    private final LoadedModel<?> model;
    private final LLMOptions options;
    private final Worker worker;
    private final Map<GgufTokenizer, int[]> newlineCache =
            Collections.synchronizedMap(new WeakHashMap<>());
    private final TurnTemplate template; // memoized model framing, null when the model has none
    private final Set<Integer> stopTokens; // memoized model stops
    private final PromptCache<?>
            promptCache; // per-model cache; null without a StateCodec or when disabled
    private final SessionPool<?>
            sessionPool; // tier 1: last-N live conversations (append-only reuse)

    Generation(LoadedModel<?> model, LLMOptions options, Worker worker) {
        this.model = model;
        this.options = options;
        this.worker = worker;
        this.template = model.chatTemplate().orElse(null);
        this.stopTokens = model.stopTokens();
        this.promptCache =
                RuntimeFlags.PROMPT_CACHE ? buildCache(model, options.modelPath()) : null;
        this.sessionPool = promptCache != null ? new SessionPool<>(RuntimeFlags.SESSIONS) : null;
    }

    private static <S extends RuntimeState> PromptCache<S> buildCache(
            LoadedModel<S> m, java.nio.file.Path modelPath) {
        StateCodec<S> codec = m.model().stateCodec().orElse(null);
        if (codec == null) return null;
        return new PromptCache<>(
                codec,
                CacheStore.inMemory(),
                RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES,
                PromptCache.modelSeed(modelPath));
    }

    // ---- chat / completion -------------------------------------------------

    @SuppressWarnings("unchecked")
    GenerationResult chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        boolean tools = ToolUse.offered(request);
        // TurnTemplate path: the model's own framing, for plain chat and - when the template can
        // frame tools byte-exactly and the request does not FORCE a specific call (forcing seeds
        // the
        // prompt, still a whole-render concern) - for tool conversations too. Caching is a
        // separate,
        // optional layer on top (the framing does not depend on it).
        boolean toolTemplated =
                tools
                        && template != null
                        && template.supportsTools()
                        && ToolUse.forced(request) == null;
        if ((!tools || toolTemplated) && template != null && onlyKnownKwargs(request)) {
            List<Message> conversation = toConversation(messages);
            if (conversation != null) {
                List<Tool> toolList = toolTemplated ? buildTools(request) : List.of();
                return chatTemplated(model, request, conversation, toolList, sinks);
            }
        }
        Map<String, Object> kwargs =
                request.get("chat_template_kwargs") instanceof Map<?, ?> kw
                        ? (Map<String, Object>) kw
                        : null;
        List<Integer> promptTokens =
                new ArrayList<>(
                        new JinjaChatTemplate(model.tokenizer())
                                .render(
                                        messages,
                                        tools
                                                ? Values.asArray(request.get("tools"), "tools")
                                                : null,
                                        true,
                                        requestThink(request),
                                        kwargs));
        ToolUse.seedForced(model.tokenizer(), request, promptTokens);
        if (System.getProperty("jinfer.debugPrompt") != null) {
            System.err.println("[prompt] " + model.tokenizer().decode(promptTokens));
        }
        GenerationResult result = generate(request, promptTokens, sinks);
        return tools ? ToolUse.parse(model, result, request) : result;
    }

    /**
     * Maps OpenAI messages to {@link Message}s, or null when a message can't be represented on the
     * TurnTemplate path (tool role / tool_calls / non-text content) - the caller falls back.
     */
    private List<Message> toConversation(List<Object> messages) {
        List<Message> out = new ArrayList<>(messages.size());
        for (Object raw : messages) {
            if (!(raw instanceof Map<?, ?> m)) return null;
            String role = Values.stringValue(m.get("role"), "user");
            if (m.get("function_call") != null) return null; // legacy shape: whole-render
            Object raw2 = m.get("content");
            if (raw2 instanceof List<?> parts) { // multimodal content array: only
                for (Object part : parts) { // pure-text flattens faithfully
                    if (!(part instanceof Map<?, ?> pm) || !"text".equals(pm.get("type")))
                        return null;
                }
            }
            String content = Values.messageContent(raw2);
            List<Part> callParts = toolCallParts(m.get("tool_calls"));
            if (callParts == null) return null; // malformed tool_calls: whole-render
            if (!callParts.isEmpty()) {
                List<Part> all = new ArrayList<>();
                if (!content.isEmpty()) all.add(new Part.Text(content));
                all.addAll(callParts);
                out.add(new Message(new Role(role), all));
            } else {
                out.add(new Message(new Role(role), content));
            }
        }
        return out;
    }

    /**
     * An assistant message's {@code tool_calls} array to {@link Part.ToolCall} parts (empty when
     * absent), or null when the shape is unusable (so the caller falls back to whole-render). Each
     * call's {@code arguments} JSON string is parsed to an ordered map; a non-object leaves an
     * empty argument map.
     */
    @SuppressWarnings("unchecked")
    private static List<Part> toolCallParts(Object toolCalls) {
        if (toolCalls == null) return List.of();
        if (!(toolCalls instanceof List<?> calls)) return null;
        List<Part> parts = new ArrayList<>();
        for (Object c : calls) {
            if (!(c instanceof Map<?, ?> call)) return null;
            Object fn = call.get("function");
            if (!(fn instanceof Map<?, ?> f)) return null;
            String name = Values.stringValue(f.get("name"), null);
            if (name == null) return null;
            Object args = f.get("arguments");
            Map<String, Object> argMap = new LinkedHashMap<>();
            if (args instanceof String s && !s.isBlank()) {
                try {
                    if (JsonCodec.parse(s) instanceof Map<?, ?> parsed)
                        argMap.putAll((Map<String, Object>) parsed);
                } catch (RuntimeException notJson) {
                    /* leave empty */
                }
            } else if (args instanceof Map<?, ?> parsed) {
                argMap.putAll((Map<String, Object>) parsed);
            }
            parts.add(new Part.ToolCall(Values.stringValue(call.get("id"), ""), name, argMap));
        }
        return parts;
    }

    /**
     * The offered tools as {@link Tool}s, each carrying its request JSON canonicalized to the form
     * Jinja {@code tojson} produces (so a template that embeds it stays byte-exact).
     */
    private static List<Tool> buildTools(Map<String, Object> request) {
        List<Tool> out = new ArrayList<>();
        for (Object raw : Values.asArray(request.get("tools"), "tools")) {
            if (!(raw instanceof Map<?, ?> t)) continue;
            String name =
                    t.get("function") instanceof Map<?, ?> fn
                            ? Values.stringValue(fn.get("name"), "")
                            : "";
            if (!name.isEmpty()) out.add(new Tool(name, ToolCallSyntax.jinjaJson(t)));
        }
        return out;
    }

    /**
     * chat_template_kwargs the templated path can represent: only enable_thinking (mapped to the
     * generation prompt). Anything else must reach the Jinja render, so the request falls back.
     */
    private static boolean onlyKnownKwargs(Map<String, Object> request) {
        if (!(request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs)) return true;
        for (Object key : kwargs.keySet()) {
            if (!"enable_thinking".equals(key)) return false;
        }
        return true;
    }

    /**
     * The TurnTemplate chat path: lower the conversation to the model's own framing and, when the
     * prompt cache is available, resume the longest cached prefix into the state (skipping that
     * prefill) and ingest only the delta, caching it for the next turn. Assistant history is
     * re-tokenized from text - best-effort, but the framing is byte-exact with the model's Jinja
     * template (validated by the per-model oracle), so a stable client echo reuses the whole
     * prefix.
     */
    private <S extends RuntimeState> GenerationResult chatTemplated(
            LoadedModel<S> m,
            Map<String, Object> request,
            List<Message> conversation,
            List<Tool> tools,
            Sinks sinks) {
        boolean think = requestThink(request);
        conversation = template.normalize(conversation); // the template's own framing may inject
        // a mandatory system turn (NemotronH, Llama)
        // Turn-aligned blocks: conversation-start, each turn, and the generation prompt are
        // separate
        // groups, so a follow-up request that diverges after turn k still reuses turns 0..k-1
        // (blocks
        // match completely, so one giant block would be unusable the moment the conversation
        // grows).
        List<List<Batch>> groups = new ArrayList<>();
        List<Message> turns = conversation;
        if (!tools.isEmpty()) {
            // Tools weld into the preamble with the leading system message; encode them together so
            // the whole tool-block stays one turn-stable prefix, then skip the system turn below.
            java.util.Optional<Message> system =
                    !conversation.isEmpty() && conversation.get(0).role().equals(Role.SYSTEM)
                            ? java.util.Optional.of(conversation.get(0))
                            : java.util.Optional.empty();
            groups.add(template.conversationStart(new TurnTemplate.Preamble(system, tools)));
            turns =
                    system.isPresent()
                            ? conversation.subList(1, conversation.size())
                            : conversation;
        } else {
            groups.add(template.conversationStart());
        }
        for (Message msg : turns) groups.add(template.encodeTurn(msg));
        groups.add(template.generationPrompt(think));

        int[] groupLen = new int[groups.size()];
        int total = 0;
        for (int g = 0; g < groups.size(); g++) {
            groupLen[g] = groups.get(g).stream().mapToInt(Batch::count).sum();
            total += groupLen[g];
        }
        @SuppressWarnings("unchecked")
        PromptCache<S> cache = (PromptCache<S>) promptCache;
        if (cache == null) {
            S state = m.model().newState(m.model().config().contextLength());
            List<Batch> all = new ArrayList<>(); // uncached: plain ingest, framing unchanged
            for (List<Batch> group : groups) all.addAll(group);
            for (Batch b : Batch.prepare(all, state.batchCapacity())) m.model().ingest(state, b);
            return generateFrom(m, state, request, sinks, 0).withUsage(total, 0);
        }
        long[] fingerprints = new long[total];
        int i = 0;
        for (List<Batch> group : groups)
            for (int id : Batch.tokenIds(group)) fingerprints[i++] = id;
        // Tier 1: a live pooled session whose whole stream strictly prefixes this conversation
        // continues append-only (no restore at all). Otherwise tier 2: resume the longest block
        // prefix into a fresh state - at most up to the final block (the generation prompt), so a
        // whole-prompt hit still re-ingests that block, leaving fresh logits at the cursor.
        @SuppressWarnings("unchecked")
        SessionPool<S> pool = (SessionPool<S>) sessionPool;
        int billed = total;
        return pool.withSession(
                m.model(),
                cache,
                () -> m.model().newState(m.model().config().contextLength()),
                fingerprints,
                total,
                total - groupLen[groupLen.length - 1],
                (session, tier1) -> {
                    int restored = session.position(); // reused positions: a BLOCK boundary
                    session.ingestGroups(groups); // (or the pooled stream end), not
                    // necessarily a group one - the
                    // session slices the partial group
                    Metrics.recordPromptCache(tier1, restored);
                    if (System.getProperty("jinfer.debugPrompt") != null) {
                        System.err.printf(
                                "[prompt-cache] %s %d/%d positions reused (%s)%n",
                                tier1 ? "tier1-append" : "tier2-restore",
                                restored,
                                billed,
                                cache.stats());
                    }
                    // decode from the retained logits (empty prompt continues at the cursor); the
                    // whole
                    // prompt was billed, of which `restored` came from the cache.
                    GenerationResult result =
                            generateFrom(m, session.state(), request, sinks, restored)
                                    .withUsage(billed, restored);
                    // Bring the decode back into the session (the generator stepped the state
                    // directly):
                    // the reply extends the stream and commits as a block, and the live session
                    // returns
                    // to the pool ready for the next echo to continue append-only.
                    int ingested = session.state().position() - billed;
                    if (ingested > 0) session.adopt(result.tokens().subList(0, ingested));
                    return result;
                });
    }

    GenerationResult completion(Map<String, Object> request, String prompt, Sinks sinks) {
        GgufTokenizer tokenizer = model.tokenizer();
        List<Integer> promptTokens =
                options.rawPrompt()
                        ? new ArrayList<>(tokenizer.encodeWithSpecialTokens(prompt))
                        : new ArrayList<>(tokenizer.encode(prompt));
        return generate(request, promptTokens, sinks);
    }

    /** One pass through the new-API {@link Generator}: a fresh state prefills the prompt. */
    private GenerationResult generate(
            Map<String, Object> request, List<Integer> promptTokens, Sinks sinks) {
        return runGeneration(model, null, request, promptTokens, sinks, 0);
    }

    /**
     * Decode a request onto an already-resumed state (empty prompt, the state continues at its
     * cursor); {@code cachedTokens} is the restored prefix length billed to the client.
     */
    private <S extends RuntimeState> GenerationResult generateFrom(
            LoadedModel<S> m, S state, Map<String, Object> request, Sinks sinks, int cachedTokens) {
        return runGeneration(m, state, request, List.of(), sinks, cachedTokens);
    }

    /**
     * Request fields to {@link Generator.Params}/{@link Generator.Listener}, then one generation
     * pass. Streaming counters mirror the final usage: generated tokens are counted unless they are
     * the trailing stop token removed from the result.
     */
    private <S extends RuntimeState> GenerationResult runGeneration(
            LoadedModel<S> m,
            S resumedState,
            Map<String, Object> request,
            List<Integer> promptTokens,
            Sinks sinks,
            int cachedTokens) {
        GgufTokenizer tokenizer = m.tokenizer();
        OpenAiSchema.Usage usageCounts = sinks.usage();
        float temperature = Values.floatValue(request.get("temperature"), options.temperature());
        float topp = Values.floatValue(request.get("top_p"), options.topp());
        long seed = Values.longValue(request.get("seed"), options.seed());
        int maxTokens =
                Values.intValue(
                        request.getOrDefault("max_tokens", request.get("max_completion_tokens")),
                        options.maxTokens());
        // server-side completion-token ceiling: an unbounded (or oversized) request can never run
        // the worker past llama.serverMaxTokens; hitting it reports finish_reason "length"
        if (ServerFlags.SERVER_MAX_TOKENS > 0)
            maxTokens =
                    maxTokens < 0
                            ? ServerFlags.SERVER_MAX_TOKENS
                            : Math.min(maxTokens, ServerFlags.SERVER_MAX_TOKENS);
        // defense in depth: server requests were already checked by validateGenerationParams on the
        // handler thread; these guards keep the method safe for any future non-HTTP caller
        LLMOptions.require(Values.intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        LLMOptions.require(0 <= maxTokens, "Invalid argument: max_tokens must be non-negative");
        StopSpec stops = stopSpec(request.get("stop"), stopTokens);
        boolean think = requestThink(request);
        Sampler sampler =
                Generator.configuredSampler(
                        m.model(), m.tokenizer(), think, temperature, topp, seed);
        if (think) {
            // thinking models starve the answer under tight budgets: cap the think span, by default
            // at half the completion budget (request reasoning_max_tokens overrides; -1 = uncapped)
            int reasoningBudget =
                    Values.intValue(
                            request.get("reasoning_max_tokens"),
                            maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1);
            sampler = Generator.withThinkBudget(sampler, tokenizer, reasoningBudget);
        }
        Grammar.Cursor grammarCursor = buildGrammarCursor(tokenizer, request);
        if (grammarCursor != null) {
            Map<String, Integer> specials = tokenizer.getSpecialTokens();
            int eosToken =
                    specials.getOrDefault("<eos>", specials.getOrDefault("<|endoftext|>", 2));
            // For a reasoning request, hold the grammar until after the think span - constraining
            // from token 0 would suppress the model's reasoning and degrade the answer. The newline
            // the model emits between </think> and the answer is passed through, not consumed by
            // the
            // grammar (so non-ws-tolerant grammars like choice/jsonCompact see a clean first
            // token).
            Integer thinkClose = specials.get("</think>");
            int grammarGate = think && thinkClose != null ? thinkClose : -1;
            int[] skipNl = grammarGate >= 0 ? newlineTokens(tokenizer) : null;
            sampler = Sampler.withGrammar(sampler, grammarCursor, eosToken, grammarGate, skipNl);
        }
        // Billed prompt: the whole conversation. On the cached path the state is pre-resumed to the
        // full prompt (position == total), of which cachedTokens were restored from the cache.
        int billedPrompt =
                resumedState != null
                        ? resumedState.position()
                        : Generator.consumedPromptTokens(tokenizer, promptTokens);
        IntConsumer onToken =
                usageCounts == null
                        ? null
                        : token -> {
                            usageCounts.cachedTokens = cachedTokens;
                            if (!stops.tokenStops().contains(token)) usageCounts.completionTokens++;
                        };
        if (usageCounts != null) usageCounts.promptTokens = billedPrompt;
        Generator.Params params =
                new Generator.Params(
                        sampler,
                        maxTokens,
                        ServerFlags.SERVER_REQUEST_TIMEOUT_NANOS,
                        stops,
                        inlineReasoning(request),
                        ToolUse.offered(request)
                                ? m.chatTemplate().flatMap(TurnTemplate::toolCallDetector)
                                : java.util.Optional.empty());
        Generator.Listener listener =
                new Generator.Listener(onToken, sinks.onText(), sinks.onReasoning());
        S state =
                resumedState != null
                        ? resumedState
                        : m.model()
                                .newState(
                                        m.model().config().contextLength(),
                                        Math.max(promptTokens.size(), 16));
        GenerationResult result =
                Generator.generate(m.model(), m.tokenizer(), state, promptTokens, params, listener);
        Metrics.record(result);
        return result;
    }

    // ---- sampler / grammar / stop / think wiring ---------------------------

    /**
     * Builds a grammar cursor from request params: {@code grammar} (GBNF string) or {@code
     * response_format: {type: "json_object"}}. Returns null when no constraint.
     */
    private static Grammar.Cursor buildGrammarCursor(
            GgufTokenizer tokenizer, Map<String, Object> request) {
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

    /**
     * Token ids that decode to newlines only (LF/CR), per tokenizer (cached). The chat template
     * emits {@code </think>\n} before the answer; these are passed through untouched so the
     * boilerplate newline is not consumed by the grammar (no whitespace baked into the language).
     */
    private int[] newlineTokens(GgufTokenizer tok) {
        return newlineCache.computeIfAbsent(
                tok,
                t -> {
                    List<Integer> ids = new ArrayList<>();
                    for (int i = 0, n = t.vocabularySize(); i < n; i++) {
                        byte[] b = t.decodeTokenBytes(i);
                        if (b.length == 0) continue;
                        boolean nl = true;
                        for (byte x : b) {
                            int c = x & 0xFF;
                            if (c != '\n' && c != '\r') {
                                nl = false;
                                break;
                            }
                        }
                        if (nl) ids.add(i);
                    }
                    int[] arr = new int[ids.size()];
                    for (int i = 0; i < arr.length; i++) arr[i] = ids.get(i);
                    return arr;
                });
    }

    /**
     * User stop strings stay TEXT stops only: token stops end generation anywhere, including inside
     * the think span, while text stops are matched against content alone.
     */
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

    /**
     * Effective thinking switch for a server request: chat_template_kwargs.enable_thinking
     * (llama.cpp convention) overrides the CLI --think flag. Forced tool calls never think - the
     * call marker is seeded as the first assistant token.
     */
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

    /**
     * llama.cpp-compatible reasoning_format: "none" = leave thinking inline in content (with
     * literal <think> markers) instead of routing it to the reasoning_content channel - lets
     * vanilla OpenAI clients that only render content show thinking live.
     */
    private static boolean inlineReasoning(Map<String, Object> request) {
        return "none".equals(Values.stringValue(request.get("reasoning_format"), null));
    }

    // ---- prompt-cache warming ----------------------------------------------

    /**
     * Warm prompts are not implemented on this path yet (the cache itself is live; warming would
     * pre-ingest and commit the given prompts at startup). Warns so the flag is not silently lost.
     */
    void warm() {
        if (!options.warmPrompts().isEmpty()) {
            System.err.println("warm-prompt ignored: startup warming is not implemented yet");
        }
        if (System.getProperty("jinfer.promptCacheWarm") != null) {
            System.err.println(
                    "jinfer.promptCacheWarm ignored: the property was removed - the cache "
                            + "warms itself per request (start-up warming is not implemented)");
        }
    }
}
