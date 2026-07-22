package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.CacheStore;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.SessionPool;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Thinking;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.Generator.GenerationResult;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.function.Consumer;

/**
 * The inference service: turns a parsed request into a {@link Reply}, with streaming sinks for the
 * live channels. Owns the model and drives generation through the tokens-only {@link Generator}:
 * sampler / grammar / think wiring, stop conditions, tool-call seeding/parsing. The reply token
 * stream is structured by the model's {@link ReplyParser} (text channels + structure); transport -
 * endpoint handlers and SSE streaming - lives in {@link Server}.
 *
 * <p>Plain chat on a model with a {@link ChatTemplate} is lowered through the hand-written codec
 * (injection-inert, oracle-validated framing); when the model also has a {@link StateCodec} and
 * caching is enabled, the conversation is served through a per-model {@link PromptCache}, so a
 * follow-up request that echoes the prior turns resumes their KV instead of re-prefilling. Requests
 * with tools the template cannot frame, and models without a template, take the whole-render
 * fallback.
 */
final class Generation {

    private final LoadedModel<?> model;
    private final LLMOptions options;
    private final Worker worker;
    private final Map<Tokenizer, int[]> newlineCache =
            Collections.synchronizedMap(new WeakHashMap<>());
    private final ChatTemplate template; // memoized model framing, null when the model has none
    private final JinjaChatTemplate jinjaTemplate; // whole-render fallback, compiled once
    private final Set<Integer> stopTokens; // memoized model stops
    private final PromptCache<?>
            promptCache; // per-model cache; null without a StateCodec or when disabled
    private final SessionPool<?>
            sessionPool; // tier 1: last-N live conversations (append-only reuse)

    Generation(LoadedModel<?> chatModel, LLMOptions options, Worker worker) {
        this.model = chatModel;
        this.options = options;
        this.worker = worker;
        this.template = chatModel.template().orElse(null);
        this.jinjaTemplate = new JinjaChatTemplate(model.tokenizer(), model.chatTemplateSource());
        this.stopTokens = model.stopTokens();
        this.promptCache = RuntimeFlags.PROMPT_CACHE ? buildCache(model) : null;
        this.sessionPool = promptCache != null ? new SessionPool<>(RuntimeFlags.SESSIONS) : null;
    }

    private static <S extends RuntimeState> PromptCache<S> buildCache(LoadedModel<S> m) {
        StateCodec<S> codec = m.model().stateCodec().orElse(null);
        if (codec == null) return null;
        return new PromptCache<>(
                codec, CacheStore.inMemory(), RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES, m.seed());
    }

    // ---- chat / completion -------------------------------------------------

    @SuppressWarnings("unchecked")
    Reply chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        boolean tools = ToolUse.offered(request);
        // Codec path: the model's own framing, whenever the template supports the conversation
        // byte-exactly and the request does not FORCE a specific call (forcing seeds the prompt,
        // still a whole-render concern). Caching is a separate, optional layer on top (the
        // framing does not depend on it).
        if (template != null && ToolUse.forced(request) == null && onlyKnownKwargs(request)) {
            List<Message> turns = toConversation(messages);
            if (turns != null) {
                Conversation conversation =
                        new Conversation(
                                turns,
                                tools ? buildTools(request) : List.of(),
                                requestThink(request),
                                "");
                try {
                    Reply reply = chatTemplated(model, request, conversation, sinks);
                    // Bare-call recovery (llama.cpp #21242): LFM2.5 sometimes emits pythonic
                    // calls WITHOUT its markers; the structural parser found nothing, so run the
                    // string-scan fallback (no-op when the parser produced calls; names
                    // allow-listed).
                    return tools ? ToolUse.parse(model, reply, request) : reply;
                } catch (UnsupportedConversation fallback) {
                    // the port cannot frame this shape byte-exactly: whole-render below
                }
            }
        }
        Map<String, Object> kwargs =
                request.get("chat_template_kwargs") instanceof Map<?, ?> kw
                        ? (Map<String, Object>) kw
                        : null;
        IntSequence promptTokens =
                ToolUse.seedForced(
                        model.tokenizer(),
                        request,
                        jinjaTemplate.render(
                                messages,
                                tools ? Values.asArray(request.get("tools"), "tools") : null,
                                true,
                                requestThink(request),
                                kwargs));
        if (System.getProperty("jinfer.debugPrompt") != null) {
            System.err.println("[prompt] " + model.tokenizer().decode(promptTokens));
        }
        Reply reply = generate(request, promptTokens, sinks);
        return tools ? ToolUse.parse(model, reply, request) : reply;
    }

    /**
     * Maps OpenAI messages to {@link Message}s, or null when a message can't be represented on the
     * codec path (tool role / tool_calls / non-text content) - the caller falls back.
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
     * The codec chat path: lower the conversation to the model's own framing and, when the prompt
     * cache is available, resume the longest cached prefix into the state (skipping that prefill)
     * and ingest only the delta, caching it for the next turn. Assistant history is re-tokenized
     * from text - best-effort, but the framing is byte-exact with the model's Jinja template
     * (validated by the per-model oracle), so a stable client echo reuses the whole prefix.
     */
    private <S extends RuntimeState> Reply chatTemplated(
            LoadedModel<S> m, Map<String, Object> request, Conversation conversation, Sinks sinks) {
        // Batch-aligned blocks: the codec's batch boundaries are its turn-stable cache boundaries
        // (preamble, each turn, scaffold last) - so a follow-up request that diverges after turn
        // k still reuses turns 0..k-1 (blocks match completely, so one giant block would be
        // unusable the moment the conversation grows).
        List<List<Batch>> groups = new ArrayList<>();
        for (Batch b : template.encode(conversation)) groups.add(List.of(b));

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
            return generateFrom(m, state, request, sinks, 0);
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
                    Reply result = generateFrom(m, session.state(), request, sinks, restored);
                    // Bring the decode back into the session (the generator stepped the state
                    // directly):
                    // the reply extends the stream and commits as a block, and the live session
                    // returns
                    // to the pool ready for the next echo to continue append-only.
                    int ingested = session.state().position() - billed;
                    if (ingested > 0)
                        session.adopt(result.tokens().subSequence(0, ingested).toList());
                    return result;
                });
    }

    Reply completion(Map<String, Object> request, String prompt, Sinks sinks) {
        Tokenizer tokenizer = model.tokenizer();
        IntSequence promptTokens =
                options.rawPrompt() ? jinjaTemplate.encodeRaw(prompt) : tokenizer.encode(prompt);
        return generate(request, promptTokens, sinks);
    }

    /** One pass through the tokens-only {@link Generator}: a fresh state prefills the prompt. */
    private Reply generate(Map<String, Object> request, IntSequence promptTokens, Sinks sinks) {
        return runGeneration(model, null, request, promptTokens, sinks, 0);
    }

    /** Prompt size as billed to the client: a leading BOS is template overhead, not user input. */
    private static int consumedPromptTokens(Tokenizer tokenizer, IntSequence promptTokens) {
        int bos = SpecialTokens.findFirst(tokenizer, "<bos>", "<|startoftext|>").orElse(1);
        if (!promptTokens.isEmpty() && promptTokens.getFirst() == bos) {
            return promptTokens.length() - 1;
        }
        return promptTokens.length();
    }

    /**
     * Decode a request onto an already-resumed state (empty prompt, the state continues at its
     * cursor); {@code cachedTokens} is the restored prefix length billed to the client.
     */
    private <S extends RuntimeState> Reply generateFrom(
            LoadedModel<S> m, S state, Map<String, Object> request, Sinks sinks, int cachedTokens) {
        return runGeneration(m, state, request, IntSequence.empty(), sinks, cachedTokens);
    }

    /**
     * Request sampling/limit fields plus the decode-side plumbing, then one generation pass. The
     * model's {@link ReplyParser} structures the raw token stream into text / reasoning / tool-call
     * parts, routed to the streaming sinks live and coalesced into the {@link Reply}. Streaming
     * counters mirror the final usage: generated tokens are counted unless they are the trailing
     * stop token removed from the result.
     */
    private <S extends RuntimeState> Reply runGeneration(
            LoadedModel<S> m,
            S resumedState,
            Map<String, Object> request,
            IntSequence promptTokens,
            Sinks sinks,
            int cachedTokens) {
        Tokenizer tokenizer = m.tokenizer();
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
        List<String> textStops = textStops(request.get("stop"));
        boolean think = requestThink(request);
        Sampler sampler =
                Sampler.select(m.model().config().vocabularySize(), temperature, topp, seed);
        if (think) {
            // thinking models starve the answer under tight budgets: cap the think span, by default
            // at half the completion budget (request reasoning_max_tokens overrides; -1 = uncapped)
            int reasoningBudget =
                    Values.intValue(
                            request.get("reasoning_max_tokens"),
                            maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1);
            sampler = Thinking.capBudget(sampler, tokenizer, reasoningBudget);
        } else {
            sampler = Thinking.banMarkers(sampler, tokenizer);
        }
        Grammar.Cursor grammarCursor = buildGrammarCursor(tokenizer, request);
        if (grammarCursor != null) {
            int eosToken = SpecialTokens.findFirst(tokenizer, "<eos>", "<|endoftext|>").orElse(2);
            // For a reasoning request, hold the grammar until after the think span - constraining
            // from token 0 would suppress the model's reasoning and degrade the answer. The newline
            // the model emits between </think> and the answer is passed through, not consumed by
            // the
            // grammar (so non-ws-tolerant grammars like choice/jsonCompact see a clean first
            // token).
            int grammarGate = think ? SpecialTokens.find(tokenizer, "</think>").orElse(-1) : -1;
            int[] skipNl = grammarGate >= 0 ? newlineTokens(tokenizer) : null;
            sampler = Sampler.withGrammar(sampler, grammarCursor, eosToken, grammarGate, skipNl);
        }
        // Billed prompt: the whole conversation. On the cached path the state is pre-resumed to the
        // full prompt (position == total), of which cachedTokens were restored from the cache.
        int billedPrompt =
                resumedState != null
                        ? resumedState.position()
                        : consumedPromptTokens(tokenizer, promptTokens);
        if (usageCounts != null) usageCounts.promptTokens = billedPrompt;

        // Decode side: the model's parser structures the reply. Without tools a plain span
        // parser (no call claimer) keeps the behavior: markers drop as specials, payload text
        // stays visible.
        ReplyParser parser =
                ToolUse.offered(request) && template != null
                        ? template.parser()
                        : ReplyParser.spans(tokenizer);
        FragmentRouter router =
                new FragmentRouter(
                        textStops, sinks.onText(), sinks.onReasoning(), inlineReasoning(request));
        Generator.TokenSink sink =
                token -> {
                    if (usageCounts != null) {
                        usageCounts.cachedTokens = cachedTokens;
                        if (!stopTokens.contains(token)) usageCounts.completionTokens++;
                    }
                    String fragment = parser.feed(token);
                    if (!fragment.isEmpty()) router.fragment(fragment, parser.reasoning());
                    return !router.stopped();
                };
        S state =
                resumedState != null
                        ? resumedState
                        : m.model()
                                .newState(
                                        m.model().config().contextLength(),
                                        Math.max(promptTokens.length(), 16));
        GenerationResult result =
                Generator.generate(
                        m.model(),
                        state,
                        promptTokens,
                        sampler,
                        maxTokens,
                        ServerFlags.SERVER_REQUEST_TIMEOUT_NANOS,
                        stopTokens,
                        sink);
        Message structured = parser.finish();
        router.flush();
        Reply reply = router.reply(result, structured, billedPrompt, cachedTokens, textStops);
        Metrics.record(reply);
        return reply;
    }

    /**
     * Routes the parser's text fragments to the live sinks and accumulates the streamed reply:
     * content through the stop-string holdback, reasoning to its channel (or bracketed inline as
     * {@code <think>...</think>} content for reasoning_format "none"). Structure (tool calls) comes
     * from the parser's finished {@link Message}, not the stream - calls are atomic.
     */
    private static final class FragmentRouter {
        private final StringBuilder text = new StringBuilder();
        private final StringBuilder reasoning = new StringBuilder();
        private final TextStops.Holdback holdback; // null when neither streaming nor text stops
        private final java.util.function.Consumer<String> onReasoning;
        private final InlineThink inline; // null when reasoning routes to its own channel

        FragmentRouter(
                List<String> textStops,
                Consumer<String> onText,
                Consumer<String> onReasoning,
                boolean inline) {
            this.holdback =
                    onText != null || !textStops.isEmpty()
                            ? new TextStops.Holdback(textStops, onText != null ? onText : t -> {})
                            : null;
            this.onReasoning = onReasoning;
            this.inline = inline ? new InlineThink() : null;
        }

        void fragment(String fragment, boolean reasoningChannel) {
            if (reasoningChannel) {
                if (inline != null) {
                    content(inline.project(fragment, true));
                    return;
                }
                reasoning.append(fragment);
                if (onReasoning != null) onReasoning.accept(fragment);
            } else {
                content(inline != null ? inline.project(fragment, false) : fragment);
            }
        }

        private void content(String fragment) {
            text.append(fragment);
            if (holdback != null) holdback.accept(fragment);
        }

        boolean stopped() {
            return holdback != null && holdback.stopped();
        }

        void flush() {
            if (holdback != null) holdback.flush();
        }

        /** The coalesced {@link Reply}, with stop strings applied and finish_reason mapped. */
        Reply reply(
                GenerationResult result,
                Message structured,
                int promptTokens,
                int cachedTokens,
                List<String> textStops) {
            List<Part.ToolCall> toolCalls = collectCalls(structured.content());
            TextStops.Result stopResult = TextStops.apply(text.toString(), textStops);
            boolean textStopped = stopResult.stopped() || stopped();
            String finishReason =
                    !toolCalls.isEmpty()
                            ? "tool_calls"
                            : result.stopToken() >= 0 || textStopped
                                    ? "stop"
                                    : "length".equals(result.finishReason()) ? "length" : "stop";
            return new Reply(
                    result,
                    promptTokens,
                    cachedTokens,
                    stopResult.text(),
                    reasoning.isEmpty() ? null : reasoning.toString(),
                    toolCalls,
                    finishReason);
        }

        /** Every tool call in the reply, in order, including calls made inside a think span. */
        private static List<Part.ToolCall> collectCalls(List<Part> parts) {
            List<Part.ToolCall> calls = new ArrayList<>();
            for (Part part : parts) {
                if (part instanceof Part.ToolCall c) calls.add(c);
                else if (part instanceof Part.Reasoning r) calls.addAll(collectCalls(r.content()));
            }
            return calls;
        }
    }

    // ---- sampler / grammar / stop / think wiring ---------------------------

    /**
     * Builds a grammar cursor from request params: {@code grammar} (GBNF string) or {@code
     * response_format: {type: "json_object"}}. Returns null when no constraint.
     */
    private static Grammar.Cursor buildGrammarCursor(
            Tokenizer tokenizer, Map<String, Object> request) {
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
    private int[] newlineTokens(Tokenizer tok) {
        return newlineCache.computeIfAbsent(
                tok,
                t -> {
                    List<Integer> ids = new ArrayList<>();
                    for (int i = 0, n = t.vocabulary().size(); i < n; i++) {
                        byte[] b = t.decodeBytes(new int[] {i});
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
    private static List<String> textStops(Object value) {
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
        return List.copyOf(textStops);
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
