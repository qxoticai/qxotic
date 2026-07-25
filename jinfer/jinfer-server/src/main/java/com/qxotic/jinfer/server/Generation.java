package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.CacheStore;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.SessionPool;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JinjaChatTemplate;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.chat.Values;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.Generator.GenerationResult;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
    private final ChatTemplate template; // memoized model framing, null when the model has none
    private final JinjaChatTemplate jinjaTemplate; // whole-render fallback, compiled once
    private final Set<Integer> stopTokens; // memoized model stops
    private final PromptCache<?>
            promptCache; // per-model cache; null without a StateCodec or when disabled
    private final SessionPool<?>
            sessionPool; // tier 1: last-N live conversations (append-only reuse)

    Generation(LoadedModel<?> chatModel, LLMOptions options) {
        this.model = chatModel;
        this.options = options;
        this.template = chatModel.template().orElse(null);
        this.jinjaTemplate = new JinjaChatTemplate(model.tokenizer(), model.chatTemplateSource());
        this.stopTokens = model.stopTokens();
        this.promptCache = RuntimeFlags.PROMPT_CACHE ? buildCache(model) : null;
        this.sessionPool = promptCache != null ? new SessionPool<>(RuntimeFlags.SESSIONS) : null;
        if (!options.warmPrompts().isEmpty()) {
            System.err.println("warm-prompt ignored: startup warming is not implemented yet");
        }
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
        // byte-exactly. Forced calls ride it too when the template declares its call seed (the
        // seed batch + prefix-pin + epilogue recipe, same as the langchain4j provider); only
        // templates without the hook, or a named choice the request does not offer, still take
        // the whole-render legacy path. Caching is a separate, optional layer on top.
        if (template != null && nativeForcedOk(request) && onlyKnownKwargs(request)) {
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

    /** Forced requests stay native only when the template's forced-call recipe covers them. */
    private boolean nativeForcedOk(Map<String, Object> request) {
        String forced = ToolUse.forced(request);
        if (forced == null) return true;
        if (template.callSeed().length == 0) return false;
        return forced.isEmpty() || ToolUse.names(request).contains(forced);
    }

    /** The pin's tool list: the single named function, or everything offered for "required". */
    private List<Tool> pinTools(Map<String, Object> request) {
        List<Tool> offered = buildTools(request);
        String forced = ToolUse.forced(request);
        if (forced == null || forced.isEmpty()) return offered;
        return offered.stream().filter(t -> t.name().equals(forced)).toList();
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
        if (ToolUse.forced(request) != null) {
            // the forcing trick: the family's call marker joins the prompt, so the model can
            // only COMPLETE a call (the parser is pre-fed the same seed in runGeneration)
            groups.add(List.of(Batch.prefill(template.callSeed())));
        }

        int total = 0;
        for (List<Batch> group : groups) for (Batch b : group) total += b.count();
        @SuppressWarnings("unchecked")
        PromptCache<S> cache = (PromptCache<S>) promptCache;
        if (cache == null) {
            S state = m.model().newState(m.model().config().contextLength());
            List<Batch> all = new ArrayList<>(); // uncached: plain ingest, framing unchanged
            for (List<Batch> group : groups) all.addAll(group);
            for (Batch b : Batch.prepare(all, state.batchCapacity())) m.model().ingest(state, b);
            return generateFrom(m, state, request, sinks, 0);
        }
        // Tier 1: a live pooled session whose whole stream strictly prefixes this conversation
        // continues append-only (no restore at all). Otherwise tier 2: resume the longest block
        // prefix into a fresh state - the pool derives both from the groups (content addressing
        // is the cache package's own law; the server never sees it).
        @SuppressWarnings("unchecked")
        SessionPool<S> pool = (SessionPool<S>) sessionPool;
        int billed = total;
        return pool.withSession(
                m.model(),
                cache,
                () -> m.model().newState(m.model().config().contextLength()),
                groups,
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
                    // decode from the retained logits (empty prompt continues at the cursor);
                    // the whole prompt was billed, of which `restored` came from the cache
                    Reply result = generateFrom(m, session.state(), request, sinks, restored);
                    // Bring the decode back into the session (the generator stepped the state
                    // directly): the reply extends the stream and commits as a block, and the
                    // live session returns to the pool ready for the next echo to continue
                    // append-only.
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
        // the worker past jinfer.serverMaxTokens; hitting it reports finish_reason "length"
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
        // Forced call on the CODEC path: only chatTemplated seeds callSeed into the prompt, and
        // it always arrives here with a resumed state - the whole-render fallback (legacy
        // seedForced marker) must NOT get the pin or the parser pre-feed. requestThink already
        // turns thinking off for any forced request.
        boolean forced = resumedState != null && ToolUse.forced(request) != null;
        boolean think = requestThink(request);
        // the shared sampling stack (ChatEngine): base sampling + the reasoning policy;
        // reasoning_max_tokens is the server's per-request override of the half-budget default
        Object rmt = request.get("reasoning_max_tokens");
        Sampler sampler =
                ChatEngine.sampler(
                        model,
                        temperature,
                        topp,
                        seed,
                        think,
                        maxTokens,
                        rmt == null ? null : Values.intValue(rmt, -1));
        Grammar.Cursor grammarCursor = buildGrammarCursor(tokenizer, request);
        if (grammarCursor != null) {
            // shared wiring (ChatEngine.constrained): think-gated, newline-skipped, dead-ending
            // on one of the model's OWN stops (previously a vocab scan for <eos>, which is not
            // guaranteed to be a stop token - the pin path always used stops.first)
            sampler = ChatEngine.constrained(model, sampler, grammarCursor, think);
        }
        // the shared forced-call recipe: prefix-pin the offered (or THE named) tool + the
        // family epilogue, and the parser pre-feed below starts in the seeded span state
        // (chatTemplated already put the call seed in the prompt on this path)
        ChatEngine.ForcedCall forcedCall =
                forced
                        ? ChatEngine.forceCall(model, pinTools(request), sampler).orElseThrow()
                        : null;
        if (forcedCall != null) sampler = forcedCall.sampler();
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
        // pre-feed the prompt's reply-grammar tail - the parser starts in the exact span
        // state the prompt left the model in (without this, a prompt-opened think span routes
        // reasoning into the CONTENT channel); a forced call uses the recipe's own pre-feed
        if (forcedCall != null) {
            for (int t : forcedCall.parserSeed()) parser.feed(t);
        } else if (template != null) {
            for (int t : template.replySeed(think)) parser.feed(t);
        }
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
        private final Consumer<String> onReasoning;
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
}
