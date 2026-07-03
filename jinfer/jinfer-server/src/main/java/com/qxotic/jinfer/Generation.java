package com.qxotic.jinfer;

import com.qxotic.jinfer.Generator.GenerationResult;
import com.qxotic.jinfer.Generator.StopSpec;

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
 * <p>The prompt cache and in-place chat-session resume of the legacy engine are not (yet) ported to
 * this path: every request re-encodes its full prompt and generates from a fresh state.
 */
final class Generation {

    private final LanguageModel<?, ?, ?> model;
    private final LLMOptions options;
    private final Worker worker;
    private final Map<GgufTokenizer, int[]> newlineCache = Collections.synchronizedMap(new WeakHashMap<>());

    Generation(LanguageModel<?, ?, ?> model, LLMOptions options, Worker worker) {
        this.model = model;
        this.options = options;
        this.worker = worker;
    }

    // ---- chat / completion -------------------------------------------------

    @SuppressWarnings("unchecked")
    GenerationResult chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        GgufTokenizer tokenizer = model.tokenizer();
        ChatContext chatContext = new ChatContext(
                messages,
                ToolUse.offered(request) ? Values.asArray(request.get("tools"), "tools") : null,
                request.get("tool_choice"),
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
        int consumedPromptTokens = Generator.consumedPromptTokens(tokenizer, promptTokens); // client-facing usage counts
        IntConsumer onToken = usageCounts == null ? null : token -> {
            usageCounts.cachedTokens = 0; // no prompt cache on this path
            if (!stops.tokenStops().contains(token)) usageCounts.completionTokens++;
        };
        if (usageCounts != null) usageCounts.promptTokens = consumedPromptTokens;
        Generator.Params params = new Generator.Params(sampler, maxTokens, RuntimeFlags.SERVER_REQUEST_TIMEOUT_NANOS, stops, inlineReasoning(request));
        Generator.Listener listener = new Generator.Listener(onToken, sinks.onText(), sinks.onReasoning(), sinks.onToolCall());
        GenerationResult result = runGen(model, promptTokens, params, listener);
        Metrics.record(result);
        return result;
    }

    /** One new-API generation pass on a fresh state (captures the model's state type S). */
    private <S extends RuntimeState> GenerationResult runGen(LanguageModel<?, ?, S> m, List<Integer> promptTokens,
                                                             Generator.Params params, Generator.Listener listener) {
        S state = m.newState(m.config().contextLength(), Math.max(promptTokens.size(), 16));
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
        if (RuntimeFlags.PROMPT_CACHE_WARM != null) {
            for (String f : RuntimeFlags.PROMPT_CACHE_WARM.split(",")) {
                if (!f.isBlank()) files.add(f.strip());
            }
        }
        if (!files.isEmpty()) {
            System.err.println("warm-prompt ignored: prompt cache is not available on the new-API path");
        }
    }
}
