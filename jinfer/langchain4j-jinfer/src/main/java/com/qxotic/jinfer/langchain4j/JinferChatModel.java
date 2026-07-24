package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Thinking;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.internal.JsonSchemaElementUtils;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

/**
 * langchain4j {@link ChatModel} backed by jinfer: in-process CPU inference over a local GGUF.
 * Prompting goes native-first through the model's hand-written, oracle-validated chat-template
 * codec (token-exact, injection-inert) and falls back to a scrubbed Jinja whole-render for unported
 * models or unframeable requests.
 *
 * <p>Run with jinfer's JVM flags: {@code --enable-preview --add-modules jdk.incubator.vector
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferChatModel implements ChatModel {

    final JinferEngine engine;
    final ChatRequestParameters defaults;
    final boolean thinking;
    final long seed;
    final long timeoutNanos;
    final List<ChatModelListener> listeners;
    // cached-prompt view state: EMPTY for the base model. Converted to jinfer types ONCE at view
    // creation (media decoded once, not per request); a view's conversations all start with this
    // prefix, its KV restored from the engine's block tree instead of re-prefilled.
    final CachedPrompt prefix;

    /** A view's prefix in jinfer types; {@link #EMPTY} for the base model. */
    record CachedPrompt(List<Message> messages, List<Tool> tools) {
        static final CachedPrompt EMPTY = new CachedPrompt(List.of(), List.of());

        boolean isEmpty() {
            return messages.isEmpty() && tools.isEmpty();
        }
    }

    private JinferChatModel(Builder b) {
        this.engine =
                new JinferEngine(b.modelPath, b.mediaProjector, b.contextLength, b.cachedPrompts);
        this.thinking = b.thinking;
        this.seed = b.seed;
        this.timeoutNanos = b.timeout == null ? 0 : b.timeout.toNanos();
        this.listeners = List.copyOf(b.listeners);
        this.prefix = CachedPrompt.EMPTY;
        ChatRequestParameters base =
                DefaultChatRequestParameters.builder()
                        .modelName(engine.modelName)
                        .temperature(b.temperature)
                        .topP(b.topP)
                        .maxOutputTokens(b.maxOutputTokens)
                        .build();
        // caller-supplied defaults win field-by-field; unsupported ones reject HERE, eagerly - a
        // model whose defaults can never serve a request should fail at build, not at first use
        this.defaults = b.defaultParameters == null ? base : base.overrideWith(b.defaultParameters);
        if (b.defaultParameters != null) {
            rejectUnsupported(this.defaults);
            rejectModelSwitch(engine, this.defaults);
        }
    }

    private JinferChatModel(JinferChatModel base, CachedPrompt prefix) {
        this.engine = base.engine;
        this.defaults = base.defaults;
        this.thinking = base.thinking;
        this.seed = base.seed;
        this.timeoutNanos = base.timeoutNanos;
        this.listeners = base.listeners;
        this.prefix = prefix;
    }

    /**
     * A model view whose conversations all start with {@code prefix} (+ welded {@code tools}) -
     * prefilled ONCE into the engine's block tree, restored (not recomputed) on every chat.
     * Composable: calling this on a view branches on its prefix. Immutable, shares the base engine;
     * the base model itself never touches the tree.
     */
    public JinferChatModel withCachedPrompt(
            List<ChatMessage> prefixMessages, List<ToolSpecification> tools) {
        List<Message> messages = new ArrayList<>(prefix.messages());
        messages.addAll(Mappings.toMessages(prefixMessages)); // converted ONCE, media decoded here
        List<Tool> welded = new ArrayList<>(prefix.tools());
        welded.addAll(Mappings.toTools(tools));
        CachedPrompt merged = new CachedPrompt(List.copyOf(messages), List.copyOf(welded));
        engine.define(new Conversation(merged.messages(), merged.tools(), thinking, ""));
        return new JinferChatModel(this, merged);
    }

    /** Freezes every prompt defined so far (plus any mounted base) into one artifact. */
    public void saveCachedPrompts(Path out) {
        engine.freezePrompts(out);
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return defaults;
    }

    @Override
    public List<ChatModelListener> listeners() {
        return listeners; // core's chat() dispatches onRequest/onResponse/onError
    }

    @Override
    public java.util.Set<dev.langchain4j.model.chat.Capability> supportedCapabilities() {
        // grammar-constrained decoding honors JSON schemas natively - AiServices reads this to
        // use structured output for POJO extraction instead of prompt-based JSON begging
        return java.util.Set.of(dev.langchain4j.model.chat.Capability.RESPONSE_FORMAT_JSON_SCHEMA);
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        Prepared p = prepare(this, request);
        StopSequences stops = StopSequences.of(p.stops());
        ReplyLanes lanes =
                new ReplyLanes(p.encoded().template(), engine.loaded.tokenizer(), p.callSeed());
        Generator.TokenSink sink =
                token -> {
                    String fragment = lanes.feed(token);
                    if (stops != null && !lanes.reasoning() && !fragment.isEmpty()) {
                        stops.feed(fragment); // stop strings match the content lane only
                    }
                    return stops == null || !stops.hit();
                };
        Generator.GenerationResult result =
                engine.generate(
                        p.encoded().prompt(),
                        p.sampler(),
                        p.maxTokens(),
                        timeoutNanos,
                        sink,
                        p.cached());
        AiMessage ai = Mappings.toAiMessage(lanes.finish());
        boolean stopHit = stops != null && stops.hit();
        if (stopHit) {
            ai = Mappings.withText(ai, stops.beforeCut());
        }
        return Mappings.response(engine.modelName, ai, p.promptTokens(), result, stopHit);
    }

    JinferEngine engine() {
        return engine;
    }

    /** A streaming twin sharing this model's engine and cached prefix (the GGUF loads once). */
    public JinferStreamingChatModel streaming() {
        return new JinferStreamingChatModel(this);
    }

    // ---- shared request preparation (also used by the streaming twin) ----

    record Prepared(
            JinferEngine.Encoded encoded,
            Sampler sampler,
            int maxTokens,
            int promptTokens,
            boolean cached,
            int[] callSeed,
            List<String> stops) {}

    private static final int[] NO_SEED = new int[0];

    /** Every request-shape rejection, synchronously (streaming calls this before its thread). */
    static void validate(JinferChatModel m, ChatRequest request) {
        ChatRequestParameters p = request.parameters();
        rejectUnsupported(p);
        rejectModelSwitch(m.engine, p);
        boolean requestHasTools =
                p.toolSpecifications() != null && !p.toolSpecifications().isEmpty();
        if (!m.prefix.isEmpty() && requestHasTools) {
            throw new UnsupportedFeatureException(
                    "a cached-prompt view welds its tools into the cached prefix; per-request"
                            + " toolSpecifications would silently forfeit the cache - put tools on"
                            + " withCachedPrompt(...) instead");
        }
        if (p.toolChoice() == ToolChoice.REQUIRED) {
            if (!requestHasTools && m.prefix.tools().isEmpty()) {
                throw new IllegalArgumentException("toolChoice REQUIRED without any tools");
            }
            if (m.engine.loaded.template().isEmpty() || callMarker(m.engine).isEmpty()) {
                throw new UnsupportedFeatureException(
                        "ToolChoice.REQUIRED is not supported by this model: forcing seeds the"
                                + " reply with its tool-call marker, which needs the native codec");
            }
        }
        if (p.toolChoice() == ToolChoice.NONE && !m.prefix.tools().isEmpty()) {
            throw new UnsupportedFeatureException(
                    "toolChoice NONE on a cached-prompt view is not supported: the view's tools"
                            + " are welded into its cached prefix and cannot be un-offered");
        }
    }

    static Prepared prepare(JinferChatModel m, ChatRequest request) {
        validate(m, request);
        ChatRequestParameters p = request.parameters();
        boolean cached = !m.prefix.isEmpty();
        // NONE = the model cannot use tools: never offer them, and there is nothing to call
        List<ToolSpecification> requestTools =
                p.toolChoice() == ToolChoice.NONE || p.toolSpecifications() == null
                        ? List.of()
                        : p.toolSpecifications();
        JinferEngine engine = m.engine;
        int maxTokens = p.maxOutputTokens() == null ? -1 : p.maxOutputTokens();
        boolean required = p.toolChoice() == ToolChoice.REQUIRED;
        // a think span cannot fit a tiny completion budget: below the floor, reasoning is
        // disabled outright (scaffold and sampler both) so the budget buys VISIBLE text.
        // A forced call also skips thinking: the reply is seeded INTO the call block.
        boolean think = m.thinking && (maxTokens < 0 || maxTokens >= 16) && !required;
        List<Message> messages = new ArrayList<>(m.prefix.messages());
        messages.addAll(Mappings.toMessages(request.messages()));
        Conversation conversation =
                new Conversation(
                        messages,
                        cached ? m.prefix.tools() : Mappings.toTools(requestTools),
                        think,
                        "");
        // cached views are native-only (define enforced it); the base keeps the Jinja fallback
        JinferEngine.Encoded encoded =
                cached
                        ? engine.encodeNative(conversation)
                        : engine.encode(conversation, request.messages(), requestTools);
        Sampler sampler = sampler(m, p, think, maxTokens);
        int[] callSeed = NO_SEED;
        if (required) {
            // the server's forcing trick: seed the assistant turn with the tool-call marker so
            // the model can only COMPLETE a call (the paren is deliberately not seeded - it lands
            // on a tokenization boundary the model never saw). ReplyLanes re-feeds the seed so
            // the reply parses whole.
            callSeed = new int[] {callMarker(engine).orElseThrow()}; // validate() guaranteed it
            List<Batch> prompt = new ArrayList<>(encoded.prompt());
            prompt.add(Batch.prefill(callSeed));
            encoded = new JinferEngine.Encoded(List.copyOf(prompt), encoded.template());
            // prefix-pin: the family's call grammar pins "prefix (name|...)" right after the
            // seeded marker, then releases - the called tool is GUARANTEED to be an offered one,
            // the arguments stay the model's own. No grammar = seeding-only.
            var pin = encoded.template().flatMap(t -> t.callGrammar(conversation.tools()));
            if (pin.isPresent()) {
                sampler =
                        Sampler.withPrefixGrammar(
                                sampler,
                                Grammar.of(pin.get(), engine.loaded.tokenizer()).cursor(),
                                eos(engine));
            }
        }
        int promptTokens = encoded.prompt().stream().mapToInt(Batch::count).sum();
        return new Prepared(
                encoded, sampler, maxTokens, promptTokens, cached, callSeed, p.stopSequences());
    }

    /**
     * The request's sampling stack, inside-out: the standard (temperature, topP, seed) stack, the
     * server's reasoning policy (cap the think span at half the budget, or ban the markers outright
     * when thinking is off), and grammar-constrained JSON when the request asks for it - dormant
     * through the think span (constraining from token 0 would suppress reasoning), schema compiled
     * when present, the dead-end token one of the model's real stops. Specs are cached per (grammar
     * source, vocab), so repeated schemas reuse the compiled masks.
     */
    private static Sampler sampler(
            JinferChatModel m, ChatRequestParameters p, boolean think, int maxTokens) {
        JinferEngine engine = m.engine;
        var tokenizer = engine.loaded.tokenizer();
        double temperature = p.temperature() == null ? 0.0 : p.temperature();
        double topP = p.topP() == null ? 0.95 : p.topP();
        Sampler sampler =
                Sampler.select(
                        engine.loaded.model().config().vocabularySize(),
                        (float) temperature,
                        (float) topP,
                        m.seed);
        sampler =
                think
                        ? Thinking.capBudget(
                                sampler,
                                tokenizer,
                                maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1)
                        : Thinking.banMarkers(sampler, tokenizer);
        ResponseFormat rf = p.responseFormat();
        if (rf != null && rf.type() == ResponseFormatType.JSON) {
            Grammar.Spec spec =
                    rf.jsonSchema() == null
                            ? Grammar.json(tokenizer)
                            : Grammar.fromSchema(
                                    JsonSchemaElementUtils.toMap(rf.jsonSchema().rootElement()),
                                    tokenizer);
            int gate = think ? SpecialTokens.find(tokenizer, "</think>").orElse(-1) : -1;
            int[] skipNl = gate >= 0 ? SpecialTokens.newlineTokens(tokenizer) : null;
            sampler = Sampler.withGrammar(sampler, spec.cursor(), eos(engine), gate, skipNl);
        }
        return sampler;
    }

    /** The model family's tool-call opening marker (LFM2 / Gemma 4 / Qwen spellings). */
    private static OptionalInt callMarker(JinferEngine engine) {
        return SpecialTokens.findFirst(
                engine.loaded.tokenizer(), "<|tool_call_start|>", "<|tool_call>", "<tool_call>");
    }

    /** A stop token to end generation with when a grammar dead-ends. */
    private static int eos(JinferEngine engine) {
        return engine.loaded.stopTokens().iterator().next();
    }

    /** One loaded GGUF per instance: a different {@code modelName} cannot be served. */
    private static void rejectModelSwitch(JinferEngine engine, ChatRequestParameters p) {
        if (p.modelName() != null && !p.modelName().equals(engine.modelName)) {
            throw new UnsupportedFeatureException(
                    "per-request modelName is not supported: this model IS '"
                            + engine.modelName
                            + "' (one loaded GGUF per instance)");
        }
    }

    private static void rejectUnsupported(ChatRequestParameters p) {
        if (p.topK() != null) throw new UnsupportedFeatureException("topK is not supported");
        if (p.frequencyPenalty() != null)
            throw new UnsupportedFeatureException("frequencyPenalty is not supported");
        if (p.presencePenalty() != null)
            throw new UnsupportedFeatureException("presencePenalty is not supported");
        ResponseFormat rf = p.responseFormat();
        boolean tools = p.toolSpecifications() != null && !p.toolSpecifications().isEmpty();
        if (rf != null && rf.type() == ResponseFormatType.JSON && tools)
            throw new UnsupportedFeatureException(
                    "tools together with a JSON response format are not supported:"
                            + " grammar-constrained output cannot admit tool-call syntax");
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path modelPath;
        private Path mediaProjector;
        private Path cachedPrompts;
        private int contextLength;
        private Double temperature;
        private Double topP;
        private Integer maxOutputTokens;
        private ChatRequestParameters defaultParameters;
        private List<ChatModelListener> listeners = List.of();
        private boolean thinking = true;
        private long seed = 42;
        private Duration timeout;

        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        /** Mounts a cached-prompt artifact ({@link #saveCachedPrompts}); model-seed-checked. */
        public Builder loadCachedPrompts(Path cachedPrompts) {
            this.cachedPrompts = cachedPrompts;
            return this;
        }

        /** The media sidecar (mmproj GGUF: vision/audio encoders) for multimodal models. */
        public Builder mediaProjector(Path mediaProjector) {
            this.mediaProjector = mediaProjector;
            return this;
        }

        /** Context window; 0 = the model's own maximum. */
        public Builder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topP(Double topP) {
            this.topP = topP;
            return this;
        }

        public Builder maxOutputTokens(Integer maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        public Builder listeners(List<ChatModelListener> listeners) {
            this.listeners = listeners;
            return this;
        }

        /**
         * Default request parameters, merged under each request's own (standard langchain4j
         * semantics). Unsupported parameters are rejected eagerly at build.
         */
        public Builder defaultRequestParameters(ChatRequestParameters defaultParameters) {
            this.defaultParameters = defaultParameters;
            return this;
        }

        /** The model's reasoning scaffold toggle (templates without one ignore it). Default on. */
        public Builder thinking(boolean thinking) {
            this.thinking = thinking;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public JinferChatModel build() {
            if (modelPath == null) throw new IllegalArgumentException("modelPath is required");
            return new JinferChatModel(this);
        }
    }
}
