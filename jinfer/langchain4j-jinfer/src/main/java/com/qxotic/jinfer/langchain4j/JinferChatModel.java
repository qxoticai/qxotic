package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Thinking;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

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

    private final JinferEngine engine;
    private final ChatRequestParameters defaults;
    private final boolean thinking;
    private final long seed;
    private final long timeoutNanos;
    // cached-prompt view state: empty for the base model. A view's conversations all start with
    // this prefix (KV restored from the engine's block tree, never re-prefilled).
    private final List<ChatMessage> prefixMessages;
    private final List<ToolSpecification> prefixTools;

    private JinferChatModel(Builder b) {
        this.engine =
                new JinferEngine(b.modelPath, b.mediaProjector, b.contextLength, b.cachedPrompts);
        this.thinking = b.thinking;
        this.seed = b.seed;
        this.timeoutNanos = b.timeout == null ? 0 : b.timeout.toNanos();
        this.prefixMessages = List.of();
        this.prefixTools = List.of();
        this.defaults =
                DefaultChatRequestParameters.builder()
                        .modelName(engine.modelName)
                        .temperature(b.temperature)
                        .topP(b.topP)
                        .maxOutputTokens(b.maxOutputTokens)
                        .build();
    }

    private JinferChatModel(
            JinferChatModel base,
            List<ChatMessage> prefixMessages,
            List<ToolSpecification> prefixTools) {
        this.engine = base.engine;
        this.defaults = base.defaults;
        this.thinking = base.thinking;
        this.seed = base.seed;
        this.timeoutNanos = base.timeoutNanos;
        this.prefixMessages = List.copyOf(prefixMessages);
        this.prefixTools = List.copyOf(prefixTools);
    }

    /**
     * A model view whose conversations all start with {@code prefix} (+ welded {@code tools}) -
     * prefilled ONCE into the engine's block tree, restored (not recomputed) on every chat.
     * Composable: calling this on a view branches on its prefix. Immutable, shares the base engine;
     * the base model itself never touches the tree.
     */
    public JinferChatModel withCachedPrompt(
            List<ChatMessage> prefix, List<ToolSpecification> tools) {
        List<ChatMessage> mergedMessages = new java.util.ArrayList<>(prefixMessages);
        mergedMessages.addAll(prefix);
        List<ToolSpecification> mergedTools = new java.util.ArrayList<>(prefixTools);
        mergedTools.addAll(tools);
        engine.define(
                new Conversation(
                        Mappings.toMessages(mergedMessages),
                        Mappings.toTools(mergedTools),
                        thinking,
                        ""));
        return new JinferChatModel(this, mergedMessages, mergedTools);
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
    public ChatResponse doChat(ChatRequest request) {
        Prepared p = prepare(engine, request, thinking, seed, prefixMessages, prefixTools);
        Generator.GenerationResult result =
                p.cached()
                        ? engine.cachedGenerate(
                                p.encoded().prompt(),
                                p.sampler(),
                                p.maxTokens(),
                                timeoutNanos,
                                t -> true)
                        : engine.generate(
                                p.encoded().prompt(),
                                p.sampler(),
                                p.maxTokens(),
                                timeoutNanos,
                                t -> true);
        AiMessage ai = Mappings.toAiMessage(engine.decode(p.encoded().template(), result.tokens()));
        return Mappings.response(engine.modelName, ai, p.promptTokens(), result);
    }

    JinferEngine engine() {
        return engine;
    }

    /** A streaming twin sharing this model's engine (the GGUF is loaded once). */
    public JinferStreamingChatModel streaming() {
        return JinferStreamingChatModel.over(
                engine, defaults, thinking, seed, timeoutNanos, prefixMessages, prefixTools);
    }

    // ---- shared request preparation (also used by the streaming twin) ----

    record Prepared(
            JinferEngine.Encoded encoded,
            Sampler sampler,
            int maxTokens,
            int promptTokens,
            boolean cached) {}

    static Prepared prepare(
            JinferEngine engine,
            ChatRequest request,
            boolean thinking,
            long seed,
            List<ChatMessage> prefixMessages,
            List<ToolSpecification> prefixTools) {
        ChatRequestParameters p = request.parameters();
        rejectUnsupported(p);
        boolean cached = !prefixMessages.isEmpty() || !prefixTools.isEmpty();
        List<ToolSpecification> tools =
                p.toolSpecifications() == null ? List.of() : p.toolSpecifications();
        if (cached && !tools.isEmpty()) {
            throw new UnsupportedFeatureException(
                    "a cached-prompt view welds its tools into the cached prefix; per-request"
                            + " toolSpecifications would silently forfeit the cache - put tools on"
                            + " withCachedPrompt(...) instead");
        }
        JinferEngine.Encoded encoded;
        Conversation conversation;
        if (cached) {
            List<ChatMessage> all = new java.util.ArrayList<>(prefixMessages);
            all.addAll(request.messages());
            conversation =
                    new Conversation(
                            Mappings.toMessages(all), Mappings.toTools(prefixTools), thinking, "");
            tools = prefixTools;
            // native-only: the view was created through the native codec (define enforced it)
            encoded =
                    new JinferEngine.Encoded(
                            engine.encodeNative(conversation), engine.loaded.template());
        } else {
            conversation =
                    new Conversation(
                            Mappings.toMessages(request.messages()),
                            Mappings.toTools(tools),
                            thinking,
                            "");
            encoded = engine.encode(conversation, request.messages(), tools);
        }
        double temperature = p.temperature() == null ? 0.0 : p.temperature();
        double topP = p.topP() == null ? 0.95 : p.topP();
        int maxTokens = p.maxOutputTokens() == null ? -1 : p.maxOutputTokens();
        Sampler sampler =
                Sampler.select(
                        engine.loaded.model().config().vocabularySize(),
                        (float) temperature,
                        (float) topP,
                        seed);
        // mirror the server's reasoning policy: cap the think span at half the budget, or ban the
        // markers outright when thinking is off (a thinking model would otherwise still emit them)
        sampler =
                thinking
                        ? Thinking.capBudget(
                                sampler,
                                engine.loaded.tokenizer(),
                                maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1)
                        : Thinking.banMarkers(sampler, engine.loaded.tokenizer());
        int promptTokens = encoded.prompt().stream().mapToInt(Batch::count).sum();
        return new Prepared(encoded, sampler, maxTokens, promptTokens, cached);
    }

    private static void rejectUnsupported(ChatRequestParameters p) {
        if (p.topK() != null) throw new UnsupportedFeatureException("topK is not supported");
        if (p.frequencyPenalty() != null)
            throw new UnsupportedFeatureException("frequencyPenalty is not supported");
        if (p.presencePenalty() != null)
            throw new UnsupportedFeatureException("presencePenalty is not supported");
        if (p.stopSequences() != null && !p.stopSequences().isEmpty())
            throw new UnsupportedFeatureException("stopSequences are not supported yet");
        if (p.toolChoice() == ToolChoice.REQUIRED)
            throw new UnsupportedFeatureException("toolChoice REQUIRED is not supported");
        ResponseFormat rf = p.responseFormat();
        if (rf != null && rf.type() == ResponseFormatType.JSON)
            throw new UnsupportedFeatureException("JSON response format is not supported yet");
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
        private boolean thinking = true;
        private long seed = 42;
        private Duration timeout;

        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        /** The media sidecar (mmproj GGUF: vision/audio encoders) for multimodal models. */
        /** Mounts a cached-prompt artifact ({@link #saveCachedPrompts}); model-seed-checked. */
        public Builder loadCachedPrompts(Path cachedPrompts) {
            this.cachedPrompts = cachedPrompts;
            return this;
        }

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
