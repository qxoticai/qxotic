package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Thinking;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.TokenUsage;
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

    private JinferChatModel(Builder b) {
        this.engine = b.engine != null ? b.engine : new JinferEngine(b.modelPath, b.contextLength);
        this.thinking = b.thinking;
        this.seed = b.seed;
        this.timeoutNanos = b.timeout == null ? 0 : b.timeout.toNanos();
        this.defaults =
                DefaultChatRequestParameters.builder()
                        .modelName(engine.modelName)
                        .temperature(b.temperature)
                        .topP(b.topP)
                        .maxOutputTokens(b.maxOutputTokens)
                        .build();
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return defaults;
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        Prepared p = prepare(engine, request, thinking, seed);
        Generator.GenerationResult result =
                engine.generate(
                        p.encoded().prompt(), p.sampler(), p.maxTokens(), timeoutNanos, t -> true);
        com.qxotic.jinfer.chat.Message reply =
                engine.decode(p.encoded().template(), result.tokens());
        AiMessage ai = Mappings.toAiMessage(reply);
        return ChatResponse.builder()
                .aiMessage(ai)
                .modelName(engine.modelName)
                .tokenUsage(new TokenUsage(p.promptTokens(), result.completionTokens()))
                .finishReason(
                        Mappings.toFinishReason(
                                result.finishReason(), ai.hasToolExecutionRequests()))
                .build();
    }

    /** A streaming twin sharing this model's engine (the GGUF is loaded once). */
    public JinferStreamingChatModel streaming() {
        return JinferStreamingChatModel.over(engine, defaults, thinking, seed, timeoutNanos);
    }

    // ---- shared request preparation (also used by the streaming twin) ----

    record Prepared(
            JinferEngine.Encoded encoded, Sampler sampler, int maxTokens, int promptTokens) {}

    static Prepared prepare(JinferEngine engine, ChatRequest request, boolean thinking, long seed) {
        ChatRequestParameters p = request.parameters();
        rejectUnsupported(p);
        var tools =
                p.toolSpecifications() == null
                        ? List.<dev.langchain4j.agent.tool.ToolSpecification>of()
                        : p.toolSpecifications();
        Conversation conversation =
                new Conversation(
                        Mappings.toMessages(request.messages()),
                        Mappings.toTools(tools),
                        thinking,
                        "");
        JinferEngine.Encoded encoded =
                engine.encode(
                        conversation,
                        Mappings.toMessageMaps(request.messages()),
                        Mappings.toToolMaps(tools));
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
        int promptTokens = encoded.prompt().stream().mapToInt(com.qxotic.jinfer.Batch::count).sum();
        return new Prepared(encoded, sampler, maxTokens, promptTokens);
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
        private int contextLength;
        private JinferEngine engine; // internal: share an already-loaded engine
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

        Builder engine(JinferEngine engine) {
            this.engine = engine;
            return this;
        }

        public JinferChatModel build() {
            if (engine == null && modelPath == null)
                throw new IllegalArgumentException("modelPath is required");
            return new JinferChatModel(this);
        }
    }
}
