package com.qxotic.jinfer.spring.ai;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.tool.DefaultToolCallingChatOptions;
import org.springframework.ai.model.tool.StructuredOutputChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;

/**
 * {@link ChatOptions} for {@link JinferChatModel}: the Spring AI common options plus jinfer's
 * extras ({@code seed}, {@code thinking}, {@code timeout}). Implements {@link
 * ToolCallingChatOptions} - required, or the framework's ToolCallingAdvisor silently skips tool
 * execution - and {@link StructuredOutputChatOptions}, the native structured-output hook (the
 * schema is enforced token-level via grammar-constrained decoding).
 */
public final class JinferChatOptions extends DefaultToolCallingChatOptions
        implements StructuredOutputChatOptions {

    private final Long seed;
    private final Boolean thinking;
    private final Duration timeout;
    private final String outputSchema;

    private JinferChatOptions(
            Builder b,
            List<ToolCallback> toolCallbacks,
            Map<String, Object> toolContext,
            String model,
            Double frequencyPenalty,
            Integer maxTokens,
            Double presencePenalty,
            List<String> stopSequences,
            Double temperature,
            Integer topK,
            Double topP) {
        super(
                toolCallbacks,
                toolContext,
                model,
                frequencyPenalty,
                maxTokens,
                presencePenalty,
                stopSequences,
                temperature,
                topK,
                topP);
        this.seed = b.seed;
        this.thinking = b.thinking;
        this.timeout = b.timeout;
        this.outputSchema = b.outputSchema;
    }

    /** Deterministic sampling seed; null = the model's default (42). */
    public Long getSeed() {
        return seed;
    }

    /** The model's reasoning scaffold toggle (templates without one ignore it). Default on. */
    public Boolean getThinking() {
        return thinking;
    }

    /** Wall-clock generation deadline; null = none. */
    public Duration getTimeout() {
        return timeout;
    }

    /** The JSON schema the reply must conform to (enforced by grammar); null = free text. */
    @Override
    public String getOutputSchema() {
        return outputSchema;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Builder mutate() {
        return new Builder()
                .model(getModel())
                .frequencyPenalty(getFrequencyPenalty())
                .maxTokens(getMaxTokens())
                .presencePenalty(getPresencePenalty())
                .stopSequences(getStopSequences())
                .temperature(getTemperature())
                .topK(getTopK())
                .topP(getTopP())
                .toolCallbacks(getToolCallbacks())
                .toolContext(getToolContext())
                .seed(seed)
                .thinking(thinking)
                .timeout(timeout)
                .outputSchema(outputSchema);
    }

    /**
     * Copies a (possibly foreign) {@link ChatOptions} onto {@code base}: common fields by getters,
     * tool plumbing when it is a {@link ToolCallingChatOptions}, jinfer extras only from another
     * {@code JinferChatOptions}. Foreign options are typically sparse, so only their NON-NULL
     * fields override - a null must not wipe a configured default.
     */
    static JinferChatOptions copyOnto(JinferChatOptions base, ChatOptions o) {
        Builder b = base.mutate();
        if (o.getModel() != null) b.model(o.getModel());
        if (o.getFrequencyPenalty() != null) b.frequencyPenalty(o.getFrequencyPenalty());
        if (o.getMaxTokens() != null) b.maxTokens(o.getMaxTokens());
        if (o.getPresencePenalty() != null) b.presencePenalty(o.getPresencePenalty());
        if (o.getStopSequences() != null) b.stopSequences(o.getStopSequences());
        if (o.getTemperature() != null) b.temperature(o.getTemperature());
        if (o.getTopK() != null) b.topK(o.getTopK());
        if (o.getTopP() != null) b.topP(o.getTopP());
        if (o instanceof ToolCallingChatOptions t) {
            if (t.getToolCallbacks() != null) b.toolCallbacks(t.getToolCallbacks());
            if (t.getToolContext() != null) b.toolContext(t.getToolContext());
        }
        if (o instanceof StructuredOutputChatOptions s && s.getOutputSchema() != null) {
            b.outputSchema(s.getOutputSchema());
        }
        if (o instanceof JinferChatOptions j) {
            if (j.seed != null) b.seed(j.seed);
            if (j.thinking != null) b.thinking(j.thinking);
            if (j.timeout != null) b.timeout(j.timeout);
        }
        return b.build();
    }

    public static final class Builder extends DefaultToolCallingChatOptions.Builder<Builder>
            implements StructuredOutputChatOptions.Builder<Builder> {
        private Long seed;
        private Boolean thinking;
        private Duration timeout;
        private String outputSchema;

        private Builder() {}

        public Builder seed(Long seed) {
            this.seed = seed;
            return this;
        }

        public Builder thinking(Boolean thinking) {
            this.thinking = thinking;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        @Override
        public Builder outputSchema(String outputSchema) {
            this.outputSchema = outputSchema;
            return this;
        }

        @Override
        public JinferChatOptions build() {
            return new JinferChatOptions(
                    this,
                    toolCallbacks,
                    toolContext,
                    model,
                    frequencyPenalty,
                    maxTokens,
                    presencePenalty,
                    stopSequences,
                    temperature,
                    topK,
                    topP);
        }
    }
}
