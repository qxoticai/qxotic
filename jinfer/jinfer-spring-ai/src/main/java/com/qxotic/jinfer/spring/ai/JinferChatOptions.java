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
 * extras ({@code seed}, {@code minP}, {@code thinking}, {@code timeout}). Implements {@link
 * ToolCallingChatOptions} - required, or the framework's ToolCallingAdvisor silently skips tool
 * execution - and {@link StructuredOutputChatOptions}, the native structured-output hook (the
 * schema is enforced token-level via grammar-constrained decoding).
 */
public final class JinferChatOptions extends DefaultToolCallingChatOptions
        implements StructuredOutputChatOptions {

    private final Long seed;
    private final Double minP;
    private final Boolean thinking;
    private final Integer maxReasoningTokens;
    private final String reasoningCutoffMessage;
    private final Duration timeout;
    private final String outputSchema;
    private final String grammar;

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
        this.minP = b.minP;
        this.thinking = b.thinking;
        this.maxReasoningTokens = b.maxReasoningTokens;
        this.reasoningCutoffMessage = b.reasoningCutoffMessage;
        this.timeout = b.timeout;
        this.outputSchema = b.outputSchema;
        this.grammar = b.grammar;
    }

    /** Sampling seed; null = a fresh random seed per request (set one to pin sampling). */
    public Long getSeed() {
        return seed;
    }

    /**
     * Min-p cutoff relative to the top token, in [0,1] (0 disables); null falls to the model's
     * recommended value, else 0.05. Spring AI has no standard slot for min-p, so it lives here with
     * the other jinfer extras.
     */
    public Double getMinP() {
        return minP;
    }

    /** The model's reasoning scaffold toggle (templates without one ignore it). Default on. */
    public Boolean getThinking() {
        return thinking;
    }

    /** Reasoning-span cap in generated tokens; {@code -1} uncaps, null = the family's policy. */
    public Integer getReasoningBudget() {
        return maxReasoningTokens;
    }

    /** What the model "decides" when the budget runs out, in its own words; null = a break. */
    public String getReasoningBudgetMessage() {
        return reasoningCutoffMessage;
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

    /**
     * A raw GBNF grammar the reply must conform to - the generalization of {@link
     * #getOutputSchema()}, which compiles a schema to one; null = none. Not combined with an output
     * schema: a request carrying both is refused.
     */
    public String getGrammar() {
        return grammar;
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
                .minP(minP)
                .thinking(thinking)
                .maxReasoningTokens(maxReasoningTokens)
                .reasoningCutoffMessage(reasoningCutoffMessage)
                .timeout(timeout)
                .outputSchema(outputSchema)
                .grammar(grammar);
    }

    /** Adapts portable Spring options without inventing a second defaults-merge policy. */
    static JinferChatOptions from(ChatOptions o) {
        if (o instanceof JinferChatOptions j) return j;
        Builder b =
                builder()
                        .model(o.getModel())
                        .frequencyPenalty(o.getFrequencyPenalty())
                        .maxTokens(o.getMaxTokens())
                        .presencePenalty(o.getPresencePenalty())
                        .stopSequences(o.getStopSequences())
                        .temperature(o.getTemperature())
                        .topK(o.getTopK())
                        .topP(o.getTopP());
        if (o instanceof ToolCallingChatOptions t) {
            b.toolCallbacks(t.getToolCallbacks()).toolContext(t.getToolContext());
        }
        if (o instanceof StructuredOutputChatOptions s && s.getOutputSchema() != null) {
            b.outputSchema(s.getOutputSchema());
        }
        if (o instanceof JinferChatOptions j && j.grammar != null) b.grammar(j.grammar);
        return b.build();
    }

    public static final class Builder extends DefaultToolCallingChatOptions.Builder<Builder>
            implements StructuredOutputChatOptions.Builder<Builder> {
        private Long seed;
        private Double minP;
        private Boolean thinking;
        private Integer maxReasoningTokens;
        private String reasoningCutoffMessage;
        private Duration timeout;
        private String outputSchema;
        private String grammar;

        private Builder() {}

        public Builder seed(Long seed) {
            this.seed = seed;
            return this;
        }

        public Builder minP(Double minP) {
            this.minP = minP;
            return this;
        }

        public Builder thinking(Boolean thinking) {
            this.thinking = thinking;
            return this;
        }

        public Builder maxReasoningTokens(Integer maxReasoningTokens) {
            if (maxReasoningTokens != null && maxReasoningTokens < -1)
                throw new IllegalArgumentException("maxReasoningTokens " + maxReasoningTokens);
            this.maxReasoningTokens = maxReasoningTokens;
            return this;
        }

        public Builder reasoningCutoffMessage(String reasoningCutoffMessage) {
            this.reasoningCutoffMessage = reasoningCutoffMessage;
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

        /** Raw GBNF for the reply; see {@link JinferChatOptions#getGrammar()}. */
        public Builder grammar(String grammar) {
            this.grammar = grammar;
            return this;
        }

        /**
         * Spring ChatClient's precedence seam: values from {@code other} override this builder;
         * inherited fields and tools are handled by Spring, Jinfer fields here.
         */
        @Override
        public Builder combineWith(ChatOptions.Builder<?> other) {
            super.combineWith(other);
            if (other instanceof Builder j) {
                if (j.seed != null) seed = j.seed;
                if (j.minP != null) minP = j.minP;
                if (j.thinking != null) thinking = j.thinking;
                if (j.maxReasoningTokens != null) maxReasoningTokens = j.maxReasoningTokens;
                if (j.reasoningCutoffMessage != null)
                    reasoningCutoffMessage = j.reasoningCutoffMessage;
                if (j.timeout != null) timeout = j.timeout;
                if (j.outputSchema != null) outputSchema = j.outputSchema;
                if (j.grammar != null) grammar = j.grammar;
            }
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
