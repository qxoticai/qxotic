package com.qxotic.jinfer.langchain4j;

import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import java.time.Duration;
import java.util.Objects;

/**
 * jinfer's provider-specific request parameters - the standard langchain4j knobs plus:
 *
 * <ul>
 *   <li>{@code grammar}: a raw GBNF grammar constraining the WHOLE reply (think-span gated, like
 *       the JSON response format it generalizes) - the sampler cannot emit anything outside it.
 *       Mutually exclusive with a JSON response format (a grammar cannot admit the format's
 *       syntax); the rejection is loud. Tools may ride along: the grammar takes over once the tool
 *       round is done.
 *   <li>{@code seed}: this request's sampler seed - byte-identical replay of a specific call, or
 *       deliberate variation across calls; null = the model builder's seed.
 * </ul>
 *
 * <p>Works at request level and as {@code defaultRequestParameters} (a standing grammar turns a
 * model instance into a dedicated classifier - the only way to get grammar guarantees through
 * AiServices, which builds its own requests):
 *
 * <pre>{@code
 * // per request
 * model.chat(ChatRequest.builder()
 *         .messages(UserMessage.from("Is the sky blue? yes or no."))
 *         .parameters(JinferChatRequestParameters.builder()
 *                 .grammar("root ::= \"yes\" | \"no\"")
 *                 .build())
 *         .build());
 *
 * // standing: every request through this instance is constrained (the AiServices path)
 * var classifier = JinferChatModel.builder()
 *         .modelPath(gguf)
 *         .defaultRequestParameters(JinferChatRequestParameters.builder()
 *                 .grammar("root ::= \"positive\" | \"negative\"")
 *                 .build())
 *         .build();
 * }</pre>
 *
 * <p>GBNF dialect and pitfalls: see {@code com.qxotic.jinfer.llm.Grammar}.
 */
public class JinferChatRequestParameters extends DefaultChatRequestParameters {

    private final String grammar;
    private final Long seed;
    private final Double minP;
    private final Integer reasoningBudget;
    private final Boolean thinking;
    private final String reasoningBudgetMessage;
    private final Duration timeout;

    protected JinferChatRequestParameters(Builder builder) {
        super(builder);
        this.grammar = builder.grammar;
        this.seed = builder.seed;
        this.minP = builder.minP;
        this.reasoningBudget = builder.reasoningBudget;
        this.thinking = builder.thinking;
        this.reasoningBudgetMessage = builder.reasoningBudgetMessage;
        this.timeout = builder.timeout;
    }

    /** Raw GBNF constraining the reply, or null. */
    public String grammar() {
        return grammar;
    }

    /** This request's sampler seed, or null for the model's. */
    public Long seed() {
        return seed;
    }

    /**
     * Min-p cutoff relative to the top token, in [0,1] (0 disables); null falls to the model's
     * recommended value, else 0.05. langchain4j has no standard slot for min-p, so it lives here
     * with the other jinfer extras.
     */
    public Double minP() {
        return minP;
    }

    /** Reasoning-span cap for this request; null = the model's builder default. */
    public Integer reasoningBudget() {
        return reasoningBudget;
    }

    /** The reasoning scaffold for this request; null = the model's builder default. */
    public Boolean thinking() {
        return thinking;
    }

    /** What the model "decides" when the budget runs out; null = the model's builder default. */
    public String reasoningBudgetMessage() {
        return reasoningBudgetMessage;
    }

    /** Wall-clock deadline for this request; null = the model's builder default. */
    public Duration timeout() {
        return timeout;
    }

    @Override
    public JinferChatRequestParameters overrideWith(ChatRequestParameters that) {
        return builder().overrideWith(this).overrideWith(that).build();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof JinferChatRequestParameters that
                && super.equals(that)
                && Objects.equals(grammar, that.grammar)
                && Objects.equals(seed, that.seed)
                && Objects.equals(minP, that.minP)
                && Objects.equals(reasoningBudget, that.reasoningBudget)
                && Objects.equals(thinking, that.thinking)
                && Objects.equals(reasoningBudgetMessage, that.reasoningBudgetMessage)
                && Objects.equals(timeout, that.timeout);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                super.hashCode(),
                grammar,
                seed,
                minP,
                reasoningBudget,
                thinking,
                reasoningBudgetMessage,
                timeout);
    }

    @Override
    public String toString() {
        return "JinferChatRequestParameters{grammar="
                + (grammar == null ? "null" : "'" + grammar + "'")
                + ", seed="
                + seed
                + ", minP="
                + minP
                + ", reasoningBudget="
                + reasoningBudget
                + ", thinking="
                + thinking
                + ", reasoningBudgetMessage="
                + (reasoningBudgetMessage == null ? "null" : "'" + reasoningBudgetMessage + "'")
                + ", timeout="
                + timeout
                + ", "
                + super.toString()
                + "}";
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder extends DefaultChatRequestParameters.Builder<Builder> {

        private String grammar;
        private Long seed;
        private Double minP;
        private Integer reasoningBudget;
        private Boolean thinking;
        private String reasoningBudgetMessage;
        private Duration timeout;

        @Override
        public Builder overrideWith(ChatRequestParameters parameters) {
            super.overrideWith(parameters);
            if (parameters instanceof JinferChatRequestParameters j) {
                if (j.grammar() != null) grammar(j.grammar());
                if (j.seed() != null) seed(j.seed());
                if (j.minP() != null) minP(j.minP());
                if (j.reasoningBudget() != null) reasoningBudget(j.reasoningBudget());
                if (j.thinking() != null) thinking(j.thinking());
                if (j.reasoningBudgetMessage() != null)
                    reasoningBudgetMessage(j.reasoningBudgetMessage());
                if (j.timeout() != null) timeout(j.timeout());
            }
            return this;
        }

        public Builder grammar(String grammar) {
            this.grammar = grammar;
            return this;
        }

        public Builder seed(Long seed) {
            this.seed = seed;
            return this;
        }

        public Builder minP(Double minP) {
            if (minP != null && !(minP >= 0 && minP <= 1))
                throw new IllegalArgumentException("minP must be within [0, 1]: " + minP);
            this.minP = minP;
            return this;
        }

        /** Caps the reasoning span for this request; {@code -1} uncaps, null leaves the default. */
        public Builder reasoningBudget(Integer reasoningBudget) {
            if (reasoningBudget != null && reasoningBudget < -1)
                throw new IllegalArgumentException("reasoningBudget " + reasoningBudget);
            this.reasoningBudget = reasoningBudget;
            return this;
        }

        /** The reasoning scaffold for this request; a model that always reasons refuses false. */
        public Builder thinking(Boolean thinking) {
            this.thinking = thinking;
            return this;
        }

        /** The model's own words when the reasoning budget runs out; null leaves the default. */
        public Builder reasoningBudgetMessage(String message) {
            this.reasoningBudgetMessage = message;
            return this;
        }

        /** Wall-clock deadline for this request; null leaves the default. */
        public Builder timeout(Duration timeout) {
            if (timeout != null && timeout.isNegative())
                throw new IllegalArgumentException("timeout must be >= 0: " + timeout);
            this.timeout = timeout;
            return this;
        }

        @Override
        public JinferChatRequestParameters build() {
            return new JinferChatRequestParameters(this);
        }
    }
}
