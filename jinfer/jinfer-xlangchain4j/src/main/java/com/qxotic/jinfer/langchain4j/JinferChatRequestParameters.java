package com.qxotic.jinfer.langchain4j;

import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import java.util.Objects;

/**
 * jinfer's provider-specific request parameters - the standard langchain4j knobs plus:
 *
 * <ul>
 *   <li>{@code grammar}: a raw GBNF grammar constraining the WHOLE reply (think-span gated, like
 *       the JSON response format it generalizes) - the sampler cannot emit anything outside it.
 *       Mutually exclusive with tools and with a JSON response format (a grammar cannot admit
 *       call/format syntax); both rejections are loud.
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

    protected JinferChatRequestParameters(Builder builder) {
        super(builder);
        this.grammar = builder.grammar;
        this.seed = builder.seed;
        this.minP = builder.minP;
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
                && Objects.equals(minP, that.minP);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), grammar, seed, minP);
    }

    @Override
    public String toString() {
        return "JinferChatRequestParameters{grammar="
                + (grammar == null ? "null" : "'" + grammar + "'")
                + ", seed="
                + seed
                + ", minP="
                + minP
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

        @Override
        public Builder overrideWith(ChatRequestParameters parameters) {
            super.overrideWith(parameters);
            if (parameters instanceof JinferChatRequestParameters j) {
                if (j.grammar() != null) grammar(j.grammar());
                if (j.seed() != null) seed(j.seed());
                if (j.minP() != null) minP(j.minP());
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
            this.minP = minP;
            return this;
        }

        @Override
        public JinferChatRequestParameters build() {
            return new JinferChatRequestParameters(this);
        }
    }
}
