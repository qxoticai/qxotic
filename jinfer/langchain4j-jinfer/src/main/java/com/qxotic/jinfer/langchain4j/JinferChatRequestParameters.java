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
 * AiServices, which builds its own requests).
 */
public class JinferChatRequestParameters extends DefaultChatRequestParameters {

    private final String grammar;
    private final Long seed;

    protected JinferChatRequestParameters(Builder builder) {
        super(builder);
        this.grammar = builder.grammar;
        this.seed = builder.seed;
    }

    /** Raw GBNF constraining the reply, or null. */
    public String grammar() {
        return grammar;
    }

    /** This request's sampler seed, or null for the model's. */
    public Long seed() {
        return seed;
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
                && Objects.equals(seed, that.seed);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), grammar, seed);
    }

    @Override
    public String toString() {
        return "JinferChatRequestParameters{grammar="
                + (grammar == null ? "null" : "'" + grammar + "'")
                + ", seed="
                + seed
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

        @Override
        public Builder overrideWith(ChatRequestParameters parameters) {
            super.overrideWith(parameters);
            if (parameters instanceof JinferChatRequestParameters j) {
                if (j.grammar() != null) grammar(j.grammar());
                if (j.seed() != null) seed(j.seed());
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

        @Override
        public JinferChatRequestParameters build() {
            return new JinferChatRequestParameters(this);
        }
    }
}
