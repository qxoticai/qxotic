package com.qxotic.jinfer.llm;

import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;

/**
 * The tokenizer sibling of {@code ModelProvider}: a {@link java.util.ServiceLoader} service a
 * model-provider jar registers ({@code META-INF/services}) to teach the GGUF tokenizer loader its
 * family's pre-tokenizer, normalizer, or tokenization model - the pieces resolved by name from the
 * GGUF's {@code tokenizer.ggml.*} metadata ({@code tokenizer.ggml.pre} for pre-tokenizers).
 *
 * <p>Customizers apply AFTER toknroll's builtins and jinfer's bundled registrations, so a
 * customizer can also override one. Without a matching registration, an unknown name fails loudly
 * at load with the register-it remedy - never silently mis-tokenizes.
 *
 * <pre>{@code
 * public final class MyFamilyTokenizer implements TokenizerCustomizer {
 *     public void customize(GGUFTokenizerLoader.Builder builder) {
 *         builder.registerPreTokenizer("myfamily", g -> Splitter.regex(MY_PATTERN))
 *                .registerNormalizer("myfamily", g -> Normalizer.identity());
 *     }
 * }
 * }</pre>
 */
public interface TokenizerCustomizer {

    /** Customize the loader builder with this family's tokenizer pieces. */
    void customize(GGUFTokenizerLoader.Builder builder);
}
