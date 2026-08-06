package com.qxotic.jinfer.spring.ai.autoconfigure;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Flat properties for {@link JinferChatAutoConfiguration}, bound under {@code
 * spring.ai.jinfer.chat} (constructor binding; a record needs no setter boilerplate).
 *
 * @param modelPath path to the GGUF model file (required)
 * @param companions capability to file: the auxiliary files that give the model a capability it
 *     multimodal models
 * @param cachedPrompts path to a cached-prompt artifact (.jkv) to mount at startup;
 *     model-seed-checked
 * @param cachedSessions live conversation states kept resident and reused append-only when a
 *     request's conversation strictly extends one (the multi-turn zero-restore tier); 0 (default)
 *     disables the pool
 * @param contextLength context window; 0 = the model's own maximum
 * @param thinking the model's reasoning scaffold toggle (templates without one ignore it); default
 *     on
 * @param timeout wall-clock generation deadline; null = none
 */
@ConfigurationProperties("spring.ai.jinfer.chat")
public record JinferChatProperties(
        String modelPath,
        java.util.Map<String, String> companions,
        String cachedPrompts,
        @DefaultValue("0") int cachedSessions,
        @DefaultValue("0") int contextLength,
        Double temperature,
        Double topP,
        Integer maxTokens,
        Long seed,
        Boolean thinking,
        Duration timeout) {}
