package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.spring.ai.JinferChatOptions;
import java.time.Duration;
import java.util.Map;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Flat properties for {@link JinferChatAutoConfiguration}, bound under {@code
 * spring.ai.jinfer.chat} (constructor binding; a record needs no setter boilerplate).
 *
 * @param model the model as ONE string (required): a local GGUF path, or a model ref ({@code
 *     hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M}). A remote ref resolves (and downloads, when
 *     absent) at context startup, so a typo fails the boot with the hub's own message, never the
 *     first request; a local path stays local and never touches the network
 * @param companions capability to file: auxiliary files such as a multimodal projector; values take
 *     the same path-or-ref form as {@code model}
 * @param promptCache path to a cached-prompt artifact (.jkv) to mount read-only at startup;
 *     model-seed-checked
 * @param retainedSessions live conversation states kept resident and reused append-only when a
 *     request's conversation strictly extends one; default 1, zero retains no completed state
 * @param contextLength upper bound on the conversation context (default 4096); {@code 0} uses the
 *     model's declared context length; negative values are rejected
 * @param temperature sampling temperature; null uses the model recommendation
 * @param topP nucleus sampling mass; null uses the model recommendation
 * @param maxTokens maximum completion tokens; null lets the context bound the reply
 * @param seed sampling seed; null chooses a fresh seed per request
 * @param thinking the model's reasoning scaffold toggle (templates without one ignore it); default
 *     on
 * @param timeout wall-clock generation deadline; null = none
 * @param speculationDepth draft tokens per verify block for self-speculative decoding (0 disables,
 *     unset = the engine's default); inert unless the model carries a draft head (e.g. Gemma 4's
 *     MTP sidecar as {@code companions.speculation})
 */
@ConfigurationProperties("spring.ai.jinfer.chat")
public record JinferChatProperties(
        String model,
        Map<String, String> companions,
        String promptCache,
        @DefaultValue("1") int retainedSessions,
        @DefaultValue("4096") int contextLength,
        Double temperature,
        Double topP,
        Integer maxTokens,
        Long seed,
        Boolean thinking,
        Duration timeout,
        Integer speculationDepth) {

    /** Translates Boot's flat properties into Spring AI's single generation-options object. */
    public JinferChatOptions toOptions() {
        return JinferChatOptions.builder()
                .temperature(temperature)
                .topP(topP)
                .maxTokens(maxTokens)
                .seed(seed)
                .thinking(thinking)
                .timeout(timeout)
                .build();
    }
}
