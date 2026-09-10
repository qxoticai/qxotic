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
 * @param model the model as ONE string (required): a local GGUF path, or a model ref
 *     (unsloth/gemma-4-E2B-it-GGUF:Q4_K_M). A remote ref resolves (and downloads, when absent) at
 *     context startup, so a typo fails the boot with the hub's own message, never the first
 *     request; a local path stays local and never touches the network
 * @param companions capability to file: auxiliary files such as a multimodal projector; values take
 *     the same path-or-ref form as model
 * @param promptCache path to a cached-prompt artifact (.jkv) to mount read-only at startup;
 *     model-seed-checked
 * @param retainedSessions live conversation states kept resident and reused append-only when a
 *     request's conversation strictly extends one; default 1, zero retains no completed state
 * @param contextCapacity upper bound on the conversation context (unset: 4096 or the model's
 *     maxContextLength when smaller; above it is refused at boot); 0 uses the model's
 *     maxContextLength; negative values are rejected
 * @param temperature sampling temperature; null uses the model recommendation
 * @param topP nucleus sampling mass; null uses the model recommendation
 * @param topK top-k cutoff; null uses the model recommendation, 0 disables
 * @param minP minimum probability relative to the top token; null uses the model recommendation
 * @param maxTokens maximum completion tokens; null lets the context bound the reply
 * @param seed sampling seed; null chooses a fresh seed per request
 * @param thinking the model's reasoning scaffold toggle (templates without one ignore it; a model
 *     that always reasons refuses false); default on
 * @param reasoningBudget cap on the reasoning span in generated tokens, the lever for models that
 *     always reason; -1 uncaps; unset = the family's policy
 * @param reasoningBudgetMessage what the model "decides" when the budget runs out, in its own
 *     words; unset = a paragraph break
 * @param timeout wall-clock generation deadline; null = none
 * @param speculationDepth draft tokens per verify block for self-speculative decoding (0 disables,
 *     unset = the engine's default); inert unless the model carries a draft head (e.g. Gemma 4's
 *     MTP sidecar as companions.speculation)
 */
@ConfigurationProperties("spring.ai.jinfer.chat")
public record JinferChatProperties(
        String model,
        Map<String, String> companions,
        String promptCache,
        @DefaultValue("1") int retainedSessions,
        Integer contextCapacity,
        Double temperature,
        Double topP,
        Integer topK,
        Double minP,
        Integer maxTokens,
        Long seed,
        Boolean thinking,
        Integer reasoningBudget,
        String reasoningBudgetMessage,
        Duration timeout,
        Integer speculationDepth) {

    /** Translates Boot's flat properties into Spring AI's single generation-options object. */
    public JinferChatOptions toOptions() {
        return JinferChatOptions.builder()
                .temperature(temperature)
                .topP(topP)
                .topK(topK)
                .minP(minP)
                .maxTokens(maxTokens)
                .seed(seed)
                .thinking(thinking)
                .reasoningBudget(reasoningBudget)
                .reasoningBudgetMessage(reasoningBudgetMessage)
                .timeout(timeout)
                .build();
    }
}
