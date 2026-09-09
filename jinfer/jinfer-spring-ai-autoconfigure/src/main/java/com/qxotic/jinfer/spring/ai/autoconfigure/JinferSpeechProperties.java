package com.qxotic.jinfer.spring.ai.autoconfigure;

import java.util.Map;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Configuration properties for jinfer text-to-speech, bound under {@code spring.ai.jinfer.speech}
 * (constructor binding).
 *
 * @param model the speech GGUF (e.g. an Inflect model) as a local path or model ref; configuring it
 *     is what activates the model, the same rule the rerank properties use
 * @param companions capability to file: {@code lexicon} for an Inflect model's pronunciation
 *     lexicon; values take the same path-or-ref form as {@code model}
 * @param speed playback rate multiplier; unset leaves the model's own pace alone
 * @param maxInputChars refuses an utterance longer than this before synthesis starts. Speech cost
 *     is driven by input length, so an unbounded request is a denial-of-service shape rather than a
 *     slow one; {@code 0} keeps the model's own bound
 */
@ConfigurationProperties("spring.ai.jinfer.speech")
public record JinferSpeechProperties(
        String model,
        Map<String, String> companions,
        Double speed,
        @DefaultValue("0") int maxInputChars) {}
