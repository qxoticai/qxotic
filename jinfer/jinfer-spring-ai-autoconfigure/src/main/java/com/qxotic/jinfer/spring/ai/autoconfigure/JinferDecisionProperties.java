package com.qxotic.jinfer.spring.ai.autoconfigure;

import java.util.Map;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Configuration properties for jinfer decisions, bound under {@code spring.ai.jinfer.decision}
 * (constructor binding).
 *
 * @param model the decision GGUF (e.g. Liquid's d1 or vLLM's d3) as a local path or model ref
 * @param companions capability to file: {@code media} for the model's mmproj, the towers it reads
 *     images, audio and video through; each a model ref or a local path
 * @param contextCapacity upper bound on an input with its questions, a media row a token: 0, the
 *     default, is 4096 or the model's maximum when smaller; -1 is that maximum; another value is
 *     exact, refused at boot past it
 * @param independent answer each of a call's questions as if asked alone; off by default, the
 *     questions asked the way the model was trained to take several (d1 reads the input once for
 *     all of them, much faster, an answer then free to shift with the questions asked beside it)
 */
@ConfigurationProperties("spring.ai.jinfer.decision")
public record JinferDecisionProperties(
        String model,
        Map<String, String> companions,
        @DefaultValue("0") int contextCapacity,
        @DefaultValue("false") boolean independent) {}
