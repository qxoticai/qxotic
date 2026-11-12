package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.spring.ai.JinferDecisionModel;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

/**
 * Wires a {@link JinferDecisionModel} from {@code spring.ai.jinfer.decision.*} when no {@link
 * DecisionModel} bean is supplied. Selected as Spring AI selects every model type, with {@code
 * spring.ai.model.decision=jinfer}: an app that asks no typed questions configures nothing.
 */
@AutoConfiguration
@ConditionalOnClass({JinferDecisionModel.class, DecisionModel.class})
@ConditionalOnProperty(name = "spring.ai.model.decision", havingValue = "jinfer")
@EnableConfigurationProperties(JinferDecisionProperties.class)
public class JinferDecisionAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean(DecisionModel.class)
    public JinferDecisionModel jinferDecisionModel(
            JinferDecisionProperties properties, ObjectProvider<VideoSampler> videoSampler) {
        if (!StringUtils.hasText(properties.model())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.decision.model is required: a decision GGUF (e.g. Liquid's"
                            + " d1 or vLLM's d3) as a local path or a hub ref");
        }
        JinferDecisionModel.Builder builder =
                JinferDecisionModel.builder()
                        .contextCapacity(properties.contextCapacity())
                        .independent(properties.independent());
        videoSampler.ifAvailable(builder::videoSampler); // a VideoSampler bean overrides UNIFORM
        Sources.model(
                "spring.ai.jinfer.decision.model",
                properties.model(),
                builder::model,
                builder::modelPath);
        Sources.companions(
                "spring.ai.jinfer.decision",
                properties.companions(),
                builder::companion,
                builder::companionPath);
        return builder.build();
    }
}
