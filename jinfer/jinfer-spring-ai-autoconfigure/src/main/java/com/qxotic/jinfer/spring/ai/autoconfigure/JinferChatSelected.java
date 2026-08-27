package com.qxotic.jinfer.spring.ai.autoconfigure;

import org.springframework.context.annotation.Condition;
import org.springframework.context.annotation.ConditionContext;
import org.springframework.core.type.AnnotatedTypeMetadata;
import org.springframework.util.StringUtils;

/**
 * When the jinfer chat model is wanted: {@code spring.ai.model.chat=jinfer} says so explicitly (and
 * a missing model is then a boot error, named), or no provider is selected and a jinfer chat model
 * is configured. An app that uses the starter for embeddings, reranking or speech only, with no
 * chat selection and no chat model, must boot; a {@code matchIfMissing} default alone made it fail
 * with "spring.ai.jinfer.chat.model is required".
 */
final class JinferChatSelected implements Condition {

    @Override
    public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
        String selected = context.getEnvironment().getProperty("spring.ai.model.chat");
        if (selected != null) return "jinfer".equals(selected);
        return StringUtils.hasText(
                context.getEnvironment().getProperty("spring.ai.jinfer.chat.model"));
    }
}
