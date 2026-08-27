package com.qxotic.jinfer.spring.ai;

import org.springframework.ai.test.options.AbstractChatOptionsTests;

/**
 * Spring AI's own {@link AbstractChatOptionsTests} contract for {@link JinferChatOptions}: a
 * builder yields fresh-but-equal instances, and {@code mutate()} returns a NEW builder of the
 * provider's own type each time (Spring's advisor chain and auto-configuration both mutate options
 * per request - a builder that leaked shared state would corrupt concurrent calls). Model-free, so
 * it runs in the ordinary build.
 */
class JinferChatOptionsContractTest
        extends AbstractChatOptionsTests<JinferChatOptions, JinferChatOptions.Builder> {

    @Override
    protected Class<JinferChatOptions> getConcreteOptionsClass() {
        return JinferChatOptions.class;
    }

    @Override
    protected JinferChatOptions.Builder readyToBuildBuilder() {
        return JinferChatOptions.builder();
    }
}
