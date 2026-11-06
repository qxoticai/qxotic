package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.ai.model.session.autoconfigure.SessionServiceAutoConfiguration;
import org.springframework.ai.session.InMemorySessionRepository;
import org.springframework.ai.session.SessionRepository;
import org.springframework.ai.session.SessionService;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/**
 * Context behavior, no model needed: session memory works with nothing configured, and backs off.
 */
class JinferSessionAutoConfigurationTest {

    private final ApplicationContextRunner runner =
            new ApplicationContextRunner()
                    .withConfiguration(
                            AutoConfigurations.of(
                                    JinferSessionAutoConfiguration.class,
                                    SessionServiceAutoConfiguration.class));

    @Test
    void sessionServiceOverAnInMemoryRepositoryByDefault() {
        runner.run(
                context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(SessionService.class);
                    assertThat(context.getBean(SessionRepository.class))
                            .isInstanceOf(InMemorySessionRepository.class);
                });
    }

    @Test
    void anApplicationsRepositoryWins() {
        SessionRepository mine = InMemorySessionRepository.builder().build();
        runner.withBean("mine", SessionRepository.class, () -> mine)
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).hasSingleBean(SessionRepository.class);
                            assertThat(context.getBean(SessionRepository.class)).isSameAs(mine);
                            assertThat(context).hasSingleBean(SessionService.class);
                        });
    }
}
