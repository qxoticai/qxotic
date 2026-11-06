package com.qxotic.jinfer.spring.ai.autoconfigure;

import org.springframework.ai.session.InMemorySessionRepository;
import org.springframework.ai.session.SessionRepository;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;

/**
 * An in-memory session repository when none is configured, so the starter's session memory works
 * out of the box: Spring AI's {@code SessionServiceAutoConfiguration} builds the {@code
 * SessionService} over a repository bean and ships none for memory. A JDBC repository on the
 * classpath, or the application's own, wins: this runs after Spring's JDBC auto-configuration and
 * backs off from any repository bean.
 */
@AutoConfiguration(
        afterName =
                "org.springframework.ai.model.session.jdbc.autoconfigure.JdbcSessionRepositoryAutoConfiguration",
        beforeName =
                "org.springframework.ai.model.session.autoconfigure.SessionServiceAutoConfiguration")
@ConditionalOnClass(SessionRepository.class)
public class JinferSessionAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean(SessionRepository.class)
    SessionRepository jinferInMemorySessionRepository() {
        return InMemorySessionRepository.builder().build();
    }
}
