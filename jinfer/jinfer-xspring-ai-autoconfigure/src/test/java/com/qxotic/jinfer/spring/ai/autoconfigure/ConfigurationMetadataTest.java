package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/**
 * The IDE-facing contract: the jar carries spring-configuration-metadata.json naming the
 * spring.ai.jinfer.* properties. A regression here means the configuration processor stopped
 * running - JDK 23+ javac ignores classpath-discovered processors, so the pom names it explicitly
 * in annotationProcessorPaths. Text-level assertions on purpose: no JSON parser is a compile
 * dependency of this module, and the contract under test is "these keys appear", not JSON shape.
 */
class ConfigurationMetadataTest {

    @Test
    void theJarCarriesIdeMetadataForEveryModelArea() throws Exception {
        String json;
        try (InputStream in =
                getClass().getResourceAsStream("/META-INF/spring-configuration-metadata.json")) {
            assertThat(in).as("spring-configuration-metadata.json on the classpath").isNotNull();
            json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        assertThat(json)
                .contains(
                        "\"spring.ai.jinfer.chat\"",
                        "\"spring.ai.jinfer.embedding\"",
                        "\"spring.ai.jinfer.rerank\"",
                        "\"spring.ai.jinfer.speech\"",
                        "\"spring.ai.jinfer.chat.model\"",
                        "\"spring.ai.jinfer.chat.context-length\"",
                        "\"spring.ai.jinfer.embedding.model\"",
                        "\"spring.ai.jinfer.embedding.context-length\"",
                        "\"spring.ai.jinfer.rerank.instruction\"",
                        "\"spring.ai.jinfer.speech.max-input-chars\"");
    }
}
