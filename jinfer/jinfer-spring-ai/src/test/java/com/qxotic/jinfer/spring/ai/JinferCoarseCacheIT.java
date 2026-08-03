package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;

/**
 * {@link AbstractCoarseCacheIT} on Nemotron-H (Mamba2 hybrid: ~90MB SSM residue per block at 30B
 * dims). Model-gated (30B - slow to load). Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class JinferCoarseCacheIT extends AbstractCoarseCacheIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.testModelCoarse", ModelFixture.NEMOTRON_30B_Q8.path().toString()));
    }
}
