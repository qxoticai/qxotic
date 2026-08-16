package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;

/**
 * {@link AbstractCoarseCacheIT} on Nemotron-H (Mamba2 hybrid: ~90MB SSM residue per block at 30B
 * dims). Model-gated (30B - slow to load). Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl jinfer-langchain4j}
 */
@Tag("integration")
class JinferCoarseCacheIT extends AbstractCoarseCacheIT {

    @Override
    Path modelPath() {
        return TestModels.require(
                "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf");
    }
}
