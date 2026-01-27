package ai.qxotic.jota.hip;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipRuntimeTest {

    @Test
    void runtimeLoadsWhenAvailable() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
    }
}
