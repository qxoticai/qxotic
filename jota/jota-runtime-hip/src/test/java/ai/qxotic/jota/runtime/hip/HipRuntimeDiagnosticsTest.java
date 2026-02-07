package ai.qxotic.jota.runtime.hip;

import org.junit.jupiter.api.Test;

class HipRuntimeDiagnosticsTest {

    @Test
    void printsHipRuntimeDiagnostics() {
        System.out.println(HipTestAssumptions.diagnosticsSummary());
    }
}
