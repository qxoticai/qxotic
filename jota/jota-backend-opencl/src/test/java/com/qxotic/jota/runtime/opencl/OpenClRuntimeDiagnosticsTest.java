package com.qxotic.jota.runtime.opencl;

import org.junit.jupiter.api.Test;

class OpenClRuntimeDiagnosticsTest {

    @Test
    void printsOpenClRuntimeDiagnostics() {
        System.out.println(OpenClTestAssumptions.diagnosticsSummary());
    }
}
