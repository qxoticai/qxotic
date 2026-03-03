package com.qxotic.jota.runtime.metal;

import org.junit.jupiter.api.Test;

class MetalRuntimeDiagnosticsTest {

    @Test
    void printsMetalRuntimeDiagnostics() {
        System.out.println(MetalTestAssumptions.diagnosticsSummary());
    }
}
