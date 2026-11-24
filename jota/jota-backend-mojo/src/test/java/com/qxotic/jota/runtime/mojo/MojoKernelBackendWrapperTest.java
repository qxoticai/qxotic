package com.qxotic.jota.runtime.mojo;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.runtime.mojo.codegen.MojoDialect;
import org.junit.jupiter.api.Test;

class MojoKernelBackendWrapperTest {

    @Test
    void wrapperForMojo0261() {
        String w = MojoKernelBackend.wrapper("# kernel", "k", MojoDialect.V0_26_1);
        assertTrue(w.startsWith("# kernel\n"));
        assertTrue(w.contains("from std.gpu.host import DeviceContext"));
        assertTrue(w.contains("def main():\n"));
        assertTrue(w.contains("var ctx = DeviceContext(api=\"hip\")"));
        assertTrue(w.contains("_ = ctx.compile_function[k, k]()"));
    }

    @Test
    void wrapperForMojo0262RaisesFromMain() {
        String w = MojoKernelBackend.wrapper("# kernel", "k", MojoDialect.V0_26_2);
        assertTrue(w.contains("def main() raises:\n"));
        assertTrue(w.contains("from std.gpu.host import DeviceContext"));
    }

    @Test
    void wrapperForMojo1UsesMaxGpuAndSingleParameter() {
        String w = MojoKernelBackend.wrapper("# kernel", "k", MojoDialect.V1);
        assertTrue(w.contains("from max.gpu.host import DeviceContext"));
        assertTrue(w.contains("def main() raises:\n"));
        assertTrue(w.contains("_ = ctx.compile_function[k]()"));
    }
}
