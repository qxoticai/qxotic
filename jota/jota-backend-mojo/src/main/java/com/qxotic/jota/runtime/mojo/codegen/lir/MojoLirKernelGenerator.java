package com.qxotic.jota.runtime.mojo.codegen.lir;

import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.mojo.codegen.MojoDialect;
import java.util.Map;
import java.util.Objects;

/** Generates Mojo kernel source from LIR graph. */
public final class MojoLirKernelGenerator {

    private final MojoDialect dialect;

    /** A generator for the newest supported Mojo ({@link MojoDialect#LATEST}). */
    public MojoLirKernelGenerator() {
        this(MojoDialect.LATEST);
    }

    public MojoLirKernelGenerator(MojoDialect dialect) {
        this.dialect = Objects.requireNonNull(dialect, "dialect");
    }

    public MojoDialect dialect() {
        return dialect;
    }

    public KernelProgram generate(LIRGraph graph, ScratchLayout scratchLayout, KernelCacheKey key) {
        // Create a valid Mojo identifier from the cache key (replace hyphens with underscores)
        String mojoFnName = "hip_lir_" + key.value().replace('-', '_');
        String mojoSource =
                new MojoNativeLirSourceGenerator(dialect, graph, scratchLayout, mojoFnName)
                        .generate();
        Map<String, String> options = Map.of();
        // Use the sanitized function name as the entry point
        KernelProgram mojoProgram = KernelProgram.source("mojo", mojoSource, mojoFnName, options);
        MojoLirArtifactStore.persist(key, mojoProgram);
        return mojoProgram;
    }
}
