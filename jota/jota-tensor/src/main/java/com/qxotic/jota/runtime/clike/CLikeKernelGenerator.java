package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelProgram;
import java.util.Locale;
import java.util.Objects;

public final class CLikeKernelGenerator {

    private static final int KERNEL_HASH_CHARS = 12;

    private final CLikeDialect dialect;

    public CLikeKernelGenerator(CLikeDialect dialect) {
        this.dialect = Objects.requireNonNull(dialect, "dialect");
    }

    public KernelProgram generate(LIRGraph graph, ScratchLayout scratchLayout, KernelCacheKey key) {
        String kernelName = kernelNameFor(dialect.language(), key.value());
        String source = dialect.renderSource(graph, scratchLayout, kernelName);
        return KernelProgram.source(dialect.language(), source, kernelName);
    }

    private static String kernelNameFor(String language, String cacheKey) {
        String prefix = sanitizeIdentifier(language.toLowerCase(Locale.ROOT));
        String hash =
                cacheKey.length() <= KERNEL_HASH_CHARS
                        ? cacheKey
                        : cacheKey.substring(0, KERNEL_HASH_CHARS);
        return prefix + "_lir_" + sanitizeIdentifier(hash);
    }

    private static String sanitizeIdentifier(String text) {
        StringBuilder out = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if ((c >= 'a' && c <= 'z')
                    || (c >= 'A' && c <= 'Z')
                    || (c >= '0' && c <= '9')
                    || c == '_') {
                out.append(c);
            } else {
                out.append('_');
            }
        }
        return out.toString();
    }
}
