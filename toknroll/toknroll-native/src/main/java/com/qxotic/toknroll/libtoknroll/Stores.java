package com.qxotic.toknroll.libtoknroll;

import com.qxotic.jinfer.hub.HuggingFaceSource;
import com.qxotic.jinfer.hub.ModelScopeSource;
import com.qxotic.jinfer.hub.ModelStore;
import java.nio.file.Path;

/**
 * The store hub references resolve through: jinfer's, so a file either library fetched serves the
 * other, configured by toknroll's own two knobs. {@code -Dtoknroll.cache.root} ({@code
 * TOKNROLL_CACHE_ROOT} in the environment) moves the cache, {@code -Dtoknroll.offline=true} builds
 * a store with no sources, which serves the cache and refuses the network. Absent both, the
 * standard store applies, where the {@code jinfer.*} spellings govern: in libjinfer, which carries
 * both faces on one runtime, those are the knobs.
 */
final class Stores {
    private static volatile ModelStore store;

    static ModelStore store() {
        ModelStore s = store;
        if (s == null) store = s = create();
        return s;
    }

    private static ModelStore create() {
        String root = root();
        boolean offline = flag("toknroll.offline");
        if (root == null && !offline) return ModelStore.standard();
        Path at = root == null ? ModelStore.standard().root() : Path.of(root);
        return offline
                ? ModelStore.of(at)
                : ModelStore.of(at, new HuggingFaceSource(), new ModelScopeSource());
    }

    /**
     * The cache root as the GGUF and tokenizer.json caches inside this library read it, so all
     * agree: {@code -Dtoknroll.cache.root}, else {@code TOKNROLL_CACHE_ROOT}, a blank one unset.
     */
    private static String root() {
        String root = System.getProperty("toknroll.cache.root");
        if (root == null || root.isBlank()) root = System.getenv("TOKNROLL_CACHE_ROOT");
        return root == null || root.isBlank() ? null : root;
    }

    /** What the loaders' own fetches honour: either library's spelling of offline. */
    static boolean offline() {
        return flag("toknroll.offline") || flag("jinfer.offline");
    }

    private static boolean flag(String property) {
        String value = System.getProperty(property);
        return value != null && !value.strip().equalsIgnoreCase("false");
    }

    private Stores() {}
}
