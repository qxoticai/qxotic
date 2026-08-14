package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.hub.ModelStore;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Function;
import org.opentest4j.TestAbortedException;

/**
 * The one place tests acquire their checkpoints, by REF - the same string the CLI takes. The ref
 * ALWAYS pins what it wants: a model names its quant ({@code
 * hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0} - never the bare repo, whose default quant is a guess), a
 * companion (mmproj, mtp) names its exact file ({@code
 * hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf} - the hub keeps companions out of quant
 * matching on purpose). LOOKUP ONLY: a test never downloads; an absent model aborts the test with
 * the fix in the message. Resolution is {@link ModelStore#find}: jinfer's own cache ({@code
 * -Djinfer.models} / {@code JINFER_MODELS}) first, then the HuggingFace hub cache.
 *
 * <p>The per-model override: {@code -Djinfer.testModel.<last-ref-segment>=<path>} makes that file
 * serve the ref - e.g. {@code -Djinfer.testModel.LFM2.5-8B-A1B-Q8_0.gguf=/models/my-Q4_K_M.gguf}
 * runs every suite that requires the Q8_0 ref against your file instead (a quant-form ref keys on
 * its colon segment: {@code -Djinfer.testModel.stories15M_MOE:Q8_0=...}). The knob is derived, not
 * per-suite: nothing to invent, nothing to document per test. A stale override FAILS loudly - an
 * explicit pointer that resolves to nothing is a tester error, never a silent skip.
 */
public final class TestModels {

    private static final String OVERRIDE_PREFIX = "jinfer.testModel.";

    private TestModels() {}

    /** The cached path for {@code ref}, or the test aborts - with the fix in the message. */
    public static Path require(String ref) {
        return require(ref, ModelStore::find, System::getProperty);
    }

    static Path require(
            String ref, Function<String, Optional<Path>> store, Function<String, String> props) {
        return resolve(ref, store, props)
                .orElseThrow(
                        () ->
                                new TestAbortedException(
                                        "model not cached: "
                                                + ref
                                                + " - fetch it with scripts/download-models.sh"
                                                + " (adding a line to scripts/models.txt if it's"
                                                + " missing), with any HuggingFace client into"
                                                + " the hub cache, or point -Djinfer.models /"
                                                + " JINFER_MODELS at a cache that has it"));
    }

    /** The cached path for {@code ref}, or empty - the non-aborting form, for presence probes. */
    public static Optional<Path> find(String ref) {
        return resolve(ref);
    }

    private static Optional<Path> resolve(String ref) {
        return resolve(ref, ModelStore::find, System::getProperty);
    }

    // Package-private seam for TestModelsTest: the store and the property source are injected so
    // unit tests run in-memory, without mutating process-global state.
    static Optional<Path> resolve(
            String ref, Function<String, Optional<Path>> store, Function<String, String> props) {
        requirePinned(ref);
        String name = ref.substring(ref.lastIndexOf('/') + 1);
        String override = props.apply(OVERRIDE_PREFIX + name);
        if (override == null) {
            return store.apply(ref);
        }
        Path path = Path.of(override);
        if (!Files.exists(path)) {
            throw new IllegalArgumentException(
                    "-D" + OVERRIDE_PREFIX + name + "=" + override + " does not exist");
        }
        return Optional.of(path);
    }

    /** A test ref pins its quant or names an exact file - anything else is a guess. */
    private static void requirePinned(String ref) {
        String last = ref.substring(ref.lastIndexOf('/') + 1);
        boolean pinned = last.indexOf(':') >= 0 || last.toLowerCase(Locale.ROOT).endsWith(".gguf");
        if (!pinned) {
            throw new IllegalArgumentException(
                    "test refs always specify the quant: "
                            + ref
                            + " - did you mean "
                            + ref
                            + ":Q8_0? (a companion names its exact file, e.g."
                            + " .../mmproj-F32.gguf)");
        }
    }
}
