package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.hub.ModelStore;
import java.nio.file.Path;
import java.util.Locale;
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
 */
public final class TestModels {

    private TestModels() {}

    /** The cached path for {@code ref}, or the test aborts - with the fix in the message. */
    public static Path require(String ref) {
        requirePinned(ref);
        return find(ref)
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
    public static java.util.Optional<Path> find(String ref) {
        requirePinned(ref);
        return ModelStore.find(ref);
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
