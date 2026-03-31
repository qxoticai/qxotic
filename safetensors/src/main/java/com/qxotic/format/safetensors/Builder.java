package com.qxotic.format.safetensors;

import com.qxotic.format.safetensors.impl.ImplAccessor;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Builds Safetensors instances containing metadata and tensor entries.
 *
 * @see <a href="https://github.com/huggingface/safetensors">Safetensors specification</a>
 */
public interface Builder extends Cloneable {
    static Builder newBuilder(Safetensors safetensors) {
        Objects.requireNonNull(safetensors, "safetensors");
        return ImplAccessor.newBuilder(safetensors);
    }

    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    default Safetensors build() {
        return build(true);
    }

    /**
     * Builds a Safetensors instance.
     *
     * @param recomputeTensorOffsets if true, tensor offsets will be automatically re-computed,
     *     packed in the same order and respecting the alignment
     */
    Safetensors build(boolean recomputeTensorOffsets);

    /** Creates a deep copy of this builder. */
    Builder clone();

    /** Sets the alignment value for tensor payload (must be a power of 2). */
    default Builder setAlignment(int newAlignment) {
        if (!ImplAccessor.isValidAlignment(newAlignment)) {
            throw new IllegalArgumentException(
                    "alignment must be a positive power of 2 but was " + newAlignment);
        }
        if (newAlignment == ImplAccessor.defaultAlignment()) {
            removeMetadataKey(ImplAccessor.alignmentKey());
        } else {
            putMetadataKey(ImplAccessor.alignmentKey(), Integer.toString(newAlignment));
        }
        return this;
    }

    /** Returns the current alignment (default value if __alignment__ is absent). */
    default int getAlignment() {
        String alignment = getMetadataValue(ImplAccessor.alignmentKey());
        if (alignment != null) {
            return ImplAccessor.parseAlignment(alignment);
        }
        return ImplAccessor.defaultAlignment();
    }

    /** Adds or replaces a tensor entry by name. */
    Builder putTensor(TensorEntry tensorEntry);

    /** Replaces the complete {@code __metadata__} map. */
    Builder setMetadata(Map<String, String> metadata);

    /** Removes a tensor by name. */
    Builder removeTensor(String tensorName);

    /** Checks whether a tensor exists. */
    boolean containsTensor(String tensorName);

    /** Returns tensor information by name, or null if absent. */
    TensorEntry getTensor(String tensorName);

    /** Gets all tensors (unmodifiable, order preserved). */
    Collection<TensorEntry> getTensors();

    /** Returns all metadata as an unmodifiable map (insertion order preserved). */
    Map<String, String> getMetadata();

    /** Adds or replaces one metadata key/value pair. */
    Builder putMetadataKey(String key, String value);

    /** Returns one metadata value, or null if absent. */
    String getMetadataValue(String key);

    /** Removes one metadata key. */
    Builder removeMetadataKey(String key);

    /** Returns a set of all metadata keys (order preserved). */
    Set<String> getMetadataKeys();
}
