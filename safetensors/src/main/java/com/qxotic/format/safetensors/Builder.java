package com.qxotic.format.safetensors;

import com.qxotic.format.safetensors.impl.ImplAccessor;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Builds {@link Safetensors} instances containing metadata and tensor entries.
 *
 * @see <a href="https://github.com/huggingface/safetensors">Safetensors specification</a>
 */
public interface Builder extends Cloneable {
    /** Returns a builder pre-populated with a copy of {@code safetensors}. */
    static Builder newBuilder(Safetensors safetensors) {
        Objects.requireNonNull(safetensors, "safetensors");
        return ImplAccessor.newBuilder(safetensors);
    }

    /** Returns a new empty builder. */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /** Builds with tensor offsets recomputed, packed back to back; same as {@code build(true)}. */
    default Safetensors build() {
        return build(true);
    }

    /**
     * Builds a Safetensors instance.
     *
     * @param recomputeTensorOffsets if true, tensor offsets will be automatically re-computed,
     *     packed back to back in the same order
     * @throws ArithmeticException if a tensor end offset cannot be represented as a {@code long}
     */
    Safetensors build(boolean recomputeTensorOffsets);

    /** Creates a deep copy of this builder. */
    Builder clone();

    /**
     * Aligns the start of the tensor data to a multiple of {@code newAlignment} (a power of 2), by
     * padding the header with spaces as the format allows. The tensors follow back to back - the
     * format forbids holes - so each starts aligned only as far as the sizes before it allow:
     * ordering them by element size, largest first, keeps every one naturally aligned (what
     * upstream's writer does).
     */
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

    /** Returns the current alignment (defaults to 1 if __alignment__ is absent). */
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

    /** Returns tensor information by name as an Optional. */
    default Optional<TensorEntry> findTensor(String tensorName) {
        Objects.requireNonNull(tensorName, "tensorName");
        return Optional.ofNullable(getTensor(tensorName));
    }

    /** Returns tensor information by name, or throws if absent. */
    default TensorEntry requireTensor(String tensorName) {
        Objects.requireNonNull(tensorName, "tensorName");
        return findTensor(tensorName)
                .orElseThrow(() -> new IllegalArgumentException("tensor not found: " + tensorName));
    }

    /** Gets all tensors (unmodifiable, order preserved). */
    Collection<TensorEntry> getTensors();

    /** Returns all metadata as an unmodifiable map (insertion order preserved). */
    Map<String, String> getMetadata();

    /** Adds or replaces one metadata key/value pair. */
    Builder putMetadataKey(String key, String value);

    /** Returns one metadata value, or null if absent. */
    String getMetadataValue(String key);

    /** Returns one metadata value as an Optional. */
    default Optional<String> findMetadataValue(String key) {
        Objects.requireNonNull(key, "key");
        return Optional.ofNullable(getMetadataValue(key));
    }

    /** Removes one metadata key. */
    Builder removeMetadataKey(String key);

    /** Returns a set of all metadata keys (order preserved). */
    Set<String> getMetadataKeys();
}
