package ai.qxotic.format.safetensors;

import ai.qxotic.format.safetensors.impl.ImplAccessor;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Builder interface for Safetensors headers.
 *
 * <p>Builds {@link Safetensors} instances containing metadata and tensor entries.
 *
 * @see <a href="https://github.com/huggingface/safetensors">Safetensors specification</a>
 */
public interface Builder extends Cloneable {
    /** Creates a new {@link Builder} from an existing Safetensors instance. */
    static Builder newBuilder(Safetensors safetensors) {
        Objects.requireNonNull(safetensors, "safetensors");
        return ImplAccessor.newBuilder(safetensors);
    }

    /** Creates a new empty {@link Builder}. */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /** Builds a Safetensors instance with automatic tensor offset computation. */
    default Safetensors build() {
        return build(true);
    }

    /**
     * Builds a Safetensors instance.
     *
     * @param recomputeTensorOffsets if true, tensor offsets will be automatically re-computed,
     *     packed in the same order and respecting the alignment
     * @return immutable {@link Safetensors} view of the current builder content
     */
    Safetensors build(boolean recomputeTensorOffsets);

    /**
     * Creates a deep copy of this builder.
     *
     * @return cloned builder with independent metadata and tensor maps
     */
    Builder clone();

    /**
     * Sets the alignment value for tensor payload.
     *
     * @throws IllegalArgumentException if alignment is not a power of 2
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

    /**
     * Returns the current alignment.
     *
     * <p>If {@code __alignment__} is absent from metadata, returns the default value.
     *
     * @return alignment in bytes
     * @throws IllegalArgumentException if metadata contains an invalid alignment value
     */
    default int getAlignment() {
        String alignment = getMetadataValue(ImplAccessor.alignmentKey());
        if (alignment != null) {
            return ImplAccessor.parseAlignment(alignment);
        }
        return ImplAccessor.defaultAlignment();
    }

    /**
     * Adds or replaces a tensor entry by name.
     *
     * @param tensorEntry tensor descriptor to store
     * @return this builder
     */
    Builder putTensor(TensorEntry tensorEntry);

    /**
     * Replaces the complete {@code __metadata__} map.
     *
     * @param metadata new metadata map (string keys and values)
     * @return this builder
     */
    Builder setMetadata(Map<String, String> metadata);

    /**
     * Removes a tensor by name.
     *
     * @param tensorName tensor name
     * @return this builder
     */
    Builder removeTensor(String tensorName);

    /**
     * Checks whether a tensor exists.
     *
     * @param tensorName tensor name
     * @return true if present
     */
    boolean containsTensor(String tensorName);

    /**
     * Returns tensor information by name.
     *
     * @param tensorName tensor name
     * @return tensor entry, or null if absent
     */
    TensorEntry getTensor(String tensorName);

    /**
     * Gets all tensors, order is preserved.
     *
     * @return the collection of tensor information
     */
    Collection<TensorEntry> getTensors();

    /**
     * Returns all metadata as an unmodifiable map, preserving insertion order.
     *
     * @return metadata map
     */
    Map<String, String> getMetadata();

    /**
     * Adds or replaces one metadata key/value pair.
     *
     * @param key metadata key
     * @param value metadata value
     * @return this builder
     */
    Builder putMetadataKey(String key, String value);

    /**
     * Returns one metadata value.
     *
     * @param key metadata key
     * @return value, or null if absent
     */
    String getMetadataValue(String key);

    /**
     * Removes one metadata key.
     *
     * @param key metadata key
     * @return this builder
     */
    Builder removeMetadataKey(String key);

    /**
     * Returns a set of all metadata keys, order is preserved.
     *
     * @return a set containing all metadata keys
     */
    Set<String> getMetadataKeys();
}
