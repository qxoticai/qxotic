package ai.qxotic.format.safetensors;

import ai.qxotic.format.safetensors.impl.ImplAccessor;
import java.util.Collection;
import java.util.Map;
import java.util.Set;

/**
 * Builder interface for the Safetensors format.
 *
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF format
 *     specification</a>
 */
public interface Builder extends Cloneable {
    /** Creates a new {@link Builder} from an existing GGUF instance. */
    static Builder newBuilder(Safetensors safetensors) {
        return ImplAccessor.newBuilder(safetensors);
    }

    /** Creates a new empty {@link Builder}. */
    static Builder newBuilder() {
        return ImplAccessor.newBuilder();
    }

    /** Builds a GGUF instance with automatic tensor offset computation. */
    default Safetensors build() {
        return build(true);
    }

    /**
     * Builds a GGUF instance.
     *
     * @param recomputeTensorOffsets if true, tensor offsets will be automatically re-computed,
     *     packed in the same order and respecting the alignment
     */
    Safetensors build(boolean recomputeTensorOffsets);

    /** Creates and returns a copy of this object. */
    Builder clone();

    /**
     * Sets the alignment value for tensor data.
     *
     * @throws IllegalArgumentException if alignment is not a power of 2
     */
    default Builder setAlignment(int newAlignment) {
        if (newAlignment < 0 || Integer.bitCount(newAlignment) != 1) {
            throw new IllegalArgumentException(
                    "alignment must be a power of 2 but was " + newAlignment);
        }
        if (newAlignment == ImplAccessor.defaultAlignment()) {
            removeMetadataKey(ImplAccessor.alignmentKey());
        } else {
            putMetadataKey(ImplAccessor.alignmentKey(), Integer.toString(newAlignment));
        }
        return this;
    }

    /** Gets the current alignment value or the default if not set. */
    default int getAlignment() {
        String alignment = getMetadataValue(ImplAccessor.alignmentKey());
        if (alignment != null) {
            return Integer.parseInt(alignment);
        }
        return ImplAccessor.defaultAlignment();
    }

    /** Adds or updates a tensor. */
    Builder putTensor(TensorEntry tensorEntry);

    /** Update __metadata__. */
    Builder setMetadata(Map<String, String> metadata);

    /** Removes a tensor by name. */
    Builder removeTensor(String tensorName);

    /** Checks if a tensor exists by name. */
    boolean containsTensor(String tensorName);

    /** Gets tensor information by name. */
    TensorEntry getTensor(String tensorName);

    /**
     * Gets all tensors, order is preserved.
     *
     * @return the collection of tensor information
     */
    Collection<TensorEntry> getTensors();

    /** Gets all metadata keys, order is preserved. */
    Map<String, String> getMetadata();

    Builder putMetadataKey(String key, String value);

    String getMetadataValue(String key);

    Builder removeMetadataKey(String key);

    /**
     * Returns a set of all metadata keys present in the GGUF metadata, order is preserved.
     *
     * @return a set containing all metadata keys
     */
    Set<String> getMetadataKeys();
}
