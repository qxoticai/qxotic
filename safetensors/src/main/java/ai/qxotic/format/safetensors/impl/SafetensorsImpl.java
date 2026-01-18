package ai.qxotic.format.safetensors.impl;

import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.TensorEntry;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;

final class SafetensorsImpl implements Safetensors {

    private final long dataOffset;
    private final Map<String, String> metadata;
    private final Map<String, TensorEntry> tensors;

    SafetensorsImpl(
            long tensorDataOffset, Map<String, String> metadata, Map<String, TensorEntry> tensors) {
        this.dataOffset = tensorDataOffset;
        this.metadata = Collections.unmodifiableMap(metadata);
        this.tensors = Collections.unmodifiableMap(tensors);
    }

    @Override
    public long getTensorDataOffset() {
        return dataOffset;
    }

    @Override
    public int getAlignment() {
        String alignment = this.metadata.get(ImplAccessor.alignmentKey());
        if (alignment == null) {
            return ImplAccessor.defaultAlignment();
        }
        return Integer.parseInt(alignment);
    }

    @Override
    public Map<String, String> getMetadata() {
        return metadata;
    }

    @Override
    public Collection<TensorEntry> getTensors() {
        return tensors.values();
    }

    @Override
    public TensorEntry getTensor(String tensorName) {
        return tensors.get(tensorName);
    }
}
