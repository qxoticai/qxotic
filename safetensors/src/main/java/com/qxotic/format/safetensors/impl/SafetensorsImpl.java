package com.qxotic.format.safetensors.impl;

import com.qxotic.format.safetensors.Safetensors;
import com.qxotic.format.safetensors.TensorEntry;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

final class SafetensorsImpl implements Safetensors {

    private final long tensorDataOffset;
    private final Map<String, String> metadata;
    private final Map<String, TensorEntry> tensors;

    SafetensorsImpl(
            long tensorDataOffset, Map<String, String> metadata, Map<String, TensorEntry> tensors) {
        this.tensorDataOffset = tensorDataOffset;
        // copies: a builder keeps mutating its own maps after build()
        this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        this.tensors = Collections.unmodifiableMap(new LinkedHashMap<>(tensors));
    }

    @Override
    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    @Override
    public int getAlignment() {
        return AlignmentSupport.parseMetadataAlignment(this.metadata);
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

    @Override
    public String toString() {
        return ImplAccessor.toString(this, true, true);
    }
}
