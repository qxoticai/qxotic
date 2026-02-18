package ai.qxotic.format.safetensors.impl;

import ai.qxotic.format.safetensors.Builder;
import ai.qxotic.format.safetensors.DType;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.TensorEntry;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

final class BuilderImpl implements Builder {

    private Map<String, String> metadata = new LinkedHashMap<>();
    private Map<String, TensorEntry> tensorEntries = new LinkedHashMap<>();

    BuilderImpl() {}

    static BuilderImpl fromExisting(Safetensors safetensors) {
        return new BuilderImpl()
                .setMetadata(safetensors.getMetadata())
                .setTensors(BuilderImpl.fromCollection(safetensors.getTensors()));
    }

    private static Map<String, TensorEntry> fromCollection(Collection<TensorEntry> tensors) {
        Map<String, TensorEntry> map = new LinkedHashMap<>();
        for (TensorEntry tensor : tensors) {
            TensorEntry previous = map.put(tensor.name(), tensor);
            if (previous != null) {
                throw new IllegalArgumentException("duplicated tensor name: " + tensor.name());
            }
        }
        return map;
    }

    @Override
    public BuilderImpl setMetadata(Map<String, String> newMetadata) {
        // Must preserve insertion order.
        this.metadata = new LinkedHashMap<>(newMetadata);
        return this;
    }

    @Override
    public Safetensors build(boolean recomputeTensorOffsets) {
        Map<String, TensorEntry> freshTensorEntries =
                recomputeTensorOffsets ? computeTensorOffsets() : this.tensorEntries;
        long freshTensorDataOffset =
                WriterImpl.computeTensorDataOffset(
                        metadata, freshTensorEntries.values(), getAlignment());
        return new SafetensorsImpl(freshTensorDataOffset, this.metadata, freshTensorEntries);
    }

    @Override
    public BuilderImpl clone() {
        return new BuilderImpl()
                .setMetadata(new LinkedHashMap<>(this.metadata))
                .setTensors(new LinkedHashMap<>(this.tensorEntries));
    }

    private BuilderImpl setTensors(Map<String, TensorEntry> newTensorEntries) {
        if (!(newTensorEntries instanceof LinkedHashMap)) {
            throw new IllegalArgumentException("tensor map must preserve insertion order");
        }
        boolean namesMatch =
                newTensorEntries.entrySet().stream()
                        .allMatch(e -> e.getKey().equals(e.getValue().name()));
        if (!namesMatch) {
            throw new IllegalArgumentException("tensor map keys must match tensor names");
        }
        this.tensorEntries = newTensorEntries;
        return this;
    }

    @Override
    public BuilderImpl putTensor(TensorEntry tensorEntry) {
        Objects.requireNonNull(tensorEntry);
        this.tensorEntries.put(tensorEntry.name(), tensorEntry);
        return this;
    }

    @Override
    public BuilderImpl removeTensor(String tensorName) {
        this.tensorEntries.remove(tensorName);
        return this;
    }

    @Override
    public boolean containsTensor(String tensorName) {
        return this.tensorEntries.containsKey(tensorName);
    }

    @Override
    public TensorEntry getTensor(String tensorName) {
        return this.tensorEntries.get(tensorName);
    }

    @Override
    public Collection<TensorEntry> getTensors() {
        return Collections.unmodifiableCollection(this.tensorEntries.values());
    }

    @Override
    public Map<String, String> getMetadata() {
        return Collections.unmodifiableMap(this.metadata);
    }

    @Override
    public Builder putMetadataKey(String key, String value) {
        this.metadata.put(Objects.requireNonNull(key), Objects.requireNonNull(value));
        return this;
    }

    @Override
    public String getMetadataValue(String key) {
        return this.metadata.get(key);
    }

    @Override
    public Builder removeMetadataKey(String key) {
        this.metadata.remove(key);
        return this;
    }

    @Override
    public Set<String> getMetadataKeys() {
        return Collections.unmodifiableSet(this.metadata.keySet());
    }

    private Map<String, TensorEntry> computeTensorOffsets() {
        long tensorOffset = 0;
        Map<String, TensorEntry> reindexed = new LinkedHashMap<>();
        for (Map.Entry<String, TensorEntry> entry : tensorEntries.entrySet()) {
            // Add padding, tensor start must be aligned.
            tensorOffset += WriterImpl.padding(tensorOffset, getAlignment());
            String name = entry.getKey();
            TensorEntry tensorEntry = entry.getValue();
            DType dType = tensorEntry.dtype();
            reindexed.put(name, TensorEntry.create(name, dType, tensorEntry.shape(), tensorOffset));
            long byteSize = dType.byteSizeForShape(tensorEntry.shape());
            tensorOffset += byteSize;
        }
        return reindexed;
    }
}
