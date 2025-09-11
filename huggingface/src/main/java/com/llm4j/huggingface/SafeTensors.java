package com.llm4j.huggingface;

import com.llm4j.api.BaseType;
import com.llm4j.huggingface.impl.JSON;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class SafeTensors {

    public static Map<String, HFTensorEntry> loadFromModelRoot(Path modelRootDirectory) throws IOException {
        var indexPath = modelRootDirectory.resolve("model.safetensors.index.json");
        Map<String, Object> indexJson = (Map<String, Object>) JSON.parse(Files.readString(indexPath));
        Map<String, Object> weightMap = (Map<String, Object>) indexJson.get("weight_map");
        Map<String, HFTensorEntry> allTensorEntries = new HashMap<>();
        for (Object value : Set.copyOf(weightMap.values())) {
            String containingFile = (String) value;
            var tensorEntries = loadTensorEntries(modelRootDirectory.resolve(containingFile));
            assert !hasOverlappingKeys(allTensorEntries.keySet(), tensorEntries.keySet());
            allTensorEntries.putAll(tensorEntries);
        }
        return allTensorEntries;
    }

    public static Map<String, HFTensorEntry> loadTensorEntries(Path filePath) throws IOException {
        Map<String, HFTensorEntry> tensorEntries = new HashMap<>();
        try (FileChannel fileChannel = FileChannel.open(filePath, StandardOpenOption.READ)) {
            ByteBuffer sizeBytes = ByteBuffer.allocate(8);
            int bytesRead = fileChannel.read(sizeBytes);
            assert bytesRead == 8;
            sizeBytes.clear().order(ByteOrder.LITTLE_ENDIAN);
            int headerSize = Math.toIntExact(sizeBytes.getLong(0));
            assert headerSize > 0;

            byte[] headerBytes = new byte[headerSize];
            bytesRead = fileChannel.read(ByteBuffer.wrap(headerBytes));
            assert bytesRead == headerSize;

            Map<String, Object> json = (Map<String, Object>) JSON.parse(new String(headerBytes));

            for (Map.Entry<String, Object> entry : json.entrySet()) {
                String tensorName = entry.getKey();
                if ("__metadata__".equals(tensorName)) {
                    continue;
                }
                Map<String, Object> tensorEntry = (Map<String, Object>) entry.getValue();
                String dtypeString = (String) tensorEntry.get("dtype");
                DType dtype = DType.valueOf(dtypeString);
                List<Number> shapeList = (List<Number>) tensorEntry.get("shape");

                long[] shape = shapeList.stream().mapToLong(Number::longValue).toArray();

                List<Object> offsets = (List<Object>) tensorEntry.get("data_offsets");
                long begin = ((Number) offsets.get(0)).longValue();
                assert begin >= 0;
                long end = ((Number) offsets.get(1)).longValue();
                assert end >= 0;
                assert begin <= end;
                long bufferSize = end - begin;
                assert BaseType.numberOfElements(shape) * (long) dtype.size() == bufferSize;
                tensorEntries.put(tensorName, new HFTensorEntry(tensorName, dtype, shape, Long.BYTES + headerSize + begin, bufferSize));
            }
        }

        return tensorEntries;
    }


    private static boolean hasOverlappingKeys(Set<String> map0, Set<String> map1) {
        for (String key1 : map1) {
            if (map0.contains(key1)) {
                return true;
            }
        }
        return false;
    }
}
