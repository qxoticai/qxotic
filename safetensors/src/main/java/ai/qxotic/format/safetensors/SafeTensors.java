package ai.qxotic.format.safetensors;

import ai.qxotic.format.safetensors.impl.JSON;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

public final class SafeTensors {

    private static final String METADATA_KEY = "__metadata__";

    public static Map<String, HFTensorEntry> loadFromModelRoot(Path modelRootDirectory) throws IOException {
        Objects.requireNonNull(modelRootDirectory, "modelRootDirectory");
        var indexPath = modelRootDirectory.resolve("model.safetensors.index.json");
        Map<String, Object> indexJson = (Map<String, Object>) JSON.parse(Files.readString(indexPath));
        Map<String, Object> weightMap = (Map<String, Object>) indexJson.get("weight_map");
        if (weightMap == null) {
            throw new IOException("Missing 'weight_map' in index file: " + indexPath);
        }
        Map<String, HFTensorEntry> allTensorEntries = new HashMap<>();
        for (Object value : Set.copyOf(weightMap.values())) {
            String containingFile = (String) value;
            var tensorEntries = loadTensorEntries(modelRootDirectory.resolve(containingFile));
            if (hasOverlappingKeys(allTensorEntries.keySet(), tensorEntries.keySet())) {
                throw new IOException("Duplicate tensor names found across safetensors files in: " + modelRootDirectory);
            }
            allTensorEntries.putAll(tensorEntries);
        }
        return allTensorEntries;
    }

    public static Map<String, HFTensorEntry> loadTensorEntries(Path filePath) throws IOException {
        Objects.requireNonNull(filePath, "filePath");
        Map<String, HFTensorEntry> tensorEntries = new HashMap<>();
        try (FileChannel fileChannel = FileChannel.open(filePath, StandardOpenOption.READ)) {
            ByteBuffer sizeBytes = ByteBuffer.allocate(8);
            int bytesRead = fileChannel.read(sizeBytes);
            if (bytesRead != 8) {
                throw new IOException("Invalid safetensors file: expected 8 bytes for header size, got " + bytesRead + " in " + filePath);
            }
            sizeBytes.clear().order(ByteOrder.LITTLE_ENDIAN);
            long headerSizeLong = sizeBytes.getLong(0);
            if (headerSizeLong <= 0 || headerSizeLong > Integer.MAX_VALUE) {
                throw new IOException("Invalid header size: " + headerSizeLong + " in " + filePath);
            }
            int headerSize = (int) headerSizeLong;

            byte[] headerBytes = new byte[headerSize];
            bytesRead = fileChannel.read(ByteBuffer.wrap(headerBytes));
            if (bytesRead != headerSize) {
                throw new IOException("Failed to read header: expected " + headerSize + " bytes, got " + bytesRead + " in " + filePath);
            }

            Map<String, Object> json = (Map<String, Object>) JSON.parse(new String(headerBytes));

            for (Map.Entry<String, Object> entry : json.entrySet()) {
                String tensorName = entry.getKey();
                if (METADATA_KEY.equals(tensorName)) {
                    continue;
                }
                Map<String, Object> tensorEntry = (Map<String, Object>) entry.getValue();
                String dtypeString = (String) tensorEntry.get("dtype");
                if (dtypeString == null) {
                    throw new IOException("Missing dtype for tensor: " + tensorName + " in " + filePath);
                }
                DType dtype;
                try {
                    dtype = DType.valueOf(dtypeString);
                } catch (IllegalArgumentException e) {
                    throw new IOException("Unknown dtype '" + dtypeString + "' for tensor: " + tensorName + " in " + filePath, e);
                }
                List<Number> shapeList = (List<Number>) tensorEntry.get("shape");
                if (shapeList == null) {
                    throw new IOException("Missing shape for tensor: " + tensorName + " in " + filePath);
                }

                long[] shape = shapeList.stream().mapToLong(Number::longValue).toArray();

                List<Object> offsets = (List<Object>) tensorEntry.get("data_offsets");
                if (offsets == null || offsets.size() != 2) {
                    throw new IOException("Invalid data_offsets for tensor: " + tensorName + " in " + filePath);
                }
                long begin = ((Number) offsets.get(0)).longValue();
                long end = ((Number) offsets.get(1)).longValue();
                if (begin < 0 || end < 0) {
                    throw new IOException("Invalid offsets [" + begin + ", " + end + "] for tensor: " + tensorName + " in " + filePath);
                }
                if (begin > end) {
                    throw new IOException("Invalid offsets: begin (" + begin + ") > end (" + end + ") for tensor: " + tensorName + " in " + filePath);
                }
                long bufferSize = end - begin;
                long expectedSize = numberOfElements(shape) * (long) dtype.size();
                if (expectedSize != bufferSize) {
                    throw new IOException("Size mismatch for tensor " + tensorName + ": expected " + expectedSize + " bytes, got " + bufferSize + " in " + filePath);
                }
                tensorEntries.put(tensorName, new HFTensorEntry(tensorName, dtype, shape, Long.BYTES + headerSize + begin, bufferSize));
            }
        }

        return tensorEntries;
    }

    private static long numberOfElements(long[] shape) {
        return Arrays.stream(shape).reduce(1L, Math::multiplyExact);
    }

    private static boolean hasOverlappingKeys(Set<String> set1, Set<String> set2) {
        Set<String> smaller = set1.size() <= set2.size() ? set1 : set2;
        Set<String> larger = set1.size() > set2.size() ? set1 : set2;
        for (String key : smaller) {
            if (larger.contains(key)) {
                return true;
            }
        }
        return false;
    }
}
