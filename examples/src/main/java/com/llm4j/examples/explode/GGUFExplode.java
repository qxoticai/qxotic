import ai.qxotic.model.llama.Timer;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public class GGUFExplode {

    private static final String METADATA_JSON = "metadata.json";

    public static String toJSON(GGUF gguf, boolean pretty) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        for (String key : gguf.getMetadataKeys()) {
            MetadataValueType valueType = gguf.getType(key);
            if (valueType == MetadataValueType.ARRAY) {
                MetadataValueType componentType = gguf.getComponentType(key);
                Object value = switch (componentType) {
                    case UINT8, INT8 -> {
                        byte[] array = gguf.getValue(byte[].class, key);
                        yield IntStream.range(0, array.length).mapToObj(i -> array[i]).toList();
                    }
                    case UINT16, INT16 -> {
                        short[] array = gguf.getValue(short[].class, key);
                        yield IntStream.range(0, array.length).mapToObj(i -> array[i]).toList();
                    }
                    case UINT32, INT32 -> {
                        int[] array = gguf.getValue(int[].class, key);
                        yield Arrays.stream(array).boxed().toList();
                    }
                    case FLOAT32 -> {
                        float[] array = gguf.getValue(float[].class, key);
                        yield IntStream.range(0, array.length).mapToObj(i -> array[i]).toList();
                    }
                    case BOOL -> {
                        boolean[] array = gguf.getValue(boolean[].class, key);
                        yield IntStream.range(0, array.length).mapToObj(i -> array[i]).toList();
                    }
                    case STRING -> {
                        String[] array = gguf.getValue(String[].class, key);
                        yield Arrays.asList(array);
                    }
                    case UINT64, INT64 -> {
                        long[] array = gguf.getValue(long[].class, key);
                        yield Arrays.stream(array).boxed().toList();
                    }
                    case FLOAT64 -> {
                        double[] array = gguf.getValue(double[].class, key);
                        yield Arrays.stream(array).boxed().toList();
                    }
                    case ARRAY -> throw new UnsupportedOperationException("nested arrays not supported");
                };
                metadata.put(key, value);
            } else {
                metadata.put(key, gguf.getValue(Object.class, key));
            }
        }

        Map<String, Object> metadataTypes = new LinkedHashMap<>();
        for (String key : gguf.getMetadataKeys()) {
            MetadataValueType valueType = gguf.getType(key);
            metadataTypes.put(key, valueType.toString());
            if (valueType == MetadataValueType.ARRAY) {
                MetadataValueType componentType = gguf.getComponentType(key);
                metadataTypes.put(key + "[*]", componentType.toString());
            }
        }

        List<Map<String, Object>> tensors = new ArrayList<>();
        for (TensorInfo tensor : gguf.getTensors()) {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("name", tensor.name());
            value.put("shape", LongStream.of(tensor.shape()).boxed().toList());
            value.put("ggml_type", tensor.ggmlType().toString());
            value.put("offset", tensor.offset());
            tensors.add(value);
        }

        Map<String, Object> root = new LinkedHashMap<>();
        root.put("version", gguf.getVersion());
        root.put("metadata_types", metadataTypes);
        root.put("metadata", metadata);
        root.put("tensors", tensors);

        return JSON.print(root, pretty);
    }

    public static GGUF fromJSON(String json) {
        Map<String, Object> root = (Map<String, Object>) JSON.parse(json);
        Map<String, Object> metadata = (Map<String, Object>) root.get("metadata");
        Map<String, String> metadataTypes = (Map<String, String>) root.get("metadata_types");

        Builder builder = Builder.newBuilder();
        int version = ((Number) root.get("version")).intValue();
        builder.setVersion(version);

        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            String key = entry.getKey();
            MetadataValueType valueType = MetadataValueType.valueOf(metadataTypes.get(key));
            switch (valueType) {
                case UINT8 -> builder.putUnsignedByte(key, ((Number) entry.getValue()).byteValue());
                case INT8 -> builder.putByte(key, ((Number) entry.getValue()).byteValue());
                case UINT16 -> builder.putUnsignedShort(key, ((Number) entry.getValue()).shortValue());
                case INT16 -> builder.putShort(key, ((Number) entry.getValue()).shortValue());
                case UINT32 -> builder.putUnsignedInteger(key, ((Number) entry.getValue()).intValue());
                case INT32 -> builder.putInteger(key, ((Number) entry.getValue()).intValue());
                case FLOAT32 -> builder.putFloat(key, ((Number) entry.getValue()).floatValue());
                case BOOL -> builder.putBoolean(key, (boolean) entry.getValue());
                case STRING -> builder.putString(key, (String) entry.getValue());
                case UINT64 -> builder.putUnsignedLong(key, ((Number) entry.getValue()).longValue());
                case INT64 -> builder.putLong(key, ((Number) entry.getValue()).longValue());
                case FLOAT64 -> builder.putDouble(key, ((Number) entry.getValue()).doubleValue());
                case ARRAY -> {
                    MetadataValueType componentType = MetadataValueType.valueOf(metadataTypes.get(key + "[*]"));
                    switch (componentType) {
                        case UINT8, INT8 -> {
                            List<Number> list = (List<Number>) entry.getValue();
                            byte[] array = new byte[list.size()];
                            for (int i = 0; i < list.size(); i++) {
                                byte byteNumber;
                                Number number = list.get(i);
                                if (number instanceof BigInteger bigIntegerNumber) {
                                    byteNumber = bigIntegerNumber.byteValueExact();
                                } else if (number instanceof Long longNumber) {
                                    byteNumber = toByteExact(longNumber);
                                } else if (number instanceof Integer integerNumber) {
                                    byteNumber = toByteExact(integerNumber);
                                } else if (number instanceof Short shortNumber) {
                                    byteNumber = toByteExact(shortNumber);
                                } else {
                                    byteNumber = number.byteValue();
                                }
                                array[i] = byteNumber;
                            }
                            if (componentType == MetadataValueType.UINT8) {
                                builder.putArrayOfUnsignedByte(key, array);
                            } else {
                                builder.putArrayOfByte(key, array);
                            }
                        }
                        case UINT16, INT16 -> {
                            List<Number> list = (List<Number>) entry.getValue();
                            short[] array = new short[list.size()];
                            for (int i = 0; i < list.size(); i++) {
                                short shortNumber;
                                Number number = list.get(i);
                                if (number instanceof BigInteger bigIntegerNumber) {
                                    shortNumber = bigIntegerNumber.shortValueExact();
                                } else if (number instanceof Long longNumber) {
                                    shortNumber = toShortExact(longNumber);
                                } else if (number instanceof Integer integerNumber) {
                                    shortNumber = toShortExact(integerNumber);
                                } else {
                                    shortNumber = number.shortValue();
                                }
                                array[i] = shortNumber;
                            }
                            if (componentType == MetadataValueType.UINT16) {
                                builder.putArrayOfUnsignedShort(key, array);
                            } else {
                                builder.putArrayOfShort(key, array);
                            }
                        }
                        case UINT32, INT32 -> {
                            List<Number> list = (List<Number>) entry.getValue();
                            assert list.stream().allMatch(Objects::nonNull);
                            int[] array = new int[list.size()];
                            for (int i = 0; i < list.size(); i++) {
                                Number number = list.get(i);
                                int intNumber;
                                if (number instanceof BigInteger bigIntegerNumber) {
                                    intNumber = bigIntegerNumber.intValueExact();
                                } else if (number instanceof Long longNumber) {
                                    intNumber = Math.toIntExact(longNumber);
                                } else {
                                    intNumber = number.intValue();
                                }
                                array[i] = intNumber;
                            }
                            if (componentType == MetadataValueType.UINT32) {
                                builder.putArrayOfUnsignedInteger(key, array);
                            } else {
                                builder.putArrayOfInteger(key, array);
                            }
                        }
                        case FLOAT32 -> {
                            List<Number> list = (List<Number>) entry.getValue();
                            assert list.stream().allMatch(Objects::nonNull);
                            float[] array = new float[list.size()];
                            for (int i = 0; i < list.size(); i++) {
                                Number number = list.get(i);
                                assert number instanceof Float || number instanceof Double || number instanceof BigDecimal;
                                array[i] = number.floatValue();
                            }
                            builder.putArrayOfFloat(key, array);
                        }
                        case BOOL -> {
                            List<Boolean> list = (List<Boolean>) entry.getValue();
                            boolean[] array = new boolean[list.size()];
                            for (int i = 0; i < list.size(); i++) {
                                array[i] = list.get(i);
                            }
                            builder.putArrayOfBoolean(key, array);
                        }
                        case STRING -> {
                            List<String> list = (List<String>) entry.getValue();
                            assert list.stream().allMatch(Objects::nonNull);
                            String[] array = list.toArray(String[]::new);
                            builder.putArrayOfString(key, array);
                        }
                        case UINT64, INT64 -> {
                            List<Number> list = (List<Number>) entry.getValue();
                            assert list.stream().allMatch(Objects::nonNull);
                            long[] array = list.stream().mapToLong(Number::longValue).toArray();
                            if (componentType == MetadataValueType.UINT64) {
                                builder.putArrayOfUnsignedLong(key, array);
                            } else {
                                builder.putArrayOfLong(key, array);
                            }
                        }
                        case FLOAT64 -> {
                            List<Number> list = (List<Number>) entry.getValue();
                            double[] array = list.stream().peek(number -> {
                                assert number instanceof Float || number instanceof Double || number instanceof BigDecimal;
                            }).mapToDouble(Number::doubleValue).toArray();
                            builder.putArrayOfDouble(key, array);
                        }
                        case ARRAY -> throw new UnsupportedOperationException("nested arrays");
                    }

                }
            }
        }

        Object tensorsValue = root.get("tensors");
        // Support tensors being a list or a map.
        Collection<Map<String, Object>> tensors = tensorsValue instanceof List
                ? (List<Map<String, Object>>) tensorsValue
                : ((Map<String, Map<String, Object>>) tensorsValue).values();

        for (Map<String, Object> tensor : tensors) {
            String name = (String) tensor.get("name");
            long[] shape = ((List<Number>) tensor.get("shape")).stream().mapToLong(Number::longValue).toArray();
            GGMLType ggmlType = GGMLType.valueOf((String) tensor.get("ggml_type"));
            long offset = ((Number) tensor.get("offset")).longValue();
            builder.putTensor(TensorInfo.create(name, shape, ggmlType, offset));
        }

        return builder.build();
    }

    private static short toShortExact(long longValue) {
        short shortValue = (short) longValue;
        if (shortValue != longValue) {
            throw new ArithmeticException("short overflow");
        }
        return shortValue;
    }

    private static byte toByteExact(long longValue) {
        byte byteValue = (byte) longValue;
        if (byteValue != longValue) {
            throw new ArithmeticException("byte overflow");
        }
        return byteValue;
    }

    public static void main(String[] args) throws IOException {

        Path modelPath = Path.of("/home/mukel/Desktop/playground/models/hf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-BF16.gguf.assembled");
        Path explodedDirectory = Path.of("/home/mukel/Desktop/playground/models/hf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-BF16.gguf.exploded");

        try (Timer timer = Timer.log(TimeUnit.SECONDS, "Total time")) {
            //explode(modelPath, explodedDirectory);
            assemble(explodedDirectory, modelPath);
        }
    }

    private static void explode(Path sourceModelPath, Path destExplodedDirectory) throws IOException {
        if (!Files.isRegularFile(sourceModelPath)) {
            throw new NoSuchFileException(sourceModelPath.toString());
        }

        Files.createDirectories(destExplodedDirectory);

        Path tensorsDirectory = destExplodedDirectory.resolve("tensors");
        Files.createDirectories(tensorsDirectory);

        Path metadataPath = destExplodedDirectory.resolve(METADATA_JSON);
        GGUF gguf;
        try (Timer timer = Timer.log("Reading GGUF metadata")) {
            gguf = GGUF.read(sourceModelPath);
        }

        String jsonMetadata;
        try (Timer timer = Timer.log("Converting GGUF metadata to JSON")) {
            jsonMetadata = toJSON(gguf, true);
        }

        try (Timer timer = Timer.log("Writing " + METADATA_JSON)) {
            Files.writeString(metadataPath, jsonMetadata, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
        }

        int tensorsWritten = 0;
        int totalTensors = gguf.getTensors().size();
        try (FileChannel inputChannel = FileChannel.open(sourceModelPath, StandardOpenOption.READ)) {
            for (TensorInfo tensorInfo : gguf.getTensors()) {
                ++tensorsWritten;
                try (Timer timer = Timer.log("Writing (" + tensorsWritten + "/" + totalTensors + ") tensor " + tensorInfo)) {
                    GGMLType ggmlType = tensorInfo.ggmlType();
                    long sizeInBytes = ggmlType.byteSizeForShape(tensorInfo.shape());
                    Path outputTensorPath = tensorsDirectory.resolve(tensorInfo.name());
                    try (FileChannel outputChannel = FileChannel.open(outputTensorPath, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
                         Arena arena = Arena.ofConfined()) {

                        MemorySegment inputSegment = inputChannel.map(FileChannel.MapMode.READ_ONLY, gguf.getTensorDataOffset() + tensorInfo.offset(), sizeInBytes, arena);
                        MemorySegment outputSegment = outputChannel.map(FileChannel.MapMode.READ_WRITE, 0, sizeInBytes, arena);
                        MemorySegment.copy(inputSegment, 0, outputSegment, 0, sizeInBytes);
                    }
                }
            }
        }
    }

    private static void assemble(Path sourceExplodedDirectory, Path destModelPath) throws IOException {

        String jsonMetadata;
        try (Timer timer = Timer.log("Loading " + METADATA_JSON)) {
            jsonMetadata = Files.readString(sourceExplodedDirectory.resolve(METADATA_JSON));
        }
        GGUF gguf;
        try (Timer timer = Timer.log("Converting JSON metadata to GGUF")) {
            gguf = fromJSON(jsonMetadata);
        }

        try (FileChannel outputChannel = FileChannel.open(destModelPath, StandardOpenOption.READ, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            try (Timer timer = Timer.log("Writing GGUF metadata")) {
                GGUF.write(gguf, outputChannel);
            }

            int tensorsWritten = 0;
            int totalTensors = gguf.getTensors().size();
            for (TensorInfo tensorInfo : gguf.getTensors()) {
                ++tensorsWritten;
                try (Timer timer = Timer.log("Writing (" + tensorsWritten + "/" + totalTensors + ") tensor " + tensorInfo)) {
                    GGMLType ggmlType = tensorInfo.ggmlType();
                    long sizeInBytes = ggmlType.byteSizeForShape(tensorInfo.shape());
                    Path inputTensorPath = sourceExplodedDirectory.resolve("tensors").resolve(tensorInfo.name());

                    long tensorFileSize = Files.size(inputTensorPath);
                    if (sizeInBytes != tensorFileSize) {
                        throw new IllegalArgumentException(inputTensorPath + " size (" + tensorFileSize + ") doesn't match metadata, expected size " + sizeInBytes);
                    }

                    try (FileChannel inputChannel = FileChannel.open(inputTensorPath, StandardOpenOption.READ);
                         Arena arena = Arena.ofConfined()) {
                        MemorySegment inputSegment = inputChannel.map(FileChannel.MapMode.READ_ONLY, 0, sizeInBytes, arena);
                        MemorySegment outputSegment = outputChannel.map(FileChannel.MapMode.READ_WRITE, gguf.getTensorDataOffset() + tensorInfo.offset(), sizeInBytes, arena);
                        MemorySegment.copy(inputSegment, 0, outputSegment, 0, sizeInBytes);
                    }
                }
            }

            outputChannel.force(false);
        }
    }
}
