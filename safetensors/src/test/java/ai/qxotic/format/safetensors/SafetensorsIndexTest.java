package ai.qxotic.format.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class SafetensorsIndexTest extends SafetensorsTest {

    @Test
    public void testLoadSingleFile(@TempDir Path tempDir) throws IOException {
        // Create a single safetensors file
        Path modelFile = tempDir.resolve("model.safetensors");
        Safetensors st =
                Builder.newBuilder()
                        .putTensor(
                                TensorEntry.create(
                                        "layer1.weight", DType.F32, new long[] {10, 20}, 0))
                        .putTensor(
                                TensorEntry.create(
                                        "layer2.weight", DType.F32, new long[] {20, 30}, 0))
                        .build();
        Safetensors.write(st, modelFile);

        // Load directly from file
        SafetensorsIndex index = SafetensorsIndex.load(modelFile);

        assertNotNull(index);
        assertEquals(tempDir, index.getRootPath());
        Collection<String> tensorNames = index.getTensorNames();
        assertEquals(2, tensorNames.size());
        assertTrue(tensorNames.contains("layer1.weight"));
        assertTrue(tensorNames.contains("layer2.weight"));
        assertEquals(modelFile, index.getSafetensorsPath("layer1.weight"));
        assertEquals(modelFile, index.getSafetensorsPath("layer2.weight"));
    }

    @Test
    public void testLoadDirectoryWithSingleFile(@TempDir Path tempDir) throws IOException {
        // Create model.safetensors in directory
        Path modelFile = tempDir.resolve("model.safetensors");
        Safetensors st =
                Builder.newBuilder()
                        .putTensor(
                                TensorEntry.create(
                                        "embedding.weight", DType.F32, new long[] {1000, 512}, 0))
                        .build();
        Safetensors.write(st, modelFile);

        // Load from directory
        SafetensorsIndex index = SafetensorsIndex.load(tempDir);

        assertNotNull(index);
        assertEquals(tempDir, index.getRootPath());
        Collection<String> tensorNames = index.getTensorNames();
        assertEquals(1, tensorNames.size());
        assertTrue(tensorNames.contains("embedding.weight"));
        assertEquals(modelFile, index.getSafetensorsPath("embedding.weight"));
    }

    @Test
    public void testLoadShardedModel(@TempDir Path tempDir) throws IOException {
        // Create shard files
        Path shard1 = tempDir.resolve("model-00001-of-00002.safetensors");
        Path shard2 = tempDir.resolve("model-00002-of-00002.safetensors");

        Safetensors st1 =
                Builder.newBuilder()
                        .putTensor(
                                TensorEntry.create(
                                        "layer.0.weight", DType.F32, new long[] {100, 100}, 0))
                        .putTensor(
                                TensorEntry.create(
                                        "layer.1.weight", DType.F32, new long[] {100, 100}, 0))
                        .build();
        Safetensors.write(st1, shard1);

        Safetensors st2 =
                Builder.newBuilder()
                        .putTensor(
                                TensorEntry.create(
                                        "layer.2.weight", DType.F32, new long[] {100, 100}, 0))
                        .putTensor(
                                TensorEntry.create(
                                        "layer.3.weight", DType.F32, new long[] {100, 100}, 0))
                        .build();
        Safetensors.write(st2, shard2);

        // Create index.json
        String indexJson =
                "{\n"
                        + "  \"metadata\": {\n"
                        + "    \"total_size\": 160000\n"
                        + "  },\n"
                        + "  \"weight_map\": {\n"
                        + "    \"layer.0.weight\": \"model-00001-of-00002.safetensors\",\n"
                        + "    \"layer.1.weight\": \"model-00001-of-00002.safetensors\",\n"
                        + "    \"layer.2.weight\": \"model-00002-of-00002.safetensors\",\n"
                        + "    \"layer.3.weight\": \"model-00002-of-00002.safetensors\"\n"
                        + "  }\n"
                        + "}\n";
        Files.writeString(tempDir.resolve("model.safetensors.index.json"), indexJson);

        // Load sharded model
        SafetensorsIndex index = SafetensorsIndex.load(tempDir);

        assertNotNull(index);
        assertEquals(tempDir, index.getRootPath());
        Collection<String> tensorNames = index.getTensorNames();
        assertEquals(4, tensorNames.size());
        assertTrue(tensorNames.contains("layer.0.weight"));
        assertTrue(tensorNames.contains("layer.1.weight"));
        assertTrue(tensorNames.contains("layer.2.weight"));
        assertTrue(tensorNames.contains("layer.3.weight"));

        assertEquals(shard1, index.getSafetensorsPath("layer.0.weight"));
        assertEquals(shard1, index.getSafetensorsPath("layer.1.weight"));
        assertEquals(shard2, index.getSafetensorsPath("layer.2.weight"));
        assertEquals(shard2, index.getSafetensorsPath("layer.3.weight"));
    }

    @Test
    public void testGetSafetensorsPathForNonExistentTensor(@TempDir Path tempDir)
            throws IOException {
        Path modelFile = tempDir.resolve("model.safetensors");
        Safetensors st =
                Builder.newBuilder()
                        .putTensor(
                                TensorEntry.create(
                                        "existing.weight", DType.F32, new long[] {10, 10}, 0))
                        .build();
        Safetensors.write(st, modelFile);

        SafetensorsIndex index = SafetensorsIndex.load(tempDir);
        assertNull(index.getSafetensorsPath("nonexistent.weight"));
    }

    @Test
    public void testLoadNonExistentPath(@TempDir Path tempDir) {
        Path nonExistent = tempDir.resolve("does_not_exist");
        assertThrows(IOException.class, () -> SafetensorsIndex.load(nonExistent));
    }

    @Test
    public void testLoadDirectoryWithNoSafetensors(@TempDir Path tempDir) throws IOException {
        // Create empty directory
        Path emptyDir = tempDir.resolve("empty");
        Files.createDirectory(emptyDir);

        assertThrows(IOException.class, () -> SafetensorsIndex.load(emptyDir));
    }

    @Test
    public void testLoadMissingShardFile(@TempDir Path tempDir) throws IOException {
        // Create index.json that references non-existent shard
        String indexJson =
                "{\n"
                        + "  \"weight_map\": {\n"
                        + "    \"layer.weight\": \"missing-shard.safetensors\"\n"
                        + "  }\n"
                        + "}\n";
        Files.writeString(tempDir.resolve("model.safetensors.index.json"), indexJson);

        assertThrows(IOException.class, () -> SafetensorsIndex.load(tempDir));
    }

    @Test
    public void testInvalidIndexJsonFormat(@TempDir Path tempDir) throws IOException {
        // Create invalid JSON (missing closing brace)
        String invalidJson =
                "{\n"
                        + "  \"weight_map\": {\n"
                        + "    \"layer.weight\": \"shard.safetensors\"\n"
                        + "  }\n";
        Files.writeString(tempDir.resolve("model.safetensors.index.json"), invalidJson);

        assertThrows(SafetensorsFormatException.class, () -> SafetensorsIndex.load(tempDir));
    }

    @Test
    public void testMissingWeightMap(@TempDir Path tempDir) throws IOException {
        // Valid JSON but missing weight_map
        String indexJson =
                "{\n" + "  \"metadata\": {\n" + "    \"total_size\": 1000\n" + "  }\n" + "}\n";
        Files.writeString(tempDir.resolve("model.safetensors.index.json"), indexJson);

        assertThrows(SafetensorsFormatException.class, () -> SafetensorsIndex.load(tempDir));
    }

    @Test
    public void testWeightMapNotObject(@TempDir Path tempDir) throws IOException {
        // weight_map is a string instead of object
        String indexJson = "{ \"weight_map\": \"invalid\" }";
        Files.writeString(tempDir.resolve("model.safetensors.index.json"), indexJson);

        assertThrows(SafetensorsFormatException.class, () -> SafetensorsIndex.load(tempDir));
    }

    @Test
    public void testWeightMapInvalidKeys(@TempDir Path tempDir) throws IOException {
        // weight_map has non-string keys (note: this is tricky in JSON, but we can simulate by
        // having numbers)
        String indexJson = "{\"weight_map\": {\"123\": \"shard.safetensors\"}}";
        // Actually in JSON, all keys are strings, so let's test invalid values instead
        // Let me create a better test for this
        Files.writeString(tempDir.resolve("model.safetensors.index.json"), indexJson);

        // This should actually work since JSON keys are always strings
        // We need a shard file to exist
        Path shardFile = tempDir.resolve("shard.safetensors");
        Safetensors st =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("tensor", DType.F32, new long[] {1}, 0))
                        .build();
        Safetensors.write(st, shardFile);

        // This should succeed
        SafetensorsIndex index = SafetensorsIndex.load(tempDir);
        assertNotNull(index);
    }

    @Test
    public void testWeightMapInvalidValues(@TempDir Path tempDir) throws IOException {
        // weight_map values are not strings
        String indexJson =
                "{\n" + "  \"weight_map\": {\n" + "    \"layer.weight\": 123\n" + "  }\n" + "}\n";
        Files.writeString(tempDir.resolve("model.safetensors.index.json"), indexJson);

        assertThrows(SafetensorsFormatException.class, () -> SafetensorsIndex.load(tempDir));
    }

    @Test
    public void testEmptySafetensors(@TempDir Path tempDir) throws IOException {
        // Create empty safetensors file (no tensors)
        Path modelFile = tempDir.resolve("model.safetensors");
        Safetensors st = Builder.newBuilder().build();
        Safetensors.write(st, modelFile);

        SafetensorsIndex index = SafetensorsIndex.load(tempDir);

        assertNotNull(index);
        assertEquals(0, index.getTensorNames().size());
    }

    @Test
    public void testLoadInvalidFileType(@TempDir Path tempDir) throws IOException {
        // Try to load a .txt file
        Path txtFile = tempDir.resolve("model.txt");
        Files.writeString(txtFile, "not a safetensors file");

        assertThrows(IOException.class, () -> SafetensorsIndex.load(txtFile));
    }

    @Test
    public void testShardedModelWithMetadata(@TempDir Path tempDir) throws IOException {
        // Create shard with metadata
        Path shardFile = tempDir.resolve("model-00001-of-00001.safetensors");
        Safetensors st =
                Builder.newBuilder()
                        .putMetadataKey("format", "pt")
                        .putMetadataKey("model_type", "llama")
                        .putTensor(
                                TensorEntry.create("weight", DType.F32, new long[] {100, 100}, 0))
                        .build();
        Safetensors.write(st, shardFile);

        // Create index.json with additional metadata
        String indexJson =
                "{\n"
                        + "  \"metadata\": {\n"
                        + "    \"total_size\": 40000,\n"
                        + "    \"format\": \"pt\"\n"
                        + "  },\n"
                        + "  \"weight_map\": {\n"
                        + "    \"weight\": \"model-00001-of-00001.safetensors\"\n"
                        + "  }\n"
                        + "}\n";
        Files.writeString(tempDir.resolve("model.safetensors.index.json"), indexJson);

        SafetensorsIndex index = SafetensorsIndex.load(tempDir);

        assertNotNull(index);
        assertEquals(1, index.getTensorNames().size());
        assertTrue(index.getTensorNames().contains("weight"));
    }
}
