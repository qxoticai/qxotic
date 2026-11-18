package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** A URL is never a model: a hub page's URL is refused with the exact ref to pass instead. */
class ModelRefUrlTest {

    @Test
    void aHubPageNamesItsRef() {
        String hf = "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF";
        String ms = "https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF";
        String[][] cases = {
            {hf, "hf.co/unsloth/gemma-4-E2B-it-GGUF"},
            {hf + "/tree/main", "hf.co/unsloth/gemma-4-E2B-it-GGUF"},
            {hf + "/tree/main/MTP", "hf.co/unsloth/gemma-4-E2B-it-GGUF/MTP"},
            {
                hf + "/blob/main/mmproj-F32.gguf",
                "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf"
            },
            {
                hf + "/resolve/main/mmproj-F32.gguf?download=true",
                "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf"
            },
            {hf + "/blob/v2/sub/mtp.gguf", "hf.co/unsloth/gemma-4-E2B-it-GGUF@v2/sub/mtp.gguf"},
            {hf + "/blob/9fe1c3a/x.gguf", "hf.co/unsloth/gemma-4-E2B-it-GGUF@9fe1c3a/x.gguf"},
            {"https://hf.co/Qwen/Qwen3-0.6B-GGUF", "hf.co/Qwen/Qwen3-0.6B-GGUF"},
            {
                "http://www.huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q8_0.gguf#top",
                "hf.co/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
            },
            {ms, "modelscope.cn/Qwen/Qwen3-0.6B-GGUF"},
            {ms + "/files", "modelscope.cn/Qwen/Qwen3-0.6B-GGUF"},
            {
                ms + "/file/view/master/Qwen3-0.6B-Q8_0.gguf",
                "modelscope.cn/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
            },
            {
                ms + "/resolve/master/Qwen3-0.6B-Q8_0.gguf",
                "modelscope.cn/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
            },
            {ms + "/tree/v1", "modelscope.cn/Qwen/Qwen3-0.6B-GGUF@v1"},
        };
        for (String[] c : cases) {
            assertEquals(c[1], String.valueOf(ModelRef.suggestionFor(c[0])), c[0]);
        }
    }

    @Test
    void anythingElseNamesNoRef() {
        for (String url :
                List.of(
                        "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/discussions",
                        "https://huggingface.co/datasets/acme/corpus",
                        "https://huggingface.co/spaces/acme/demo",
                        "https://huggingface.co/acme/x/blob/refs%2Fpr%2F1/f.gguf",
                        "https://huggingface.co/acme",
                        "https://huggingface.co/",
                        "https://modelscope.cn/Qwen/Qwen3-0.6B-GGUF",
                        "https://example.org/models/x.gguf",
                        "hf.co/acme/x")) {
            assertNull(ModelRef.suggestionFor(url), url);
        }
    }

    @Test
    void everyDoorRefusesAUrlWithTheRefToPass(@TempDir Path root) throws IOException {
        ModelStore store = ModelStore.of(root); // no sources: nothing could be fetched anyway
        List<Consumer<String>> doors =
                List.of(
                        ModelRef::parse,
                        ModelStore::requireRef,
                        store::resolve,
                        store::find,
                        store::remove);
        String link = "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/blob/main/x-Q8_0.gguf";
        for (var door : doors) {
            String message =
                    assertThrows(IllegalArgumentException.class, () -> door.accept(link))
                            .getMessage();
            assertTrue(
                    message.contains("Pass: hf.co/LiquidAI/LFM2.5-350M-GGUF/x-Q8_0.gguf"), message);
            assertTrue(message.contains("LiquidAI's repository on hf.co"), message);
            for (String other : List.of("https://example.org/x.gguf", "ftp://example.org/x.gguf")) {
                assertTrue(
                        assertThrows(IllegalArgumentException.class, () -> door.accept(other))
                                .getMessage()
                                .contains("download the file and pass its local path"),
                        other);
            }
        }
        try (var left = Files.list(root)) {
            assertEquals(0, left.count(), "a refused URL leaves nothing behind");
        }
    }
}
