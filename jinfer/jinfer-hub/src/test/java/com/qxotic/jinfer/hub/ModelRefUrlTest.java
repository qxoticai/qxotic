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

/**
 * A URL is never a model: a hub page's URL is refused with the exact ref to pass instead, and
 * anything else with the download remedy. A wrong suggestion is worse than none, so every shape
 * that could mislead names no ref.
 */
class ModelRefUrlTest {

    private static final String HF = "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF";
    private static final String MS = "https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF";

    private static void suggests(String expected, String... urls) {
        for (String url : urls)
            assertEquals(expected, String.valueOf(ModelRef.suggestionFor(url)), url);
    }

    private static void suggestsNothing(String... urls) {
        for (String url : urls) assertNull(ModelRef.suggestionFor(url), url);
    }

    @Test
    void aRepositoryPageInEverySpelling() {
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF",
                HF,
                HF + "/",
                HF + "/tree/main",
                HF + "/tree/main/",
                HF + "?not-for-all-audiences=true",
                HF + "#model-card",
                "https://hf.co/unsloth/gemma-4-E2B-it-GGUF",
                "https://www.huggingface.co/unsloth/gemma-4-E2B-it-GGUF",
                "http://huggingface.co/unsloth/gemma-4-E2B-it-GGUF",
                "HTTPS://HuggingFace.CO/unsloth/gemma-4-E2B-it-GGUF",
                "https://huggingface.co:443/unsloth/gemma-4-E2B-it-GGUF",
                "https://someone@huggingface.co/unsloth/gemma-4-E2B-it-GGUF",
                "https://huggingface.co//unsloth//gemma-4-E2B-it-GGUF//",
                "  " + HF + "  ");
    }

    @Test
    void revisionsAreKeptExceptTheDefaultBranch() {
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF/x.gguf", HF + "/blob/main/x.gguf");
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF@v2/x.gguf", HF + "/blob/v2/x.gguf");
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF@v1.0.3/x.gguf", HF + "/resolve/v1.0.3/x.gguf");
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF@release-2_b/x.gguf",
                HF + "/blob/release-2_b/x.gguf");
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF@9fe1c3a/x.gguf", HF + "/blob/9fe1c3a/x.gguf");
        String commit = "0123456789abcdef0123456789abcdef01234567";
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF@" + commit, HF + "/tree/" + commit);
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF@" + commit + "/MTP",
                HF + "/tree/" + commit + "/MTP");
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF@master/x.gguf",
                HF + "/blob/master/x.gguf"); // Hugging Face's default is main: master is pinned
        suggests("modelscope.cn/Qwen/Qwen3-0.6B-GGUF/x.gguf", MS + "/resolve/master/x.gguf");
        suggests("modelscope.cn/Qwen/Qwen3-0.6B-GGUF@main/x.gguf", MS + "/resolve/main/x.gguf");
        suggests("modelscope.cn/Qwen/Qwen3-0.6B-GGUF@v1", MS + "/tree/v1");
    }

    @Test
    void pathsAndFilesAsTheRepositoryHoldsThem() {
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF/MTP", HF + "/tree/main/MTP");
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF/a/b/c", HF + "/tree/main/a/b/c");
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf",
                HF + "/blob/main/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf",
                HF + "/resolve/main/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf?download=true");
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf",
                HF + "/blob/main/mmproj-F32.gguf");
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF/tokenizer.json",
                HF + "/blob/main/tokenizer.json");
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF/model.v2.Q8_0.gguf",
                HF + "/blob/main/model.v2.Q8_0.gguf");
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF/m q.gguf", HF + "/blob/main/m%20q.gguf");
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF/a+b.gguf", HF + "/blob/main/a+b.gguf");
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF/a+b.gguf", HF + "/blob/main/a%2Bb.gguf");
        suggests("hf.co/unsloth/gemma-4-E2B-it-GGUF/✓.gguf", HF + "/blob/main/%E2%9C%93.gguf");
        suggests(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0",
                "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0"); // a quant typed on
    }

    @Test
    void publishersAndModelsAsTheyAreNamed() {
        for (String repo :
                List.of(
                        "LiquidAI/LFM2.5-350M-GGUF",
                        "Qwen/Qwen3.5-4B-GGUF",
                        "ggml-org/gemma-4-E2B-it-GGUF",
                        "deepseek-ai/DeepSeek-V4-Pro",
                        "black-forest-labs/FLUX.1-dev",
                        "mradermacher/Qwen3-Reranker-0.6B-GGUF",
                        "bartowski/openai_gpt-oss-20b-GGUF",
                        "TheBloke/Llama-2-7B-GGUF",
                        "nvidia/NVIDIA-Nemotron-Nano-12B-v2")) {
            suggests("hf.co/" + repo, "https://huggingface.co/" + repo);
            suggests("modelscope.cn/" + repo, "https://modelscope.cn/models/" + repo);
        }
    }

    @Test
    void modelScopePages() {
        suggests(
                "modelscope.cn/Qwen/Qwen3-0.6B-GGUF",
                MS,
                MS + "/",
                MS + "/files",
                MS + "/summary",
                MS + "/tree/master",
                "https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF",
                "http://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF/files");
        suggests(
                "modelscope.cn/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
                MS + "/file/view/master/Qwen3-0.6B-Q8_0.gguf",
                MS + "/resolve/master/Qwen3-0.6B-Q8_0.gguf");
        suggests(
                "modelscope.cn/Qwen/Qwen3-0.6B-GGUF@v1/sub/x.gguf",
                MS + "/file/view/v1/sub/x.gguf");
    }

    @Test
    void aBadStructureNamesNoRef() {
        suggestsNothing(
                "https://huggingface.co",
                "https://huggingface.co/",
                "https://huggingface.co/acme",
                "https://huggingface.co/acme/",
                HF + "/blob",
                HF + "/blob/main/../../etc/passwd",
                "https://huggingface.co/../etc/passwd",
                "https://huggingface.co/acme/../x",
                "https://huggingface.co/acme%2Fx/y",
                "https://huggingface.co/acme/x%2Fy",
                HF + "/blob/refs%2Fpr%2F1/f.gguf",
                HF + "/tree/main%2F..",
                HF + "/discussions",
                HF + "/discussions/12",
                HF + "/commits/main",
                HF + "/settings",
                HF + "/edit/main/README.md",
                "https://modelscope.cn/Qwen/Qwen3-0.6B-GGUF",
                "https://modelscope.cn/models/Qwen",
                "https://modelscope.cn/datasets/Qwen/x",
                "https://modelscope.cn/studios/Qwen/x",
                MS + "/feedback",
                "https://",
                "https:///acme/x",
                "https:/huggingface.co/acme/x",
                "https://huggingface.co/acme x/y",
                "https://huggingface.co:notaport/acme/x",
                "https://[huggingface.co/acme/x",
                "",
                "   ");
    }

    @Test
    void theSiteOwnSectionsAreNeverOwners() {
        for (String section :
                List.of(
                        "datasets",
                        "spaces",
                        "docs",
                        "blog",
                        "papers",
                        "collections",
                        "learn",
                        "posts")) {
            suggestsNothing(
                    "https://huggingface.co/" + section + "/acme/x",
                    "https://hf.co/" + section + "/acme/x/blob/main/y.gguf");
        }
        suggestsNothing(
                "https://huggingface.co/docs/transformers",
                "https://huggingface.co/papers/2401.01234");
    }

    @Test
    void anotherProviderNamesNoRef() {
        suggestsNothing(
                "https://github.com/acme/x",
                "https://huggingface.com/acme/x",
                "https://hf.com/acme/x",
                "https://hf-mirror.com/acme/x",
                "https://huggingface.co.evil.example/acme/x",
                "https://evil.huggingface.co/acme/x",
                "https://cdn-lfs.huggingface.co/acme/x",
                "https://modelscope.com/models/acme/x",
                "https://modelscope.cn.evil.example/models/acme/x",
                "https://ollama.com/library/llama3",
                "https://example.org/models/x.gguf",
                "https://127.0.0.1/acme/x",
                "http://localhost:8080/acme/x");
    }

    @Test
    void onlyABrowsersLinkNamesARef() {
        suggestsNothing(
                "ftp://huggingface.co/acme/x",
                "file://huggingface.co/acme/x",
                "ssh://huggingface.co/acme/x",
                "git://huggingface.co/acme/x",
                "hf://acme/x",
                "s3://huggingface.co/acme/x",
                "hf.co/acme/x", // a reference, not a URL
                "acme/x");
    }

    @Test
    void theMessageNamesTheRefAndThePublisher() {
        assertEquals(
                "'"
                        + HF
                        + "/blob/v2/x.gguf' is a URL, not a model reference.\n"
                        + "Pass: hf.co/unsloth/gemma-4-E2B-it-GGUF@v2/x.gguf\n"
                        + "That is unsloth's repository on hf.co: check it is the publisher you"
                        + " meant to trust.",
                ModelRef.notAReference(HF + "/blob/v2/x.gguf"));
        assertTrue(
                ModelRef.notAReference(MS)
                        .endsWith(
                                "That is Qwen's repository on modelscope.cn: check it is the"
                                        + " publisher you meant to trust."));
        for (String other :
                List.of(
                        "https://example.org/x.gguf",
                        "https://huggingface.co/datasets/acme/x",
                        "https://evil.huggingface.co/acme/x",
                        "ftp://huggingface.co/acme/x")) {
            assertEquals(
                    "'"
                            + other
                            + "' is a URL, not a model reference: download the file and pass its"
                            + " local path, or name a model on hf.co or modelscope.cn as"
                            + " owner/repo[:quant]",
                    ModelRef.notAReference(other));
        }
    }

    @Test
    void everyDoorRefusesAUrlAndTouchesNothing(@TempDir Path root) throws IOException {
        ModelStore store = ModelStore.of(root); // no sources: nothing could be fetched anyway
        List<Consumer<String>> doors =
                List.of(
                        ModelRef::parse,
                        ModelStore::requireRef,
                        store::select,
                        store::resolve,
                        store::find,
                        store::remove,
                        url -> store.resolveAll(List.of(url)),
                        url -> store.resolveAll(List.of("hf.co/acme/x:Q8_0", url)),
                        url -> store.pullAll(List.of(url), false),
                        url -> store.pullAll(List.of(url), true));
        for (var door : doors) {
            String link = HF + "/blob/main/x-Q8_0.gguf";
            String message =
                    assertThrows(IllegalArgumentException.class, () -> door.accept(link))
                            .getMessage();
            assertTrue(
                    message.contains("Pass: hf.co/unsloth/gemma-4-E2B-it-GGUF/x-Q8_0.gguf"),
                    message);
            for (String other :
                    List.of(
                            "https://example.org/x.gguf",
                            "ftp://example.org/x.gguf",
                            "file:///tmp/x.gguf",
                            "https://huggingface.co/datasets/acme/x")) {
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
