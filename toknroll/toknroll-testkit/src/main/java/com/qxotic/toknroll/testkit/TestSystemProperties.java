package com.qxotic.toknroll.testkit;

public final class TestSystemProperties {
    private TestSystemProperties() {}

    // Keep this key in sync with runtime cache property readers in HF/GGUF modules.
    public static final String ARTIFACT_CACHE_ROOT = "toknroll.cache.root";
    public static final String FIXTURE_DIR = "toknroll.test.fixtureDir";
    public static final String GOLDEN_DIR = "toknroll.test.goldenDir";
    public static final String CORPUS_PATH = "toknroll.test.corpus.path";
    public static final String MAX_CHUNKS = "toknroll.test.maxChunks";
    public static final String CHUNK_SIZE = "toknroll.test.chunk.size";
    public static final String HF_MAX_CHUNKS = "toknroll.test.hf.maxChunks";
    public static final String GGUF_MAX_CHUNKS = "toknroll.test.gguf.maxChunks";
    public static final String GGUF_GROUND_TRUTH_SOURCE = "toknroll.test.gguf.groundTruthSource";
    public static final String HF_SOURCE_COMPARISON_VERBOSE =
            "toknroll.test.hf.sourceComparison.verbose";
}
