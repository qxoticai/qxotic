package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.TestTokenizers;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

final class WikiBenchmarkSupport {

    private WikiBenchmarkSupport() {}

    static String readUtf8(Path path) throws IOException {
        return Files.readString(path, StandardCharsets.UTF_8);
    }

    static String loadCorpusText(String corpus) throws IOException {
        return readUtf8(WikiCorpusPaths.forCorpus(corpus));
    }

    static String loadCorpusSlice(String corpus, int sliceMiB) throws IOException {
        String corpusText = loadCorpusText(corpus);
        int maxChars = Math.min(corpusText.length(), sliceMiB * 1024 * 1024);
        return corpusText.substring(0, maxChars);
    }

    static Tokenizer createTokenizer(String encoding, boolean parallel) {
        Tokenizer tokenizer = TestTokenizers.tiktoken(encoding, splitterForEncoding(encoding));
        return parallel ? ParallelBenchmarkPipelines.from(tokenizer) : tokenizer;
    }

    static Tokenizer createO200kTokenizer(boolean parallel) {
        return createTokenizer("o200k_base", parallel);
    }

    private static Splitter splitterForEncoding(String encoding) {
        return Splitter.regex(TiktokenFixtures.splitPattern(encoding));
    }
}
