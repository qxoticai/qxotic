package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** JMH benchmark for Tok'n'Roll HF-backed tokenizers on enwik corpora. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class HfModelWikiBenchmark {

    @Param({"enwik8"})
    public String corpus;

    @Param({"1", "4"})
    public int sliceMiB;

    @Param({
        "openai.gpt-oss",
        "meta.llama3",
        "moonshot.kimi2_6",
        "huggingface.smollm3",
        "alibaba.qwen3_5",
        "google.gemma4",
        "ibm.granite4_0",
        "microsoft.phi4",
        "deepseek.v3_2",
        "mistral.v0_3_spbpe"
    })
    public String modelFamily;

    private Tokenizer tokenizer;
    private String text;
    private IntSequence encoded;

    @Setup(Level.Trial)
    public void setup() {
        try {
            String corpusText = WikiBenchmarkSupport.loadCorpusText(corpus);
            int maxChars = Math.min(corpusText.length(), sliceMiB * 1024 * 1024);
            text = corpusText.substring(0, maxChars);
            tokenizer = loadTokenizer(modelFamily);
            encoded = tokenizer.encode(text);
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to initialize HF wiki benchmark for " + modelFamily, e);
        }
    }

    @Benchmark
    public void encode(Blackhole blackhole) {
        blackhole.consume(tokenizer.encode(text));
    }

    @Benchmark
    public void decode(Blackhole blackhole) {
        blackhole.consume(tokenizer.decode(encoded));
    }

    @Benchmark
    public void countTokens(Blackhole blackhole) {
        blackhole.consume(tokenizer.countTokens(text));
    }

    private static Tokenizer loadTokenizer(String familyId) {
        switch (familyId) {
            case "openai.gpt-oss":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "openai", "gpt-oss-20b", "main", false, false);
            case "meta.llama3":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "unsloth", "Llama-3.2-1B-Instruct", "main", false, false);
            case "moonshot.kimi2_6":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "moonshotai", "Kimi-K2.6", "main", false, false);
            case "huggingface.smollm3":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "HuggingFaceTB", "SmolLM3-3B", "main", false, false);
            case "alibaba.qwen3_5":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "Qwen", "Qwen3.5-0.8B", "main", false, false);
            case "google.gemma4":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "google", "gemma-4-e2b-it", "main", false, false);
            case "ibm.granite4_0":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "ibm-granite", "granite-4.0-h-1b", "main", false, false);
            case "microsoft.phi4":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "microsoft", "phi-4", "main", false, false);
            case "deepseek.v3_2":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "deepseek-ai", "DeepSeek-V3.2", "main", false, false);
            case "mistral.v0_3_spbpe":
                return HuggingFaceTokenizerLoader.fromPretrained(
                        "mistralai", "Mistral-7B-Instruct-v0.3", "main", false, false);
            default:
                throw new IllegalArgumentException("Unsupported HF model family: " + familyId);
        }
    }
}
