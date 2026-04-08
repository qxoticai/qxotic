package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import java.lang.reflect.Field;

/** Prints token counts used by {@link ModelTokenizerBenchmark} scenarios. */
public final class ModelTokenizerTokenCountProbe {

    private ModelTokenizerTokenCountProbe() {}

    public static void main(String[] args) throws Exception {
        String corpus = args.length > 0 ? args[0] : "prose";
        String size = args.length > 1 ? args[1] : "32k";

        String[] models = {"gpt2", "llama3", "qwen35", "mistral-tekken"};
        String[] implementations = {"reference", "classic", "fast"};

        Field encodedField = ModelTokenizerBenchmark.class.getDeclaredField("encoded");
        encodedField.setAccessible(true);

        System.out.println("model,implementation,corpus,size,tokens");
        for (String model : models) {
            for (String implementation : implementations) {
                ModelTokenizerBenchmark benchmark = new ModelTokenizerBenchmark();
                benchmark.model = model;
                benchmark.implementation = implementation;
                benchmark.corpus = corpus;
                benchmark.size = size;
                benchmark.setup();

                IntSequence encoded = (IntSequence) encodedField.get(benchmark);
                System.out.println(
                        model
                                + ","
                                + implementation
                                + ","
                                + corpus
                                + ","
                                + size
                                + ","
                                + encoded.length());
            }
        }
    }
}
