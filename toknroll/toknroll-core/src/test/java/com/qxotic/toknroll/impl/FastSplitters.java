package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;

public final class FastSplitters {

    private FastSplitters() {}

    public static Splitter cl100k() {
        return FastCl100kSplitter.INSTANCE;
    }

    public static Splitter r50k() {
        return FastR50kSplitter.INSTANCE;
    }

    public static Splitter o200k() {
        return FastO200kSplitter.INSTANCE;
    }

    public static Splitter llama3() {
        return FastLlama3Splitter.INSTANCE;
    }

    public static Splitter qwen35() {
        return FastQwen35Splitter.INSTANCE;
    }

    public static Splitter tekken() {
        return FastTekkenSplitter.INSTANCE;
    }
}
