package com.qxotic.toknroll.gguf;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import java.util.*;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TestQwenRegexSplitter {
    private static final String HF_QWEN35_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final Pattern HF_QWEN35_COMPILED = Pattern.compile(HF_QWEN35_PATTERN);

    static final Splitter HF_QWEN35_SPLITTER =
            new Splitter() {
                @Override
                public void splitAll(
                        CharSequence text,
                        int startInclusive,
                        int endExclusive,
                        SplitConsumer consumer) {
                    Matcher m = HF_QWEN35_COMPILED.matcher(text);
                    m.region(startInclusive, endExclusive);
                    while (m.find()) {
                        consumer.accept(text, m.start(), m.end());
                    }
                }
            };

    public static void main(String[] args) throws Exception {
        String[] inputs = {
            "thai ไทยภาษาไทย without spaces",
            "thai detailed: สวัสดีชาวโลก วันนี้อากาศดีมาก ไปเดินเล่นกันไหม",
            "thai with digits: เวอร์ชัน 2.5 เปิดตัววันที่ 03/04/2026",
            "Thai: สวัสดีชาวโลก",
            "Thai detailed: สวัสดีชาวโลก วันนี้อากาศดีมาก ไปเดินเล่นกันไหม",
            "Thai with digits: เวอร์ชัน 2.5 เปิดตัววันที่ 03/04/2026",
            "Thai สวัสดีชาวโลก วันนี้อากาศดีมาก",
            "<|system|>You are helpful.<|user|>Compare tokenizer throughput and latency.",
        };

        Optional<Tokenizer> maybeBase = ModelFamilyTokenizers.create("alibaba.qwen3_5");
        if (maybeBase.isEmpty()) {
            throw new IllegalStateException("Qwen 3.5 tokenizer unavailable");
        }
        Tokenizer base = maybeBase.get();

        System.out.println("=== Java with HF regex splitter ===");
        for (String text : inputs) {
            IntSequence seq = base.encode(text);
            String dec = base.decode(seq);
            System.out.println("TEXT: " + text);
            System.out.println("IDS : " + seq);
            System.out.println("MATCH: " + dec.equals(text));
            System.out.println("LEN : " + seq.length());
            System.out.println();
        }
    }
}
