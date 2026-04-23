package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.TestTokenizers;
import java.util.List;
import java.util.Optional;

public class MeasureTokensPerCharLarge {

    static final String[] TEST_TEXTS = {
        "Hello world, this is a test of tokenization efficiency.",
        "The quick brown fox jumps over the lazy dog.",
        "public static void main(String[] args) { System.out.println(\"Hello\"); }",
        "Machine learning is a subset of artificial intelligence.",
        "你好世界，这是一个中文测试文本。",
        "日本語のテキストをトークン化します。",
        "한국어 텍스트 토큰화 테스트입니다.",
        "العربية هي لغة جميلة وغنية.",
        "Русский язык — один из самых распространенных языков мира.",
        "Γειά σου Κόσμε, αυτό είναι ένα ελληνικό κείμενο.",
        "Hello世界مرحباשלוםПривет",
        "12345 67890 11111 22222 33333",
        "The price is $1,234.56 and the discount is 25%.",
        "Visit https://example.com/path?query=value for more info.",
        "😀🎉👨‍👩‍👧‍👦🏳️‍🌈❤️👍🏽",
        "Unicode test: café, naïve, résumé, Zürich, Ålesund.",
        "Mixed: Hello 世界, مرحبا بالعالم, שלום עולם, Привет мир!",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "<|system|>You are helpful<|end|><|user|>Hello<|end|>",
        "SolidGoldMagikarp",
        "To be or not to be, that is the question.",
        "All human beings are born free and equal in dignity and rights.",
        "The defendant was charged with assault and battery.",
        "Photosynthesis converts light energy into chemical energy.",
        "E=mc² is the world's most famous equation.",
        "In 1492, Columbus sailed the ocean blue.",
        "The GDP increased by 3.5% in Q4 2024.",
        "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }",
        "SELECT id, name, email FROM users WHERE active = true ORDER BY created_at DESC;",
        "README.md LICENSE.txt src/main/java/com/example/Main.java",
        "Error: NullPointerException at line 42 in Tokenizer.java",
        "TODO: Fix the bug in the tokenization pipeline.",
        "// FIXME: This is a hack, refactor later.",
        "git commit -m \"Fix tokenization bug\" && git push origin main",
        "docker build -t myapp:latest . && docker run -p 8080:8080 myapp:latest",
        "curl -X POST https://api.example.com/v1/tokenize -H \"Content-Type: application/json\" -d"
                + " '{\"text\":\"hello\"}'",
        "npm install @qxotic/toknroll && npm run build",
        "pip install transformers torch numpy",
        "conda create -n toknroll python=3.11 && conda activate toknroll",
        " ssh user@server.example.com \"cd /app && ./deploy.sh\"",
        "echo $PATH | tr ':' '\\n' | head -5",
        "ls -la | grep \"\\.java$\" | wc -l",
        "cat file.txt | sort | uniq -c | sort -nr | head -10",
        "find . -name \"*.log\" -mtime +7 -delete",
        "tar -czf backup.tar.gz /home/user/documents /home/user/projects",
        "scp -r local_dir/ user@remote:/path/to/destination/",
        "rsync -avz --delete /src/ /dst/",
        "python -m pytest tests/ -xvs -k \"test_tokenize\"",
        "mvn clean test -pl toknroll-core -Dtest=TokenizerGoldenTest",
        "gradle build --scan && gradle test",
        "cargo build --release && cargo test",
        "go test ./... -v -count=1",
        "pytest --cov=myapp --cov-report=html tests/",
    };

    public static void main(String[] args) {
        System.out.println("=== TikToken Encodings ===");
        for (String enc :
                List.of("r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base")) {
            try {
                Tokenizer t = TestTokenizers.tiktokenReference(enc);
                measure(t, enc);
            } catch (Exception e) {
                System.out.println(enc + ": ERROR - " + e.getMessage());
            }
        }

        System.out.println("\\n=== Model Families ===");
        String[] families = {
            "google.gemma3",
            "google.gemma4",
            "alibaba.qwen3_5",
            "meta.llama3",
            "moonshot.kimi2_5",
            "ibm.granite4_0",
            "huggingface.smollm3",
            "mistral.gpt2_pretekken",
            "deepseek.v3_2",
            "microsoft.phi4",
            "mistral.tekken",
            "openai.gpt-oss"
        };
        for (String family : families) {
            try {
                Optional<Tokenizer> opt = TestTokenizers.modelFamily(family);
                if (opt.isPresent()) {
                    measure(opt.get(), family);
                } else {
                    System.out.println(family + ": NOT AVAILABLE");
                }
            } catch (Exception e) {
                System.out.println(family + ": ERROR - " + e.getMessage());
            }
        }
    }

    static void measure(Tokenizer tokenizer, String name) {
        long totalChars = 0;
        long totalTokens = 0;
        for (String text : TEST_TEXTS) {
            IntSequence tokens = tokenizer.encode(text);
            totalChars += text.length();
            totalTokens += tokens.length();
        }
        double ratio = (double) totalTokens / totalChars;
        System.out.printf(
                "%-30s ratio=%.4f (%d tokens / %d chars)%n", name, ratio, totalTokens, totalChars);
    }
}
