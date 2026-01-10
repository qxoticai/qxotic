package com.llm4j.test;

import com.llm4j.model.llama.Timer;
import com.llm4j.tokenizers.IntSequence;
import com.llm4j.tokenizers.Tokenizer;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Disabled
public class EncodeTest extends TokenizerTest {

    private static final Path WIKI_CORPUS = Path.of("/home/mukel/Desktop/playground/text/wiki-corpus/");

    @Disabled("too large")
    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testAllLanguages(Tokenizer tokenizer) throws IOException {
        try (Stream<Path> list = Files.list(WIKI_CORPUS)) {
            List<String> allLanguages = list.filter(Files::isRegularFile)
                    .map(Path::getFileName)
                    .map(Path::toString)
                    .toList();
            testWikiCorpus(tokenizer, allLanguages);
        }
    }

    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testCommonLanguages(Tokenizer tokenizer) {
        testWikiCorpus(tokenizer, List.of("en", "es", "de", "zh", "ru", "hi", "ko", "ja", "he"));
    }

    private static void testWikiCorpus(Tokenizer tokenizer, List<String> languages) {
        languages.stream().parallel().forEach(language -> {
            try (Timer timer = Timer.log(language)) {
                Path path = WIKI_CORPUS.resolve(language);
                assertTrue(Files.exists(path));
                assertTrue(Files.isRegularFile(path));

                String text = null;
                try {
                    text = Files.readString(path);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                IntSequence tokens = tokenizer.encode(text);
                assertEquals(text, tokenizer.decode(tokens));
            }
        });
    }
}
