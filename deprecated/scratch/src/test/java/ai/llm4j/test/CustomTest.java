package ai.llm4j.test;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

@Disabled
public class CustomTest extends TokenizerTest {
    private static final String[] TEST_STRINGS = {
        "", // empty string
        "?", // single character
        "hello world!!!? (안녕하세요!) lol123 😉", // fun small string
        "🌟🎉🌈🍕💡🎸🌺🚀"
    };

    private static final String SPECIALS_STRING =
            """
            <|endoftext|>Hello world this is one document
            <|endoftext|>And this is another document
            <|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
            <|endoftext|>Last document!!! 👋<|endofprompt|>
            """
                    .strip();

    private static final String LLAMA_TEXT =
            """
            <|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
            Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
            The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
            <|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
            """
                    .strip();

    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testEncodeDecodeIdentity(Tokenizer tokenizer) {
        for (String text : TEST_STRINGS) {
            IntSequence tokens = tokenizer.encode(text);
            String decoded = tokenizer.decode(tokens);
            assertEquals(text, decoded);
        }
    }
}
