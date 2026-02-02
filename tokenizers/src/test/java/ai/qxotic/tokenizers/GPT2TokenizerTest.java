package ai.qxotic.tokenizers;

import ai.qxotic.tokenizers.impl.ClassicBPE;
import ai.qxotic.tokenizers.impl.Tiktoken;
import ai.qxotic.tokenizers.impl.GPT2Tokenizer;
import ai.qxotic.tokenizers.impl.RegexSplitter;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class GPT2TokenizerTest {

    private static final String R50K_BASE_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    private static Path resourcePath(String fileName) {
        try {
            return Path.of(
                    GPT2TokenizerTest.class
                            .getClassLoader()
                            .getResource("tiktoken/" + fileName)
                            .toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Failed to resolve " + fileName, e);
        }
    }

    private Tokenizer createGPT2Tokenizer() {
        try {
            Path tiktokenPath = resourcePath("r50k_base.tiktoken");
            var mergeableRanks = ClassicBPE.loadMergeableRanks(
                    tiktokenPath.toString(), R50K_BASE_HASH);
            
            return ClassicBPE.classicFromTiktoken(
                    mergeableRanks,
                    Map.of("", 50256),
                    Normalizer.IDENTITY,
                    RegexSplitter.create(R50K_PATTERN)
            );
        } catch (Exception e) {
            throw new IllegalStateException("Failed to create GPT2 tokenizer", e);
        }
    }

    private Tokenizer createJTokkitTokenizer() {
        try {
            Path tiktokenPath = resourcePath("r50k_base.tiktoken");
            var mergeableRanks = ClassicBPE.loadMergeableRanks(
                    tiktokenPath.toString(), R50K_BASE_HASH);
            
            Pattern pattern = Pattern.compile(R50K_PATTERN);
            
            Map<String, Integer> specialTokens = Map.of();
            
            return Tiktoken.createFromTiktoken(
                    "r50k_base_test",
                    mergeableRanks,
                    pattern,
                    specialTokens
            );
        } catch (Exception e) {
            throw new IllegalStateException("Failed to create jtokkit tokenizer", e);
        }
    }

    @Test
    void testDaughterPortrayingCharliesMother() {
        Tokenizer tokenizer = createGPT2Tokenizer();
        String text = "daughter, portraying Charlie's mother";
        
        IntSequence tokens = tokenizer.encode(text);
        
        Assertions.assertEquals(6, tokens.length(), "Should produce 6 tokens");
        
        int[] expectedTokens = {29642, 11, 42458, 11526, 338, 2802};
        for (int i = 0; i < expectedTokens.length; i++) {
            Assertions.assertEquals(expectedTokens[i], tokens.intAt(i),
                    "Token at position " + i + " should be " + expectedTokens[i]);
        }
        
        String decoded = tokenizer.decode(tokens);
        Assertions.assertEquals(text, decoded, "Round-trip should work");
        
        int tokenCount = tokenizer.countTokens(text);
        Assertions.assertEquals(tokens.length(), tokenCount, "countTokens should match encode length");
        
        for (int i = 0; i < tokens.length(); i++) {
            int tokenId = tokens.intAt(i);
            Assertions.assertTrue(tokenizer.vocabulary().contains(tokenId),
                    "Token " + tokenId + " should be in vocabulary");
        }
    }

    @Test
    void testDaughterPortrayingCharliesMotherWithJTokkit() {
        Tokenizer tokenizer = createJTokkitTokenizer();
        String text = "daughter, portraying Charlie's mother";
        
        IntSequence tokens = tokenizer.encode(text);
        
        Assertions.assertEquals(6, tokens.length(), "Should produce 6 tokens");
        
        int[] expectedTokens = {29642, 11, 42458, 11526, 338, 2802};
        for (int i = 0; i < expectedTokens.length; i++) {
            Assertions.assertEquals(expectedTokens[i], tokens.intAt(i),
                    "Token at position " + i + " should be " + expectedTokens[i]);
        }
        
        String decoded = tokenizer.decode(tokens);
        Assertions.assertEquals(text, decoded, "Round-trip should work");
        
        int tokenCount = tokenizer.countTokens(text);
        Assertions.assertEquals(tokens.length(), tokenCount, "countTokens should match encode length");
        
        for (int i = 0; i < tokens.length(); i++) {
            int tokenId = tokens.intAt(i);
            Assertions.assertTrue(tokenizer.vocabulary().contains(tokenId),
                    "Token " + tokenId + " should be in vocabulary");
        }
    }

    @Test
    void testBothTokenizersProduceSameTokens() {
        Tokenizer classicTokenizer = createGPT2Tokenizer();
        Tokenizer jtokkitTokenizer = createJTokkitTokenizer();
        String text = "daughter, portraying Charlie's mother";
        
        IntSequence classicTokens = classicTokenizer.encode(text);
        IntSequence jtokkitTokens = jtokkitTokenizer.encode(text);
        
        Assertions.assertEquals(classicTokens.length(), jtokkitTokens.length(),
                "Both tokenizers should produce same number of tokens");
        
        for (int i = 0; i < classicTokens.length(); i++) {
            Assertions.assertEquals(classicTokens.intAt(i), jtokkitTokens.intAt(i),
                    "Token at position " + i + " should match");
        }
    }

    @Test
    void testPortrayingWithAndWithoutSpace() {
        Tokenizer tokenizer = createJTokkitTokenizer();
        
        String textWithSpace = " portraying";
        IntSequence tokensWithSpace = tokenizer.encode(textWithSpace);
        
        Assertions.assertEquals(1, tokensWithSpace.length(), 
                "' portraying' (with leading space) should be 1 token");
        Assertions.assertEquals(42458, tokensWithSpace.intAt(0),
                "' portraying' should be token 42458");
        
        String textWithoutSpace = "portraying";
        IntSequence tokensWithoutSpace = tokenizer.encode(textWithoutSpace);
        
        Assertions.assertEquals(3, tokensWithoutSpace.length(), 
                "'portraying' (without space) should be 3 tokens");
        int[] expectedTokens = {634, 2433, 278};
        for (int i = 0; i < expectedTokens.length; i++) {
            Assertions.assertEquals(expectedTokens[i], tokensWithoutSpace.intAt(i),
                    "Token at position " + i + " for 'portraying' should be " + expectedTokens[i]);
        }
    }
}
