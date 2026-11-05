package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.toknroll.ByteLevel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** How toknroll_token and toknroll_id spell a token, on the handle's {@link Native.Loaded}. */
class LoadedTest {
    @TempDir Path dir;

    /**
     * A byte-level vocabulary keeps an added token in its own alphabet, three spaces as "ĠĠĠ",
     * which no text contains: its spelling is its text, the one TOKNROLL_SPECIALS matches.
     */
    @Test
    void anAddedTokenIsSpelledAsItsTextAndFoundByIt() throws IOException {
        StringBuilder vocab = new StringBuilder();
        for (int b = 0; b < 256; b++)
            vocab.append(quoted(ByteLevel.encode(new byte[] {(byte) b})))
                    .append(':')
                    .append(b)
                    .append(',');
        String json =
                "{\"version\":\"1.0\",\"added_tokens\":[{\"id\":256,\"content\":\"  "
                    + " \",\"single_word\":false,"
                    + "\"lstrip\":false,\"rstrip\":false,\"normalized\":true,\"special\":false}],"
                    + "\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
                    + "\"trim_offsets\":true,\"use_regex\":true},\"post_processor\":null,"
                    + "\"decoder\":{\"type\":\"ByteLevel\"},\"model\":{\"type\":\"BPE\",\"vocab\":{"
                        + vocab.substring(0, vocab.length() - 1)
                        + "},\"merges\":[]}}";
        Path file = Files.writeString(dir.resolve("tokenizer.json"), json);
        Native.Loaded loaded = Native.Loaded.of(Sources.load(file.toString(), ModelStore.of(dir)));
        assertEquals("   ", loaded.spelling(256));
        assertEquals(256, loaded.id("   "));
        assertEquals("Ġ", loaded.spelling(' ')); // a normal token, as the vocabulary writes it
        assertEquals(' ', loaded.id("Ġ"));
    }

    private static String quoted(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
