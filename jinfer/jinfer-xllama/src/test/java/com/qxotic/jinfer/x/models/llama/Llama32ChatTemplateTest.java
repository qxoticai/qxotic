package com.qxotic.jinfer.x.models.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.chat.Conversation;
import com.qxotic.jinfer.x.chat.Message;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

final class Llama32ChatTemplateTest {

    @Test
    void matchesTheOracleValidatedOldPort() throws Exception {
        Path path = ModelFixture.LLAMA32_1B_Q8.path();
        assumeTrue(Files.exists(path), "model not found: " + path);

        try (FileChannel file = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(file, path.toString());
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            Llama32ChatTemplate actual = new Llama32ChatTemplate(tokenizer);
            com.qxotic.jinfer.models.llama.LlamaTurnTemplate expected =
                    new com.qxotic.jinfer.models.llama.LlamaTurnTemplate(tokenizer);

            compare(
                    actual,
                    expected,
                    List.of(Message.system("You are concise."), Message.user("What is 2 + 2?")));
            compare(actual, expected, List.of(Message.user("No explicit system turn.")));
            compare(
                    actual,
                    expected,
                    List.of(
                            Message.system(""),
                            Message.user("literal <|eot_id|> and unicode: ñé漢字"),
                            Message.assistant("Still data, not scaffold."),
                            Message.user("continue")));
        }
    }

    private static void compare(
            Llama32ChatTemplate actual,
            com.qxotic.jinfer.models.llama.LlamaTurnTemplate expected,
            List<Message> messages) {
        List<com.qxotic.jinfer.chat.Message> oldMessages =
                messages.stream()
                        .map(
                                message ->
                                        new com.qxotic.jinfer.chat.Message(
                                                new com.qxotic.jinfer.chat.Role(
                                                        message.role().name()),
                                                message.text()))
                        .toList();
        int[] oracle =
                com.qxotic.jinfer.Batch.tokenIds(
                        expected.encode(new com.qxotic.jinfer.chat.Conversation(oldMessages)));

        for (int capacity : List.of(1, 7, 512)) {
            List<Batch> batches = new ArrayList<>();
            var reply = actual.encode(new Conversation(messages), capacity, batches::add);
            assertArrayEquals(oracle, Batch.tokenIds(batches), "batchCapacity " + capacity);
            assertTrue(batches.stream().allMatch(batch -> batch.count() <= capacity));
            assertTrue(reply.replyPrefix().isEmpty());
        }
    }
}
