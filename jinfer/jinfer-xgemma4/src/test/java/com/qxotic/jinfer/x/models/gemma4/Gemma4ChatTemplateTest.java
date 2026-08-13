package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Embedder;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jinfer.x.boundary.MultiModal;
import com.qxotic.jinfer.x.chat.Channel;
import com.qxotic.jinfer.x.chat.ChatTemplate;
import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.Conversation;
import com.qxotic.jinfer.x.chat.Message;
import com.qxotic.jinfer.x.chat.Role;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

final class Gemma4ChatTemplateTest {

    @Test
    void plainTurnsMatchTheOracleValidatedOldPort() throws Exception {
        Tokenizer tokenizer = tokenizer();
        Gemma4ChatTemplate actual = new Gemma4ChatTemplate(tokenizer);
        com.qxotic.jinfer.models.gemma4.Gemma4TurnTemplate expected =
                new com.qxotic.jinfer.models.gemma4.Gemma4TurnTemplate(tokenizer);
        List<Message> messages =
                List.of(
                        Message.system(" You are concise. "),
                        Message.user("literal <|turn> and unicode: ñé漢字"),
                        Message.assistant(" history "),
                        Message.user("continue"));
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
            actual.encode(new Conversation(messages), capacity, batches::add);
            assertArrayEquals(oracle, Batch.tokenIds(batches), "batchCapacity " + capacity);
            assertTrue(batches.stream().allMatch(batch -> batch.count() <= capacity));
        }
    }

    @Test
    void imageAndAudioStayStructuralAndOrdered() throws Exception {
        Tokenizer tokenizer = tokenizer();
        byte[] imageKey = {1, 2};
        byte[] audioKey = {3, 4};
        Media.Image image = new Media.Image(new float[] {0, 0, 0}, 1, 1, 3);
        Media.Audio audio = new Media.Audio(new float[] {0}, 16_000, 1);
        Message message =
                new Message(
                        Role.USER,
                        List.of(
                                new Content.Text("look "),
                                new Content.Media(image, imageKey),
                                new Content.Text(" and hear "),
                                new Content.Media(audio, audioKey),
                                new Content.Text(".")));

        try (Arena arena = Arena.ofConfined()) {
            MultiModal media = new TestMedia(arena);
            Gemma4ChatTemplate template = new Gemma4ChatTemplate(tokenizer, media, false);
            List<Batch> batches = new ArrayList<>();
            template.encode(new Conversation(List.of(message)), 4, batches::add);

            List<Batch.Input.Embeddings> embeddings =
                    batches.stream()
                            .map(Batch::input)
                            .filter(Batch.Input.Embeddings.class::isInstance)
                            .map(Batch.Input.Embeddings.class::cast)
                            .toList();
            assertEquals(2, embeddings.size());
            assertTrue(embeddings.get(0).bidirectional());
            assertFalse(embeddings.get(1).bidirectional());
            assertArrayEquals(imageKey, embeddings.get(0).contentKey());
            assertArrayEquals(audioKey, embeddings.get(1).contentKey());
            assertArrayEquals(expectedMediaTokens(tokenizer), tokenIds(batches));
        }
    }

    @Test
    void nonThinkingScaffoldAndReplySeedStayTogether() throws Exception {
        Tokenizer tokenizer = tokenizer();
        Message message = Message.user("answer directly");
        Conversation conversation = new Conversation(List.of(message), List.of(), false, "");
        List<Batch> batches = new ArrayList<>();
        ChatTemplate.ReplyState reply =
                new Gemma4ChatTemplate(tokenizer, null, true)
                        .encode(conversation, 32, batches::add);

        com.qxotic.jinfer.models.gemma4.Gemma4TurnTemplate old =
                new com.qxotic.jinfer.models.gemma4.Gemma4TurnTemplate(tokenizer, null, 0, true);
        var oldPrompt =
                old.encodePrompt(
                        new com.qxotic.jinfer.chat.Conversation(
                                List.of(
                                        new com.qxotic.jinfer.chat.Message(
                                                com.qxotic.jinfer.chat.Role.USER, message.text())),
                                List.of(),
                                false,
                                ""));

        assertArrayEquals(
                com.qxotic.jinfer.Batch.tokenIds(oldPrompt.batches()), Batch.tokenIds(batches));
        assertFalse(reply.replyPrefix().isEmpty());
        assertEquals(Channel.CONTENT, reply.parser().channel());
    }

    private static int[] expectedMediaTokens(Tokenizer tokenizer) {
        IntSequence.Builder out = IntSequence.newBuilder();
        out.add(SpecialTokens.require(tokenizer, "<bos>"));
        out.add(SpecialTokens.require(tokenizer, "<|turn>"));
        out.addAll(tokenizer.encode("user\nlook "));
        out.add(SpecialTokens.require(tokenizer, "<|image>"));
        out.add(SpecialTokens.require(tokenizer, "<image|>"));
        out.addAll(tokenizer.encode(" and hear "));
        out.add(SpecialTokens.require(tokenizer, "<|audio>"));
        out.add(SpecialTokens.require(tokenizer, "<audio|>"));
        out.addAll(tokenizer.encode("."));
        out.add(SpecialTokens.require(tokenizer, "<turn|>"));
        out.addAll(tokenizer.encode("\n"));
        out.add(SpecialTokens.require(tokenizer, "<|turn>"));
        out.addAll(tokenizer.encode("model\n"));
        return out.build().toArray();
    }

    private static int[] tokenIds(List<Batch> batches) {
        IntSequence.Builder out = IntSequence.newBuilder();
        for (Batch batch : batches) {
            if (batch.input() instanceof Batch.Input.Tokens tokens)
                out.addAll(IntSequence.of(tokens.ids()));
        }
        return out.build().toArray();
    }

    private static Tokenizer tokenizer() throws Exception {
        Path path = TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0");
        try (FileChannel file = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(file, path.toString());
            return GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        }
    }

    private record TestMedia(Arena arena) implements MultiModal {
        @Override
        public Set<Class<? extends Media>> modalities() {
            return Set.of(Media.Image.class, Media.Audio.class);
        }

        @Override
        @SuppressWarnings("unchecked")
        public <R extends Media> Optional<Embedder<R>> embedder(Class<R> modality) {
            if (!modalities().contains(modality)) return Optional.empty();
            int rows = modality == Media.Image.class ? 2 : 1;
            Embedder<R> embedder =
                    (source, maxChunkSize, sink) ->
                            sink.accept(Views.allocateF32(new PanamaMemoryArena(arena), rows, 3));
            return Optional.of(embedder);
        }
    }
}
