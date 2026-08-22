package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.MediaEncodingCache;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

final class Gemma4ChatTemplateTest {

    @Test
    void imageAndAudioStayStructuralAndOrdered() throws Exception {
        Tokenizer tokenizer = tokenizer();
        ContentKey imageKey = new ContentKey("image:test");
        ContentKey audioKey = new ContentKey("audio:test");
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
            Multimodal media = new TestMedia(arena);
            Gemma4ChatTemplate template = new Gemma4ChatTemplate(tokenizer, media, false);
            assertEquals(
                    IntSequence.of(SpecialTokens.require(tokenizer, "<bos>")),
                    template.promptStart());
            List<Batch> batches = new ArrayList<>();
            template.encode(
                    new Conversation(List.of(message), List.of(), false, ""), 4, batches::add);

            List<Batch.Input.Embeddings> embeddings =
                    batches.stream()
                            .map(Batch::input)
                            .filter(Batch.Input.Embeddings.class::isInstance)
                            .map(Batch.Input.Embeddings.class::cast)
                            .toList();
            assertEquals(2, embeddings.size());
            assertTrue(embeddings.get(0).bidirectional());
            assertFalse(embeddings.get(1).bidirectional());
            assertEquals(imageKey, embeddings.get(0).contentKey());
            assertEquals(audioKey, embeddings.get(1).contentKey());
            assertArrayEquals(expectedMediaTokens(tokenizer), tokenIds(batches));
        }
    }

    @Test
    void repeatedImageReplaysProjectedRows() throws Exception {
        Tokenizer tokenizer = tokenizer();
        ContentKey key = new ContentKey("image:test");
        Message message =
                new Message(
                        Role.USER,
                        List.of(
                                new Content.Media(
                                        new Media.Image(new float[] {0, 0, 0}, 1, 1, 3), key)));
        AtomicInteger projections = new AtomicInteger();
        try (Arena arena = Arena.ofConfined()) {
            Gemma4ChatTemplate template =
                    new Gemma4ChatTemplate(tokenizer, new TestMedia(arena, projections), false);
            MediaEncodingCache cache = new MediaEncodingCache();
            for (int pass = 0; pass < 2; pass++) {
                template.encode(new Conversation(List.of(message)), 4, cache, ignored -> {});
            }
        }
        assertEquals(1, projections.get());
    }

    @Test
    void nonThinkingReplyStartsInContentChannel() throws Exception {
        Tokenizer tokenizer = tokenizer();
        Message message = Message.user("answer directly");
        Conversation conversation = new Conversation(List.of(message), List.of(), false, "");
        ChatTemplate.ReplyState reply =
                new Gemma4ChatTemplate(tokenizer, null, true)
                        .encode(conversation, 32, ignored -> {});

        assertFalse(reply.replyPrefix().isEmpty());
        assertEquals(Channel.CONTENT, reply.parser().channel());
    }

    @Test
    void thinkMarkersDeclaresTheChannelSpan() throws Exception {
        Tokenizer tokenizer = tokenizer();
        ChatTemplate.ThinkMarkers markers =
                new Gemma4ChatTemplate(tokenizer, null, true).thinkMarkers();
        assertEquals("<|channel>", markers.open());
        assertEquals("<channel|>", markers.close());
        // the declared spellings resolve in the family vocabulary - the engine's ban/cap keys on
        // their ids, so a spelling that does not resolve would silently disable both policies
        assertTrue(SpecialTokens.find(tokenizer, markers.open()).isPresent());
        assertTrue(SpecialTokens.find(tokenizer, markers.close()).isPresent());
    }

    @Test
    void thinkingWithoutSystemOrToolsStillEmitsTheSeededSystemTurn() throws Exception {
        Tokenizer tokenizer = tokenizer();
        List<Batch> batches = new ArrayList<>();
        new Gemma4ChatTemplate(tokenizer, null, true)
                .encode(
                        new Conversation(List.of(Message.user("hi")), List.of(), true, ""),
                        32,
                        batches::add);
        int[] ids = tokenIds(batches);

        // the E2B dual-mode template: bos, then <|turn>system\n<|think|>\n<turn|> even with no
        // system message and no tools - the seed primes the <|channel>thought span in the reply
        IntSequence.Builder prefix = IntSequence.newBuilder();
        prefix.add(SpecialTokens.require(tokenizer, "<bos>"));
        prefix.add(SpecialTokens.require(tokenizer, "<|turn>"));
        prefix.addAll(tokenizer.encode("system\n"));
        prefix.add(SpecialTokens.require(tokenizer, "<|think|>"));
        prefix.addAll(tokenizer.encode("\n"));
        prefix.add(SpecialTokens.require(tokenizer, "<turn|>"));
        int[] expected = prefix.build().toArray();
        assertTrue(ids.length > expected.length);
        assertArrayEquals(expected, Arrays.copyOf(ids, expected.length));
    }

    @Test
    void theSeedLandsAheadOfTheSystemText() throws Exception {
        Tokenizer tokenizer = tokenizer();
        List<Batch> batches = new ArrayList<>();
        new Gemma4ChatTemplate(tokenizer, null, true)
                .encode(
                        new Conversation(
                                List.of(Message.system("be terse"), Message.user("hi")),
                                List.of(),
                                true,
                                ""),
                        32,
                        batches::add);
        int[] ids = tokenIds(batches);

        int seed = indexOf(ids, SpecialTokens.require(tokenizer, "<|think|>"));
        assertTrue(seed > 0, "seed present");
        int systemText = indexOf(ids, tokenizer.encode("be terse").intAt(0));
        assertTrue(systemText > seed, "system text follows the seed");
    }

    @Test
    void thinkingOffNeverEmitsTheSeed() throws Exception {
        Tokenizer tokenizer = tokenizer();
        List<Batch> batches = new ArrayList<>();
        new Gemma4ChatTemplate(tokenizer, null, true)
                .encode(
                        new Conversation(
                                List.of(Message.system("be terse"), Message.user("hi")),
                                List.of(),
                                false,
                                ""),
                        32,
                        batches::add);
        int[] ids = tokenIds(batches);
        assertEquals(-1, indexOf(ids, SpecialTokens.require(tokenizer, "<|think|>")));
    }

    @Test
    void thinkingLeavesTheChannelOpenToTheModel() throws Exception {
        Tokenizer tokenizer = tokenizer();
        ChatTemplate.ReplyState reply =
                new Gemma4ChatTemplate(tokenizer, null, true)
                        .encode(
                                new Conversation(
                                        List.of(Message.user("think about it")),
                                        List.of(),
                                        true,
                                        ""),
                                32,
                                ignored -> {});
        // a seeded model opens <|channel>thought itself: no scaffolded prefix (contrast with the
        // non-thinking load, which carries the content-channel prefix)
        assertTrue(reply.replyPrefix().isEmpty());
        assertEquals(Channel.CONTENT, reply.parser().channel());
    }

    private static int indexOf(int[] ids, int id) {
        for (int i = 0; i < ids.length; i++) {
            if (ids[i] == id) return i;
        }
        return -1;
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

    private record TestMedia(Arena arena, AtomicInteger projections) implements Multimodal {
        private TestMedia(Arena arena) {
            this(arena, new AtomicInteger());
        }

        @Override
        @SuppressWarnings("unchecked")
        public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
            if (modality != Media.Image.class && modality != Media.Audio.class)
                return Optional.empty();
            int rows = modality == Media.Image.class ? 2 : 1;
            MediaProjector<R> projector =
                    new MediaProjector<>() {
                        @Override
                        public int positions(R source) {
                            return rows;
                        }

                        @Override
                        public void project(
                                R source,
                                int maxChunkSize,
                                java.util.function.Consumer<MemoryView<?>> sink) {
                            projections.incrementAndGet();
                            sink.accept(Views.allocateF32(new PanamaMemoryArena(arena), rows, 3));
                        }
                    };
            return Optional.of(projector);
        }
    }
}
