package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Modality;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.Verbatim;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpeechOutput;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * LFM3-Omni's spoken replies through the engine, as an application meets them: text to speech,
 * voice chat with its transcript, a spoken turn replayed in history. The model's own speech
 * recognition is the judge - what it hears must be what was meant.
 */
@Tag("integration")
final class Lfm3SpokenChatTest {

    private static final Sampling SEEDED = new Sampling(0f, 1f, 0, 0f, 42L);
    private static Arena weights;
    private static ChatEngine engine;

    @BeforeAll
    static void load() throws Exception {
        weights = Arenas.newCrossThread();
        engine =
                new ChatEngine(
                        Models.load(
                                TestModels.require(Lfm3IntegrationTest.OMNI),
                                weights,
                                Map.of(
                                        "media",
                                        TestModels.require(Lfm3IntegrationTest.OMNI_MMPROJ))),
                        "lfm3-speech",
                        PromptCache.Options.DEFAULTS);
    }

    @AfterAll
    static void close() {
        engine.close();
        Arenas.close(weights);
    }

    /**
     * Text to speech is a request answered in audio alone: streamed in parts, and whole, the parts
     * are the waveform, and the words survive. Each call draws its own speech, so the recognizer
     * may write the numbers as digits.
     */
    @Test
    void textToSpeechReadsTheText() throws IOException {
        Streamed out = new Streamed();
        ChatEngine.Completion done =
                engine.complete(reading("Hello world, one two three.", "male", null), out);
        Media.Audio audio = done.reply().speech().orElseThrow();
        System.out.printf("tts: %d parts, %.2f s%n", out.chunks.size(), seconds(audio));
        assertTrue(seconds(audio) > 1, "speech");
        assertArrayEquals(audio.pcm(), out.audio().pcm(), "the parts are the waveform");
        assertEquals("", done.reply().text(), "audio alone");
        assertTrue(caption(done).isEmpty(), "speech alone has no transcript lane");
        String heard = hear(audio).replace(",", "").replace(".", "");
        System.out.println("heard: " + heard);
        assertTrue(heard.contains("hello world"), heard);
        assertTrue(
                heard.contains("one two three") || heard.contains("123") || heard.contains("1 2 3"),
                heard);

        // a sink that cancels after a part ends the speech there
        List<Media> first = new ArrayList<>();
        engine.complete(
                reading("Hello world, one two three.", "male", null),
                new ChatEngine.ReplySink() {
                    @Override
                    public void onMedia(Media part) {
                        first.add(part);
                    }

                    @Override
                    public boolean cancelled() {
                        return !first.isEmpty();
                    }
                });
        assertEquals(1, first.size(), "stopped after one part");
    }

    /** A seed makes a reading reproducible, sample for sample; another seed is another take. */
    @Test
    void aSeededReadingIsReproducible() {
        String text = "The quick brown fox jumps over the lazy dog.";
        read(text, 1L); // warm: cold JIT code differs by an LSB
        Media.Audio first = read(text, 7L);
        assertArrayEquals(first.pcm(), read(text, 7L).pcm(), "same seed");
        assertTrue(!Arrays.equals(first.pcm(), read(text, 8L).pcm()), "another seed, another take");
    }

    @Test
    void voiceChatSaysWhatItWrites() throws IOException {
        Streamed out = new Streamed();
        ChatEngine.Completion reply = engine.complete(chat(List.of(france())), out);
        String text = reply.reply().text();
        String heard = hear(reply.reply().speech().orElseThrow());
        System.out.println("wrote: " + text + "\nheard: " + heard);
        assertTrue(text.contains("Paris"), text);
        assertTrue(heard.contains("paris"), heard);
        assertEquals(text, out.text.toString(), "streamed = the reply's text");
        // the transcript lane, the words the speech spoke, captions the speech
        assertEquals(Optional.of(text), caption(reply), "the transcript is the words");
    }

    /**
     * A language is asked for in the conversation's own words, and the reply speaks it: French,
     * written and heard back as French.
     */
    @Test
    void aLanguageIsAskedInTheConversationsOwnWords() throws IOException {
        Message reply =
                engine.complete(
                                chat(
                                        List.of(
                                                Message.system("Always answer in French."),
                                                Message.user(
                                                        "Tell me a fun fact about octopuses."))))
                        .reply();
        String heard = hear(reply.speech().orElseThrow(), "French");
        System.out.println("wrote: " + reply.text() + "\nheard: " + heard);
        assertTrue(reply.text().matches("(?s).*\\b(les|est|ont|sont|des)\\b.*"), reply.text());
        assertTrue(heard.matches("(?s).*\\b(les|est|ont|sont|des)\\b.*"), heard);
    }

    /**
     * The request decides, never the prompt: with text and audio allowed the reply speaks, even
     * under a system turn that asks for text, and a request allowing an output this model lacks
     * still answers in what it has.
     */
    @Test
    void textAndAudioAlwaysSpeak() throws IOException {
        ChatEngine.Completion spoken =
                engine.complete(chat(List.of(Message.system("Respond in text."), france())));
        String text = spoken.reply().text();
        String heard = hear(spoken.reply().speech().orElseThrow());
        System.out.println("wrote: " + text + "\nheard: " + heard);
        assertTrue(text.contains("Paris"), text);
        assertTrue(heard.contains("paris"), heard);
        assertEquals(Optional.of(text), caption(spoken), "the transcript is the words");
        ChatEngine.Completion drawn =
                engine.complete(
                        ChatEngine.Request.builder(List.of(france()), SEEDED)
                                .maxOutputTokens(64)
                                .output(Modality.TEXT, Modality.IMAGE)
                                .build());
        assertTrue(drawn.reply().text().contains("Paris"), drawn.reply().text());
    }

    /**
     * A spoken turn replays as the positions the live reply took - frames as their rows, keyed by
     * their codes - so the next turn restores the whole of it from the cache instead of
     * recomputing, and the model reads back what it said.
     */
    @Test
    void aSpokenTurnReplaysFromTheCache() {
        ChatEngine.Completion first =
                engine.complete(chat(List.of(france())), ChatEngine.ReplySink.NONE);
        Verbatim turn = first.reply().verbatim();
        ChatEngine.Completion second =
                engine.complete(
                        chat(List.of(france(), first.reply(), Message.user("And of Germany?"))),
                        ChatEngine.ReplySink.NONE);
        int live = first.promptTokens() + first.result().tokens().length;
        System.out.printf(
                "turn 1: %d positions (%d frames); turn 2 restored %d of %d%n",
                turn.positions().length(),
                turn.frames(),
                second.restoredTokens(),
                second.promptTokens());
        assertTrue(second.restoredTokens() >= live, "the spoken turn came from the cache");
        assertTrue(second.reply().text().contains("Berlin"), second.reply().text());
    }

    /**
     * A reply may speak again after its speech ends, as LiquidAI's reference and llama.cpp allow:
     * past the end frame an opener starts a fresh stretch, and the lanes run as they did. That its
     * codec starts afresh is the parity test's: a second stretch decodes like the first.
     */
    @Test
    void aReplySpeaksAgainAfterItsSpeechEnds() {
        Lfm3 model = (Lfm3) engine.loaded().model();
        try (Lfm3.State state = model.newState(512, 64)) {
            model.ingest(state, Batch.prefill(Lfm3SpeechParityTest.ttsPrompt(model, "Hi.")));
            try (SpeechOutput.Speaker speaker = model.speaker(state, false, new Random(1))) {
                int frames = 0;
                while (!speaker.textNext() && frames++ < 200)
                    model.ingest(state, speaker.frame().row());
                assertTrue(speaker.textNext(), "the speech ended in " + frames + " frames");
                int mixed = model.opener(true);
                assertEquals(SpeechOutput.Lane.CONTROL, speaker.text(mixed), "reopened");
                model.ingest(state, Batch.step(mixed));
                int[] words =
                        engine.loaded().tokenizer().encode("one two three four five six").toArray();
                for (int i = 0; i < 6; i++) {
                    assertTrue(speaker.textNext(), "a text slot");
                    assertEquals(SpeechOutput.Lane.TEXT, speaker.text(words[i % words.length]));
                    model.ingest(state, Batch.step(words[i % words.length]));
                }
                assertTrue(!speaker.textNext(), "then frames");
                assertEquals(Lfm3Speech.CODEBOOKS, speaker.frame().codes().length, "a frame");
            }
        }
    }

    @Test
    void speechRefusesWhatItCannotHonor() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChatEngine.Request.builder(List.of(france()), SEEDED)
                                .tools(List.of(new Tool("weather", Map.of())))
                                .output(Modality.TEXT, Modality.AUDIO)
                                .build());
        assertThrows(
                UnsupportedOperationException.class,
                () ->
                        engine.complete(
                                ChatEngine.Request.builder(List.of(france()), SEEDED)
                                        .thinking(true)
                                        .output(Modality.AUDIO)
                                        .build()));
        assertThrows(
                IllegalArgumentException.class,
                () -> ChatEngine.Request.builder(List.of(france()), SEEDED).output().build());
        assertThrows(
                UnsupportedOperationException.class,
                () ->
                        engine.complete(
                                ChatEngine.Request.builder(List.of(france()), SEEDED)
                                        .output(Modality.IMAGE)
                                        .build()));
        assertThrows(
                IllegalArgumentException.class,
                () -> Lfm3ChatTemplate.ttsInstruction("en", "robot"));
    }

    private static ChatEngine.Request chat(List<Message> messages) {
        return ChatEngine.Request.builder(messages, SEEDED)
                .maxOutputTokens(400)
                .output(Modality.TEXT, Modality.AUDIO)
                .build();
    }

    /** {@code text} read aloud in English: the family's instruction, answered in audio alone. */
    private static ChatEngine.Request reading(String text, String voice, Long seed) {
        String instruction = Lfm3ChatTemplate.ttsInstruction("en", voice);
        return ChatEngine.Request.builder(
                        List.of(Message.system(instruction), Message.user(text)),
                        engine.loaded().samplingDefaults().resolve(null, null, null, null, seed))
                .output(Modality.AUDIO)
                .build();
    }

    private static Media.Audio read(String text, Long seed) {
        return engine.complete(reading(text, "female", seed)).reply().speech().orElseThrow();
    }

    private static Message france() {
        return Message.user("What is the capital of France?");
    }

    private static Optional<String> caption(ChatEngine.Completion completion) {
        for (Content part : completion.reply().content())
            if (part instanceof Content.Media m && m.media() instanceof Media.Audio)
                return m.caption();
        throw new AssertionError("no speech in " + completion.reply());
    }

    /** What Omni's own ears make of {@code speech}, at 16 kHz as its audio input takes it. */
    private static String hear(Media.Audio speech) throws IOException {
        return hear(speech, "English");
    }

    /** What the model's own speech recognition hears in {@code speech}, in {@code language}. */
    private static String hear(Media.Audio speech, String language) throws IOException {
        Media.Audio heard = AudioCodec.decode(AudioCodec.wav(speech));
        return engine.complete(
                        ChatEngine.Request.builder(
                                        List.of(
                                                Message.system("Perform ASR in " + language + "."),
                                                new Message(
                                                        Role.USER,
                                                        List.of(new Content.Media(heard)))),
                                        SEEDED)
                                .maxOutputTokens(64)
                                .build())
                .reply()
                .text()
                .toLowerCase();
    }

    private static double seconds(Media.Audio audio) {
        return audio.pcm().length / (double) audio.sampleRate();
    }

    /** A streaming caller: the transcript's deltas and the speech's chunks as they arrive. */
    private static final class Streamed implements ChatEngine.ReplySink {
        final StringBuilder text = new StringBuilder();
        final List<Media.Audio> chunks = new ArrayList<>();

        @Override
        public void on(ChatEngine.Delta delta) {
            text.append(delta.text());
        }

        @Override
        public void onMedia(Media part) {
            chunks.add((Media.Audio) part);
        }

        Media.Audio audio() {
            return Media.Audio.concat(chunks);
        }
    }
}
