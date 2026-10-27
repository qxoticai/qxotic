package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Speech;
import com.qxotic.jinfer.chat.SpokenTurn;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
    private static final Message VOICE_CHAT =
            Message.system("Respond with interleaved text and audio.");
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
     * Read aloud, streamed in parts, and whole: the parts are the waveform, and the words survive.
     * Each call draws its own speech, so the recognizer may write the numbers as digits.
     */
    @Test
    void textToSpeechReadsTheText() throws IOException {
        List<Media.Audio> parts = new ArrayList<>();
        engine.speak("Hello world, one two three.", "en", "male", parts::add);
        Media.Audio audio = Media.Audio.concat(parts);
        System.out.printf("tts: %d parts, %.2f s%n", parts.size(), seconds(audio));
        assertTrue(seconds(audio) > 1, "speech");
        String heard = hear(audio).replace(",", "").replace(".", "");
        System.out.println("heard: " + heard);
        assertTrue(heard.contains("hello world"), heard);
        assertTrue(
                heard.contains("one two three") || heard.contains("123") || heard.contains("1 2 3"),
                heard);

        // a sink that stops after a part ends the speech there
        List<Media.Audio> first = new ArrayList<>();
        engine.speak("Hello world, one two three.", "en", "male", part -> !first.add(part));
        assertEquals(1, first.size(), "stopped after one part");
        assertTrue(seconds(engine.speak("Good morning.", null, null)) > 0.3, "the whole waveform");
    }

    @Test
    void voiceChatSaysWhatItWrites() throws IOException {
        Streamed out = new Streamed();
        ChatEngine.Completion reply = engine.complete(chat(List.of(VOICE_CHAT, france())), out);
        String text = reply.reply().text();
        String heard = hear(reply.reply().speech().orElseThrow());
        System.out.println("wrote: " + text + "\nheard: " + heard);
        assertTrue(text.contains("Paris"), text);
        assertTrue(heard.contains("paris"), heard);
        assertEquals(text, out.text.toString(), "streamed = the reply's text");
    }

    /**
     * A spoken turn replays as the positions the live reply took - frames as their rows, keyed by
     * their codes - so the next turn restores the whole of it from the cache instead of
     * recomputing, and the model reads back what it said.
     */
    @Test
    void aSpokenTurnReplaysFromTheCache() {
        ChatEngine.Completion first =
                engine.complete(chat(List.of(VOICE_CHAT, france())), ChatEngine.ReplySink.NONE);
        SpokenTurn turn = speech(first).turn();
        ChatEngine.Completion second =
                engine.complete(
                        chat(
                                List.of(
                                        VOICE_CHAT,
                                        france(),
                                        first.reply(),
                                        Message.user("And of Germany?"))),
                        ChatEngine.ReplySink.NONE);
        int live = first.promptTokens() + turn.positions().length() - 1;
        System.out.printf(
                "turn 1: %d positions (%d frames); turn 2 restored %d of %d%n",
                turn.positions().length(),
                turn.frames().size(),
                second.restoredTokens(),
                second.promptTokens());
        assertTrue(second.restoredTokens() >= live, "the spoken turn came from the cache");
        assertTrue(second.reply().text().contains("Berlin"), second.reply().text());
    }

    @Test
    void speechRefusesWhatItCannotHonor() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChatEngine.Request.builder(List.of(france()), SEEDED)
                                .tools(List.of(new Tool("weather", Map.of())))
                                .speech(Speech.withTranscript())
                                .build());
        assertThrows(
                UnsupportedOperationException.class,
                () ->
                        engine.complete(
                                ChatEngine.Request.builder(List.of(france()), SEEDED)
                                        .thinking(true)
                                        .speech(Speech.only())
                                        .build()));
        assertThrows(IllegalArgumentException.class, () -> engine.speak("Hi.", "en", "robot"));
    }

    private static ChatEngine.Request chat(List<Message> messages) {
        return ChatEngine.Request.builder(messages, SEEDED)
                .maxOutputTokens(400)
                .speech(Speech.withTranscript())
                .build();
    }

    private static Message france() {
        return Message.user("What is the capital of France?");
    }

    private static Content.Speech speech(ChatEngine.Completion completion) {
        for (Content part : completion.reply().content())
            if (part instanceof Content.Speech speech) return speech;
        throw new AssertionError("no speech in " + completion.reply());
    }

    /** What Omni's own ears make of {@code speech}, at 16 kHz as its audio input takes it. */
    private static String hear(Media.Audio speech) throws IOException {
        Media.Audio heard = AudioCodec.decode(AudioCodec.wav(speech));
        return engine.complete(
                        ChatEngine.Request.builder(
                                        List.of(
                                                Message.system("Perform ASR in English."),
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
        public void onSpeech(Media.Audio speech) {
            chunks.add(speech);
        }

        Media.Audio audio() {
            return Media.Audio.concat(chunks);
        }
    }
}
