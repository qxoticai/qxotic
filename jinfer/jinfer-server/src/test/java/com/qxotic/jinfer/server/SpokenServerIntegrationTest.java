package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * LFM3-Omni's spoken replies over HTTP, in OpenAI's shape: {@code modalities} and {@code audio} on
 * the request, {@code message.audio} with the speech, its transcript and the id the turn is kept
 * under, the id echoed back as the assistant turn in history, {@code delta.audio} when streamed.
 * The model's own ears, through the same server, are the judge of what it said.
 */
@Tag("integration")
class SpokenServerIntegrationTest {

    static final String OMNI = "hf.co/LiquidAI/LFM3-Omni-2B-GGUF:Q8_0";
    static final String OMNI_MMPROJ =
            "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf";

    /** The voice's rate; a stream's pcm16 chunks name no rate, as OpenAI's do not. */
    private static final int VOICE_RATE = 24_000;

    private static final Map<String, Object> FRANCE =
            Map.of("role", "user", "content", "What is the capital of France?");

    static ChatEngine engine;
    static Server.Running server;
    static String base;
    static final HttpClient client = HttpClient.newHttpClient();

    @BeforeAll
    static void start() throws Exception {
        engine =
                new ChatEngine(
                        TestModels.require(OMNI),
                        Map.of("media", TestModels.require(OMNI_MMPROJ)),
                        PromptCache.Options.DEFAULTS.withContextCapacity(4096));
        server = Server.start(engine, ServerConfig.local(0));
        base = "http://127.0.0.1:" + server.address().getPort();
    }

    @AfterAll
    static void stop() {
        if (server != null) server.close();
        if (engine != null) engine.close();
    }

    @Test
    void theModelCardSaysItSpeaks() throws Exception {
        String models = get(base + "/v1/models").body();
        assertTrue(models.contains("\"output_modalities\":[\"text\",\"audio\"]"), models);
    }

    /**
     * Asked in OpenAI's shape, the reply speaks: {@code message.audio} carries the WAV, the
     * transcript, an id and its expiry, and {@code content} is null as OpenAI's is. The id echoed
     * as the assistant turn replays the spoken turn in the next request; an unknown id is a 400.
     */
    @Test
    void aSpokenReplyIsKeptUnderItsAudioIdAndHeardBack() throws Exception {
        HttpResponse<String> r = post(base + "/v1/chat/completions", spoken("wav", false));
        assertEquals(200, r.statusCode(), r.body());
        Map<String, Object> message = message(r.body());
        assertNull(message.get("content"), "the words are the transcript: " + message);
        Map<String, Object> audio = Values.asObject(message.get("audio"), "audio");
        String id = Values.stringValue(audio.get("id"), "");
        assertTrue(id.startsWith("audio_"), id);
        long now = System.currentTimeMillis() / 1000;
        assertTrue(
                Values.longValue(audio.get("expires_at"), "expires_at", 0) > now, audio.toString());
        String transcript = Values.stringValue(audio.get("transcript"), "");
        assertTrue(transcript.contains("Paris"), transcript);
        byte[] wav = Base64.getDecoder().decode(Values.stringValue(audio.get("data"), ""));
        Media.Audio clip = AudioCodec.decode(wav);
        assertTrue(seconds(clip) > 0.5, "speech");
        String heard = hear(wav);
        System.out.printf("wrote: %s%nheard: %s (%.2f s)%n", transcript, heard, seconds(clip));
        assertTrue(heard.contains("paris"), heard);

        HttpResponse<String> next =
                post(base + "/v1/chat/completions", followUp(id, "And of Germany?"));
        assertEquals(200, next.statusCode(), next.body());
        assertTrue(words(message(next.body())).contains("Berlin"), next.body());

        HttpResponse<String> unknown =
                post(base + "/v1/chat/completions", followUp("audio_nobody", "And of Germany?"));
        assertEquals(400, unknown.statusCode(), unknown.body());
        assertTrue(unknown.body().contains("unknown or expired"), unknown.body());
    }

    /**
     * Streamed, the reply is {@code delta.audio} chunks under one id: the transcript and the pcm16
     * data as they come, the expiry last; no plain content delta when it spoke. The streamed id is
     * a kept turn like a call's, and the chunks joined are speech the model hears.
     */
    @Test
    void aStreamedSpokenReplyIsTranscriptAndPcm16ChunksUnderOneId() throws Exception {
        HttpResponse<String> r = post(base + "/v1/chat/completions", spoken("pcm16", true));
        assertEquals(200, r.statusCode(), r.body());
        assertTrue(r.body().contains("data: [DONE]"), r.body());
        String id = null;
        Long expiresAt = null;
        StringBuilder transcript = new StringBuilder();
        ByteArrayOutputStream pcm = new ByteArrayOutputStream();
        int audioChunks = 0;
        for (Map<String, Object> event : eventPayloads(r.body())) {
            Map<String, Object> delta = delta(event);
            assertNull(delta.get("content"), "a spoken reply streams no content: " + delta);
            if (delta.get("audio") == null) continue;
            Map<String, Object> audio = Values.asObject(delta.get("audio"), "audio");
            String chunkId = Values.stringValue(audio.get("id"), "");
            if (id == null) id = chunkId;
            assertEquals(id, chunkId, "one id per reply");
            if (audio.get("transcript") != null)
                transcript.append(Values.stringValue(audio.get("transcript"), ""));
            if (audio.get("data") != null) {
                pcm.writeBytes(
                        Base64.getDecoder().decode(Values.stringValue(audio.get("data"), "")));
                audioChunks++;
            }
            if (audio.get("expires_at") != null)
                expiresAt = Values.longValue(audio.get("expires_at"), "expires_at", 0);
        }
        assertNotNull(id, "no audio chunk: " + r.body());
        assertNotNull(expiresAt, "the expiry closes the stream");
        assertTrue(transcript.toString().contains("Paris"), transcript.toString());
        Media.Audio clip = pcm16(pcm.toByteArray());
        String heard = hear(AudioCodec.wav(clip));
        System.out.printf(
                "streamed %d chunks, %.2f s; wrote: %s%nheard: %s%n",
                audioChunks, seconds(clip), transcript, heard);
        assertTrue(seconds(clip) > 0.5, "speech");
        assertTrue(heard.contains("paris"), heard);

        // the streamed id is history too
        HttpResponse<String> next =
                post(
                        base + "/v1/chat/completions",
                        followUp(id, "Repeat the city you just named."));
        assertEquals(200, next.statusCode(), next.body());
        assertTrue(words(message(next.body())).contains("Paris"), next.body());
    }

    /** The engine's refusals are 400s in words; the shape's own rules too. */
    @Test
    void whatSpeechCannotHonorIsRefusedInWords() throws Exception {
        Map<String, Object> tool =
                Map.of(
                        "type",
                        "function",
                        "function",
                        Map.of("name", "weather", "description", "weather"));
        HttpResponse<String> tools =
                post(
                        base + "/v1/chat/completions",
                        JsonCodec.stringify(
                                Map.of(
                                        "messages",
                                        List.of(FRANCE),
                                        "modalities",
                                        List.of("text", "audio"),
                                        "audio",
                                        Map.of("format", "wav"),
                                        "tools",
                                        List.of(tool),
                                        "max_tokens",
                                        8)));
        assertEquals(400, tools.statusCode(), tools.body());
        assertTrue(tools.body().contains("cannot offer tools"), tools.body());
        HttpResponse<String> alone =
                post(
                        base + "/v1/chat/completions",
                        JsonCodec.stringify(
                                Map.of(
                                        "messages",
                                        List.of(FRANCE),
                                        "modalities",
                                        List.of("audio"),
                                        "audio",
                                        Map.of("format", "wav"),
                                        "max_tokens",
                                        8)));
        assertEquals(400, alone.statusCode(), alone.body());
        assertTrue(alone.body().contains("modalities must be"), alone.body());
    }

    private static String spoken(String format, boolean stream) {
        return JsonCodec.stringify(
                Map.of(
                        "messages",
                        List.of(FRANCE),
                        "modalities",
                        List.of("text", "audio"),
                        "audio",
                        Map.of("format", format),
                        "stream",
                        stream,
                        "temperature",
                        0,
                        "seed",
                        42,
                        "max_tokens",
                        400));
    }

    /**
     * The conversation continued past the spoken turn kept under {@code id}, audio still allowed:
     * the next reply speaks too, its words the transcript.
     */
    private static String followUp(String id, String question) {
        return JsonCodec.stringify(
                Map.of(
                        "messages",
                        List.of(
                                FRANCE,
                                Map.of("role", "assistant", "audio", Map.of("id", id)),
                                Map.of("role", "user", "content", question)),
                        "modalities",
                        List.of("text", "audio"),
                        "audio",
                        Map.of("format", "wav"),
                        "temperature",
                        0,
                        "seed",
                        42,
                        "max_tokens",
                        400));
    }

    /** A reply's words: its content, or its transcript when it spoke. */
    private static String words(Map<String, Object> message) {
        if (message.get("content") != null) return Values.stringValue(message.get("content"), "");
        return Values.stringValue(
                Values.asObject(message.get("audio"), "audio").get("transcript"), "");
    }

    /** What Omni's own ears make of {@code wav}, through the same server. */
    private static String hear(byte[] wav) throws Exception {
        Map<String, Object> part =
                Map.of(
                        "type",
                        "input_audio",
                        "input_audio",
                        Map.of("data", Base64.getEncoder().encodeToString(wav), "format", "wav"));
        HttpResponse<String> r =
                post(
                        base + "/v1/chat/completions",
                        JsonCodec.stringify(
                                Map.of(
                                        "messages",
                                        List.of(
                                                Map.of(
                                                        "role",
                                                        "system",
                                                        "content",
                                                        "Perform ASR in English."),
                                                Map.of("role", "user", "content", List.of(part))),
                                        "temperature",
                                        0,
                                        "max_tokens",
                                        64)));
        assertEquals(200, r.statusCode(), r.body());
        return Values.stringValue(message(r.body()).get("content"), "").toLowerCase();
    }

    private static Media.Audio pcm16(byte[] bytes) {
        ByteBuffer in = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] pcm = new float[bytes.length / 2];
        for (int i = 0; i < pcm.length; i++) pcm[i] = in.getShort() / 32768f;
        return new Media.Audio(pcm, VOICE_RATE, 1);
    }

    private static double seconds(Media.Audio audio) {
        return audio.pcm().length / (double) audio.sampleRate();
    }

    private static Map<String, Object> message(String body) {
        return choice(Values.asObject(JsonCodec.parse(body), "reply"), "message");
    }

    private static Map<String, Object> delta(Map<String, Object> event) {
        return choice(event, "delta");
    }

    private static Map<String, Object> choice(Map<String, Object> reply, String field) {
        List<Object> choices = Values.asArray(reply.get("choices"), "choices");
        return Values.asObject(Values.asObject(choices.get(0), "choice").get(field), field);
    }

    private static List<Map<String, Object>> eventPayloads(String body) {
        List<Map<String, Object>> events = new ArrayList<>();
        for (String line : body.split("\n")) {
            if (line.startsWith("data: ") && !line.equals("data: [DONE]"))
                events.add(Values.asObject(JsonCodec.parse(line.substring(6)), "event"));
        }
        return events;
    }

    private static HttpResponse<String> post(String uri, String body) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(uri))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private static HttpResponse<String> get(String uri) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(uri)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }
}
