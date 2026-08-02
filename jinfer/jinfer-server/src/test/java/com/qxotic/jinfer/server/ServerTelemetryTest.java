package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.sun.net.httpserver.HttpServer;
import java.lang.foreign.Arena;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Test;

/**
 * The server runs {@link com.qxotic.jinfer.llm.Generator} directly rather than through {@code
 * ChatEngine}, so it has its OWN generation seam - and a served request emits nothing unless that
 * seam reports it. This pins that, and pins {@code queueTime}, which crosses from the worker thread
 * that dequeued the job to the event that describes it.
 */
class ServerTelemetryTest {

    @Test
    void aServedRequestEmitsAnInferenceEventCarryingItsQueueWait() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        Path jfr = Files.createTempFile("jinfer-server", ".jfr");
        try (Arena arena = Arena.ofShared()) {
            LoadedModel<?> model = Models.load(gguf, 2048, arena);
            LLMOptions options =
                    new LLMOptions(
                            gguf,
                            null,
                            null,
                            false,
                            true,
                            "127.0.0.1",
                            0,
                            1f,
                            0.95f,
                            42L,
                            2048,
                            true,
                            false,
                            true,
                            false,
                            false,
                            false,
                            false,
                            null,
                            false);
            HttpServer server = Server.start(model, options);
            try {
                String base = "http://127.0.0.1:" + server.getAddress().getPort();
                HttpClient client =
                        HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
                try (Recording recording = new Recording()) {
                    recording.enable("jinfer.Inference");
                    recording.start();
                    HttpResponse<String> response =
                            client.send(
                                    HttpRequest.newBuilder(
                                                    URI.create(base + "/v1/chat/completions"))
                                            .header("Content-Type", "application/json")
                                            .POST(
                                                    HttpRequest.BodyPublishers.ofString(
                                                            "{\"model\":\""
                                                                    + gguf.getFileName()
                                                                    + "\",\"messages\":"
                                                                    + "[{\"role\":\"user\","
                                                                    + "\"content\":\"Hi\"}],"
                                                                    + "\"max_tokens\":8}"))
                                            .build(),
                                    HttpResponse.BodyHandlers.ofString());
                    assertEquals(200, response.statusCode(), response.body());
                    recording.stop();
                    recording.dump(jfr);
                }
            } finally {
                server.stop(0);
            }
        }

        List<RecordedEvent> events = events(jfr);
        assertEquals(1, events.size(), "one served request, one event");
        RecordedEvent event = events.get(0);
        assertEquals("chat", event.getString("operation"));
        assertEquals("", event.getString("errorType"));
        assertTrue(event.getString("model").endsWith(".gguf"), event.getString("model"));
        assertTrue(event.getInt("inputTokens") > 0);
        assertTrue(event.getInt("outputTokens") > 0);
        assertTrue(event.getLong("prefillTime") > 0);
        assertTrue(
                event.getLong("queueTime") > 0,
                "queueTime crosses threads via Telemetry; 0 means it never reached the event");
    }

    private static List<RecordedEvent> events(Path jfr) throws Exception {
        try (RecordingFile file = new RecordingFile(jfr)) {
            List<RecordedEvent> found = new java.util.ArrayList<>();
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                if (event.getEventType().getName().equals("jinfer.Inference")) found.add(event);
            }
            return found;
        }
    }
}
