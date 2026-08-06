package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.lang.foreign.Arena;
import java.net.http.HttpClient;
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
 * A served request must emit exactly ONE {@code jinfer.Inference} event - the engine emits it, the
 * server must not add its own on top (the double-emission the migration to ChatEngine introduced
 * and this test caught) - and it must carry {@code queueTime}, which crosses from the worker thread
 * that dequeued the job to the event that describes it.
 */
class ServerTelemetryTest {

    @Test
    void aServedRequestEmitsOneInferenceEventCarryingItsQueueWait() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        Path jfr = Files.createTempFile("jinfer-server", ".jfr");
        try (Arena arena = Arena.ofShared()) {
            LoadedModel<?> model = Models.load(gguf, 2048, arena);
            try (Server.Running server = Server.start(model, ServerTestSupport.config(gguf))) {
                HttpClient client =
                        HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
                try (Recording recording = new Recording()) {
                    recording.enable("jinfer.Inference");
                    recording.start();
                    HttpResponse<String> response = chat(client, server, gguf);
                    assertEquals(200, response.statusCode(), response.body());
                    recording.stop();
                    recording.dump(jfr);
                }
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

    /**
     * {@link Server#start} promises each call an independent instance. The counters were static, so
     * two servers in one JVM reported each other's traffic - scrape one, see both.
     */
    @Test
    void twoServersInOneJvmDoNotShareCounters() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        try (Arena arena = Arena.ofShared()) {
            LoadedModel<?> model = Models.load(gguf, 2048, arena);
            try (Server.Running busy = Server.start(model, ServerTestSupport.config(gguf));
                    Server.Running idle = Server.start(model, ServerTestSupport.config(gguf))) {
                HttpClient client =
                        HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
                assertEquals(200, chat(client, busy, gguf).statusCode());

                assertTrue(
                        requestsTotal(client, busy) >= 1,
                        "the server that served it must count it");
                assertEquals(
                        0,
                        requestsTotal(client, idle),
                        "the other server must not see another instance's traffic");
            }
        }
    }

    private static long requestsTotal(HttpClient client, Server.Running server) throws Exception {
        String body =
                ServerTestSupport.get(client, ServerTestSupport.baseUrl(server) + "/metrics")
                        .body();
        return ServerTestSupport.counter(body, "jinfer_requests_total");
    }

    private static HttpResponse<String> chat(HttpClient client, Server.Running server, Path gguf)
            throws Exception {
        return ServerTestSupport.post(
                client,
                ServerTestSupport.baseUrl(server) + "/v1/chat/completions",
                ServerTestSupport.chatBody(gguf.getFileName().toString(), "Hi", 8));
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
