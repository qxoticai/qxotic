package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class HttpAccessTest {

    private HttpServer server;

    @AfterEach
    void stop() {
        if (server != null) server.stop(0);
    }

    @Test
    void bearerAndOriginAreEnforcedBeforeTheRoute() throws Exception {
        ServerConfig.Access access =
                new ServerConfig.Access("secret", Set.of("https://allowed.test"));
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    if (!Http.preamble(exchange, access)) Http.sendJson(exchange, 200, "ok");
                });
        server.start();
        URI uri = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/");
        HttpClient client = HttpClient.newHttpClient();

        assertEquals(
                401,
                client.send(
                                HttpRequest.newBuilder(uri).GET().build(),
                                HttpResponse.BodyHandlers.ofString())
                        .statusCode());
        assertEquals(
                403,
                client.send(
                                HttpRequest.newBuilder(uri)
                                        .header("Authorization", "Bearer secret")
                                        .header("Origin", "https://denied.test")
                                        .GET()
                                        .build(),
                                HttpResponse.BodyHandlers.ofString())
                        .statusCode());
        HttpResponse<String> accepted =
                client.send(
                        HttpRequest.newBuilder(uri)
                                .header("Authorization", "Bearer secret")
                                .header("Origin", "https://allowed.test")
                                .GET()
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        assertEquals(200, accepted.statusCode());
        // the scheme is case-insensitive (RFC 7235): a proxy that lowercases it must not lock out
        assertEquals(
                200,
                client.send(
                                HttpRequest.newBuilder(uri)
                                        .header("Authorization", "bearer secret")
                                        .header("Origin", "https://allowed.test")
                                        .GET()
                                        .build(),
                                HttpResponse.BodyHandlers.ofString())
                        .statusCode());
        assertEquals(
                "https://allowed.test",
                accepted.headers().firstValue("Access-Control-Allow-Origin").orElseThrow());
    }

    @Test
    void stalledUploadReleasesItsAdmissionPermit() throws Exception {
        Semaphore admissions = new Semaphore(1);
        CountDownLatch finished = new CountDownLatch(1);
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext(
                "/",
                Server.gated(
                        exchange -> {
                            try {
                                byte[] body =
                                        Http.readBody(exchange, 64, Duration.ofMillis(100));
                                if (body != null) Http.sendJson(exchange, 200, "ok");
                            } finally {
                                finished.countDown();
                            }
                        },
                        admissions,
                        1));
        server.start();
        int port = server.getAddress().getPort();

        try (Socket stalled = new Socket(InetAddress.getLoopbackAddress(), port)) {
            stalled.getOutputStream()
                    .write(
                            ("POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\n\r\n{")
                                    .getBytes(StandardCharsets.US_ASCII));
            stalled.getOutputStream().flush();

            assertTrue(finished.await(2, TimeUnit.SECONDS), "stalled body read did not expire");
            assertEquals(1, admissions.availablePermits());
            assertEquals(
                    200,
                    HttpClient.newHttpClient()
                            .send(
                                    HttpRequest.newBuilder(
                                                    URI.create(
                                                            "http://127.0.0.1:" + port + "/"))
                                            .POST(HttpRequest.BodyPublishers.ofString("{}"))
                                            .build(),
                                    HttpResponse.BodyHandlers.ofString())
                            .statusCode());
        }
    }
}
