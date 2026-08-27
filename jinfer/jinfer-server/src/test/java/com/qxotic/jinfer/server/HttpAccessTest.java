package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.sun.net.httpserver.HttpServer;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Set;
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
}
