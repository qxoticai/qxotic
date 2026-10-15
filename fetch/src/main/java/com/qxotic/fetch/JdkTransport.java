package com.qxotic.fetch;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/** The JDK's client. Public with a no-argument constructor because it is a service provider. */
public final class JdkTransport implements Transport {
    private static final HttpClient CLIENT =
            HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.NEVER)
                    .connectTimeout(Duration.ofSeconds(30))
                    .build();

    @Override
    public Response get(URI url, Map<String, String> headers, Duration timeout) throws IOException {
        HttpRequest.Builder request = HttpRequest.newBuilder(url).timeout(timeout).GET();
        headers.forEach(request::header);
        HttpResponse<InputStream> response;
        try {
            response = CLIENT.send(request.build(), HttpResponse.BodyHandlers.ofInputStream());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while fetching " + url, e);
        }
        return new Response() {
            @Override
            public int status() {
                return response.statusCode();
            }

            @Override
            public String header(String name) {
                return response.headers().firstValue(name).orElse(null);
            }

            @Override
            public InputStream body() {
                return response.body();
            }

            @Override
            public void close() throws IOException {
                response.body().close();
            }
        };
    }
}
