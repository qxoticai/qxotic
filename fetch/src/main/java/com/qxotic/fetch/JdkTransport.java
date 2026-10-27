package com.qxotic.fetch;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/** The JDK's client. Public with a no-argument constructor because it is a service provider. */
public final class JdkTransport implements Transport {
    /**
     * HTTP/1.1 on purpose: HTTP/2 would multiplex every ranged GET onto ONE connection, and it is
     * the connections, not the requests, that a parallel download multiplies. Proxies come from the
     * environment (HTTPS_PROXY, NO_PROXY), as curl reads them, so the two transports agree.
     */
    private static final HttpClient CLIENT =
            HttpClient.newBuilder()
                    .version(HttpClient.Version.HTTP_1_1)
                    .followRedirects(HttpClient.Redirect.NEVER)
                    .proxy(Proxies.selector())
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
            public List<String> headers(String name) {
                return response.headers().allValues(name);
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
