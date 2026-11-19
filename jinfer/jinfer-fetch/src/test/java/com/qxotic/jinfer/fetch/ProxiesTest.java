package com.qxotic.jinfer.fetch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.URI;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ProxiesTest {

    @Test
    void noProxyAcceptsTheLeadingDotSpelling() {
        // NO_PROXY=.corp.internal is the common form; it used to match nothing ("..corp.internal")
        var selector =
                new Proxies.EnvProxySelector(
                        null, "http://proxy:3128", ".corp.internal, localhost");
        assertEquals(
                List.of(Proxy.NO_PROXY),
                selector.select(URI.create("https://hub.corp.internal/x")));
        assertEquals(
                List.of(Proxy.NO_PROXY), selector.select(URI.create("https://corp.internal/x")));
        assertEquals(
                List.of(Proxy.NO_PROXY), selector.select(URI.create("http://localhost:8080/")));
        assertNotEquals(
                List.of(Proxy.NO_PROXY), selector.select(URI.create("https://huggingface.co/")));
    }

    @Test
    void theVariablesAreReadAsCurlReadsThem() {
        URI http = URI.create("http://mirror.example/x");
        URI https = URI.create("https://hf.co/x");
        // uppercase HTTP_PROXY is ignored, as curl ignores it (httpoxy): a native image's curl
        // would go direct, so the JVM build must too
        var upperOnly =
                Proxies.selector(
                        Map.of("HTTP_PROXY", "http://upper:1", "HTTPS_PROXY", "http://upper:2")
                                ::get);
        assertEquals(List.of(Proxy.NO_PROXY), upperOnly.select(http));
        assertEquals(proxy("upper", 2), upperOnly.select(https));
        // lowercase first, as curl reads it
        var both =
                Proxies.selector(
                        Map.of(
                                        "http_proxy", "http://lower:1",
                                        "https_proxy", "http://lower:2",
                                        "HTTPS_PROXY", "http://upper:2",
                                        "no_proxy", "hf.co",
                                        "NO_PROXY", "mirror.example")
                                ::get);
        assertEquals(proxy("lower", 1), both.select(http));
        assertEquals(List.of(Proxy.NO_PROXY), both.select(https));
    }

    private static List<Proxy> proxy(String host, int port) {
        return List.of(new Proxy(Proxy.Type.HTTP, new InetSocketAddress(host, port)));
    }
}
