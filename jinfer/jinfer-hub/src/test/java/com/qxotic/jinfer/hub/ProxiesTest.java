package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.net.Proxy;
import java.net.URI;
import java.util.List;
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
}
