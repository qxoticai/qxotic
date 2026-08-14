package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.Set;
import org.junit.jupiter.api.Test;

class ServerConfigTest {

    @Test
    void localDefaultsAreLoopbackAndBrowserFriendly() {
        ServerConfig config = ServerConfig.local(0);
        assertTrue(config.bind().getAddress().isLoopbackAddress());
        assertTrue(config.access().allowedOrigins().contains("*"));
    }

    @Test
    void accessDefensivelyCopiesOrigins() {
        var source = new java.util.HashSet<>(Set.of("https://example.test"));
        var access = new ServerConfig.Access("token", source);
        source.clear();
        assertEquals(Set.of("https://example.test"), access.allowedOrigins());
    }

    @Test
    void badLimitsFailAtConstruction() {
        ServerConfig.Limits d = ServerConfig.Limits.DEFAULTS;
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ServerConfig.Limits(
                                0,
                                d.queueCapacity(),
                                d.maxBodyBytes(),
                                d.grammar(),
                                Duration.ofSeconds(1),
                                Duration.ZERO,
                                Duration.ZERO));
    }
}
