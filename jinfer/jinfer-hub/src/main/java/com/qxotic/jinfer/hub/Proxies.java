package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.ProxySelector;
import java.net.SocketAddress;
import java.net.URI;
import java.util.List;
import java.util.Locale;

/**
 * {@code HTTP_PROXY} / {@code HTTPS_PROXY} / {@code NO_PROXY}, which Java does not read on its own.
 *
 * <p>A {@link java.net.http.HttpClient} built without a selector uses NO proxy at all, and even
 * {@link ProxySelector#getDefault()} only honors the {@code -Dhttps.proxyHost} system properties.
 * Every other model downloader (curl, the HuggingFace clients, ollama) follows the environment
 * variables, so a corporate laptop or a machine behind a regional mirror expects them to work here
 * too. Without this the failure is a connect timeout with nothing to suggest.
 *
 * <p>The environment wins when it is set; otherwise the JVM's own default selector applies, so
 * {@code -Dhttps.proxyHost} keeps working for callers who configure Java that way.
 */
final class Proxies {

    private Proxies() {}

    static ProxySelector selector() {
        String https = env("HTTPS_PROXY", "https_proxy");
        String http = env("HTTP_PROXY", "http_proxy");
        if (https == null && http == null) {
            return ProxySelector.getDefault();
        }
        return new EnvProxySelector(http, https, env("NO_PROXY", "no_proxy"));
    }

    private static String env(String upper, String lower) {
        String value = System.getenv(upper);
        if (value == null || value.isBlank()) {
            value = System.getenv(lower);
        }
        return value == null || value.isBlank() ? null : value.strip();
    }

    private static final class EnvProxySelector extends ProxySelector {

        private final Proxy http;
        private final Proxy https;
        private final List<String> noProxy;

        EnvProxySelector(String httpUrl, String httpsUrl, String noProxy) {
            this.http = parse(httpUrl);
            this.https = parse(httpsUrl);
            this.noProxy =
                    noProxy == null
                            ? List.of()
                            : java.util.Arrays.stream(noProxy.split(","))
                                    .map(s -> s.strip().toLowerCase(Locale.ROOT))
                                    .filter(s -> !s.isEmpty())
                                    .toList();
        }

        private static Proxy parse(String url) {
            if (url == null) {
                return null;
            }
            try {
                URI uri = URI.create(url.contains("://") ? url : "http://" + url);
                int port = uri.getPort() > 0 ? uri.getPort() : 8080;
                return new Proxy(Proxy.Type.HTTP, new InetSocketAddress(uri.getHost(), port));
            } catch (RuntimeException malformed) {
                return null; // a broken proxy variable must not make direct access impossible
            }
        }

        @Override
        public List<Proxy> select(URI uri) {
            String host = uri.getHost() == null ? "" : uri.getHost().toLowerCase(Locale.ROOT);
            for (String exempt : noProxy) {
                // curl's rule: a bare name matches the host and any subdomain of it
                if (exempt.equals("*") || host.equals(exempt) || host.endsWith("." + exempt)) {
                    return List.of(Proxy.NO_PROXY);
                }
            }
            Proxy chosen = "https".equalsIgnoreCase(uri.getScheme()) ? https : http;
            return List.of(chosen == null ? Proxy.NO_PROXY : chosen);
        }

        @Override
        public void connectFailed(URI uri, SocketAddress address, IOException failure) {
            // nothing to fail over to: the caller sees the IOException with the address in it
        }
    }
}
