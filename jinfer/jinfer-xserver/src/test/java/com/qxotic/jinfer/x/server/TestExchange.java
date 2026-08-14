package com.qxotic.jinfer.x.server;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpContext;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpPrincipal;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;

/** Minimal in-memory exchange for transport unit tests. */
final class TestExchange extends HttpExchange {

    private final Headers requestHeaders = new Headers();
    private final Headers responseHeaders = new Headers();
    private final Map<String, Object> attributes = new HashMap<>();
    private InputStream request;
    private OutputStream response;
    private int responseCode = -1;
    private boolean closed;

    TestExchange(byte[] request) {
        this(new ByteArrayInputStream(request), new ByteArrayOutputStream());
    }

    TestExchange(InputStream request, OutputStream response) {
        this.request = request;
        this.response = response;
    }

    byte[] responseBytes() {
        return ((ByteArrayOutputStream) response).toByteArray();
    }

    boolean closed() {
        return closed;
    }

    @Override
    public Headers getRequestHeaders() {
        return requestHeaders;
    }

    @Override
    public Headers getResponseHeaders() {
        return responseHeaders;
    }

    @Override
    public URI getRequestURI() {
        return URI.create("/");
    }

    @Override
    public String getRequestMethod() {
        return "POST";
    }

    @Override
    public HttpContext getHttpContext() {
        return null;
    }

    @Override
    public void close() {
        closed = true;
        try {
            request.close();
            response.close();
        } catch (IOException ignored) {
        }
    }

    @Override
    public InputStream getRequestBody() {
        return request;
    }

    @Override
    public OutputStream getResponseBody() {
        return response;
    }

    @Override
    public void sendResponseHeaders(int code, long length) {
        responseCode = code;
    }

    @Override
    public InetSocketAddress getRemoteAddress() {
        return new InetSocketAddress(InetAddress.getLoopbackAddress(), 12345);
    }

    @Override
    public int getResponseCode() {
        return responseCode;
    }

    @Override
    public InetSocketAddress getLocalAddress() {
        return new InetSocketAddress(InetAddress.getLoopbackAddress(), 8080);
    }

    @Override
    public String getProtocol() {
        return "HTTP/1.1";
    }

    @Override
    public Object getAttribute(String name) {
        return attributes.get(name);
    }

    @Override
    public void setAttribute(String name, Object value) {
        attributes.put(name, value);
    }

    @Override
    public void setStreams(InputStream input, OutputStream output) {
        request = input;
        response = output;
    }

    @Override
    public HttpPrincipal getPrincipal() {
        return null;
    }
}
