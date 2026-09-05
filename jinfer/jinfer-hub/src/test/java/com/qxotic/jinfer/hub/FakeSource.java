package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A {@link ModelSource} that answers from memory, for store-level tests that must not touch the
 * network. Listings are per directory ({@code ""} is the repository root); a directory nobody
 * planted fails the way a repository answers "no" - an {@link IllegalArgumentException} caused by
 * an HTTP 404 - so the store's folder probing exercises its real fallback path.
 */
final class FakeSource implements ModelSource {

    private final String name;
    private final Map<String, List<RemoteFile>> dirs = new ConcurrentHashMap<>();
    private final List<String> requestedDirs = Collections.synchronizedList(new ArrayList<>());
    private final AtomicBoolean fetched = new AtomicBoolean();
    private volatile Exception failure; // thrown by every list, when set
    private volatile String bytes = "weights";

    FakeSource(String name) {
        this.name = name;
    }

    FakeSource serving(String dir, RemoteFile... files) {
        dirs.put(dir, List.of(files));
        return this;
    }

    FakeSource failing(Exception failure) {
        this.failure = failure;
        return this;
    }

    private IOException fetchFailure;

    /** The listing succeeds, the download does not: a gated file behind a public listing. */
    FakeSource fetchFailing(IOException failure) {
        this.fetchFailure = failure;
        return this;
    }

    FakeSource bytes(String bytes) {
        this.bytes = bytes;
        return this;
    }

    /** The directories {@link #list} was asked for, in order. */
    List<String> requestedDirs() {
        return requestedDirs;
    }

    boolean fetched() {
        return fetched.get();
    }

    @Override
    public boolean supports(ModelRef ref) {
        return ref.host().equals("hf.co");
    }

    @Override
    public List<RemoteFile> list(ModelRef ref, String dir) throws IOException {
        requestedDirs.add(dir);
        if (failure instanceof IOException io) {
            throw io;
        }
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        List<RemoteFile> files = dirs.get(dir);
        if (files == null) {
            throw new IllegalArgumentException(
                    "no folder '" + dir + "' at " + name,
                    new Fetch.HttpStatusException(404, "fake://" + name + "/" + dir, ""));
        }
        return files;
    }

    @Override
    public void fetch(ModelRef ref, RemoteFile file, Path into) throws IOException {
        if (fetchFailure != null) throw fetchFailure;
        fetched.set(true);
        Files.createDirectories(into.getParent());
        Files.writeString(into, bytes);
    }

    @Override
    public String toString() {
        return name;
    }
}
