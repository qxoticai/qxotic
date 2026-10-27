package com.qxotic.toknroll.libtoknroll;

import com.qxotic.jinfer.hub.ModelRef;
import com.qxotic.jinfer.hub.ModelSource;
import com.qxotic.jinfer.hub.RemoteFile;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A hub in memory: folders of files with one payload, answering as a repository would. */
final class FakeSource implements ModelSource {
    private final Map<String, List<RemoteFile>> dirs = new HashMap<>();
    private byte[] payload = "{}".getBytes(StandardCharsets.UTF_8);
    boolean fetched;
    int listed;

    FakeSource serving(String dir, String... names) {
        dirs.put(
                dir,
                List.of(names).stream().map(n -> new RemoteFile(n, payload.length, null)).toList());
        return this;
    }

    FakeSource bytes(byte[] bytes) {
        payload = bytes;
        return this;
    }

    @Override
    public boolean supports(ModelRef ref) {
        return ref.host().equals("hf.co");
    }

    @Override
    public List<RemoteFile> list(ModelRef ref, String dir) throws IOException {
        listed++;
        List<RemoteFile> files = dirs.get(dir);
        if (files
                == null) // a repository answers a missing folder with a 404, which the store wraps
            // with its cause
            throw new IllegalArgumentException(
                    "no folder '" + dir + "'", new FileNotFoundException(dir));
        return files;
    }

    @Override
    public void fetch(ModelRef ref, RemoteFile file, Path into) throws IOException {
        fetched = true;
        Files.createDirectories(into.getParent());
        Files.write(into, payload);
    }
}
