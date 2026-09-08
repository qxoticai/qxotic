package com.qxotic.jinfer.tts;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipFile;

/** Reads uncompressed entries from the ZIP overlay appended to an executable. */
final class SelfArchive implements AutoCloseable {
    private static final int MAX_SYMLINK_DEPTH = 8;

    private final ZipFile zip;
    private final FileChannel channel;

    private SelfArchive(ZipFile zip, FileChannel channel) {
        this.zip = zip;
        this.channel = channel;
    }

    static SelfArchive open() throws IOException {
        Path self =
                ProcessHandle.current()
                        .info()
                        .command()
                        .map(Path::of)
                        .orElseThrow(() -> new IOException("cannot find the current executable"));
        return open(self);
    }

    static SelfArchive open(Path file) throws IOException {
        FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
        try {
            return new SelfArchive(new ZipFile.Builder().setPath(file).get(), channel);
        } catch (IOException e) {
            channel.close();
            throw e;
        }
    }

    FileChannel fileChannel() {
        return channel;
    }

    record Entry(String name, long offset, long size) {}

    Entry entry(String name) throws IOException {
        return resolve(name, name, new HashSet<>());
    }

    private Entry resolve(String name, String as, Set<String> seen) throws IOException {
        if (!seen.add(name)) throw new IOException("symlink cycle at " + name);
        if (seen.size() > MAX_SYMLINK_DEPTH) throw new IOException("symlinks too deep at " + name);
        ZipArchiveEntry entry = zip.getEntry(name);
        if (entry == null) throw new IOException("entry not found: " + name);
        if (entry.isUnixSymlink()) {
            String target;
            try (InputStream in = zip.getInputStream(entry)) {
                target =
                        new String(in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8)
                                .trim();
            }
            return resolve(target, as, seen);
        }
        if (entry.getMethod() != ZipArchiveEntry.STORED)
            throw new IOException("entry must be STORED to be mapped: " + name);
        return new Entry(as, entry.getDataOffset(), entry.getSize());
    }

    List<Entry> entries() {
        List<Entry> usable = new ArrayList<>();
        for (Enumeration<ZipArchiveEntry> entries = zip.getEntries(); entries.hasMoreElements(); ) {
            String name = entries.nextElement().getName();
            try {
                usable.add(entry(name));
            } catch (IOException ignored) {
                // Listings omit entries that cannot be mapped.
            }
        }
        usable.sort(Comparator.comparing(Entry::name));
        return usable;
    }

    /** A bounded, positional view of an entry that does not move or close the archive channel. */
    ReadableByteChannel channel(Entry entry) {
        return new ReadableByteChannel() {
            private long position;
            private boolean open = true;

            @Override
            public int read(ByteBuffer destination) throws IOException {
                if (!open) throw new IOException("entry channel is closed");
                if (position == entry.size()) return -1;
                int oldLimit = destination.limit();
                int count = (int) Math.min(destination.remaining(), entry.size() - position);
                destination.limit(destination.position() + count);
                try {
                    int read =
                            SelfArchive.this.channel.read(destination, entry.offset() + position);
                    if (read < 0)
                        throw new IOException("archive ended inside entry " + entry.name());
                    position += read;
                    return read;
                } finally {
                    destination.limit(oldLimit);
                }
            }

            @Override
            public boolean isOpen() {
                return open;
            }

            @Override
            public void close() {
                open = false;
            }
        };
    }

    Path extract(Entry entry) throws IOException {
        String fileName = Path.of(entry.name()).getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        String suffix = dot >= 0 ? fileName.substring(dot) : ".bin";
        Path target = Files.createTempFile("jinfer-tts-", suffix);
        try (ReadableByteChannel source = channel(entry)) {
            Files.copy(
                    Channels.newInputStream(source), target, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            Files.deleteIfExists(target);
            throw e;
        }
        if (Files.size(target) != entry.size()) {
            Files.deleteIfExists(target);
            throw new IOException("short archive entry: " + entry.name());
        }
        return target;
    }

    @Override
    public void close() throws IOException {
        try (zip;
                channel) {
            // Both resources close even if the first close fails.
        }
    }
}
