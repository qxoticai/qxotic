// Self-archive reader: the executable itself has a ZIP overlay appended.
// Uses Apache Commons Compress for ZipArchiveEntry.getDataOffset() — the exact
// byte offset where a STORED entry's data begins, so we can mmap from there.
//
//   SelfArchive sa = SelfArchive.open();
//   SelfArchive.Entry e = sa.entry("models/nano.gguf");
//   byte[] header = sa.readAt(e.offset(), 65536);     // GGUF header
//   // mmap tensor data: sa.channel().map(..., e.offset + tensorDataOff, ...)
package com.qxotic.jinfer.models.inflect2;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipFile;

public final class SelfArchive implements AutoCloseable {

    private final ZipFile zip;
    private final FileChannel channel;

    private SelfArchive(ZipFile zip, FileChannel channel) {
        this.zip = zip;
        this.channel = channel;
    }

    /** Open the running executable itself as a self-archive. */
    public static SelfArchive open() throws IOException {
        Path self =
                ProcessHandle.current()
                        .info()
                        .command()
                        .map(Path::of)
                        .orElseThrow(() -> new IOException("cannot find current process binary"));
        return new SelfArchive(new ZipFile(self), FileChannel.open(self, StandardOpenOption.READ));
    }

    public FileChannel channel() {
        return channel;
    }

    public record Entry(String name, long offset, int size) {}

    /** Look up a STORED entry by name. Symlinks are resolved transparently (max depth 8). */
    public Entry entry(String name) throws IOException {
        return resolveEntry(name, new HashSet<>());
    }

    private Entry resolveEntry(String name, Set<String> visited) throws IOException {
        if (!visited.add(name)) throw new IOException("symlink cycle: " + name);
        if (visited.size() > 8) throw new IOException("symlink depth exceeded: " + name);

        ZipArchiveEntry e = zip.getEntry(name);
        if (e == null) throw new IOException("entry not found: " + name);

        // Follow Unix symlinks — the entry data is the target path
        if (e.isUnixSymlink()) {
            String target;
            try (InputStream in = zip.getInputStream(e)) {
                target = new String(in.readAllBytes(), StandardCharsets.UTF_8).trim();
            }
            return resolveEntry(target, visited);
        }

        if (e.getMethod() != ZipArchiveEntry.STORED)
            throw new IOException("entry must be STORED: " + name);
        return new Entry(name, e.getDataOffset(), (int) e.getSize());
    }

    /** All STORED entries, sorted by name. Symlinks show their target. */
    public List<Entry> entries() {
        List<Entry> list = new ArrayList<>();
        for (Enumeration<ZipArchiveEntry> e = zip.getEntries(); e.hasMoreElements(); ) {
            ZipArchiveEntry z = e.nextElement();
            if (z.isUnixSymlink()) {
                String target;
                try (InputStream in = zip.getInputStream(z)) {
                    target = new String(in.readAllBytes(), StandardCharsets.UTF_8).trim();
                } catch (IOException ignored) {
                    target = "?";
                }
                // Symlink entry — use the name + resolved size
                try {
                    Entry resolved = resolveEntry(z.getName(), new HashSet<>());
                    list.add(
                            new Entry(
                                    z.getName() + " → " + target,
                                    resolved.offset(),
                                    resolved.size()));
                } catch (IOException ignored) {
                    list.add(new Entry(z.getName() + " → " + target + " (broken)", 0, 0));
                }
            } else if (z.getMethod() == ZipArchiveEntry.STORED) {
                list.add(new Entry(z.getName(), z.getDataOffset(), (int) z.getSize()));
            }
        }
        list.sort((a, b) -> a.name.compareTo(b.name));
        return list;
    }

    public byte[] readAt(long offset, int count) throws IOException {
        byte[] buf = new byte[count];
        ByteBuffer bb = ByteBuffer.wrap(buf);
        channel.position(offset);
        while (bb.hasRemaining() && channel.read(bb) >= 0) {}
        return buf;
    }

    @Override
    public void close() throws IOException {
        zip.close();
        channel.close();
    }
}
