// Reads a ZIP overlay appended to a file - normally the running executable itself, so a native
// image can carry its GGUF models inside the binary:
//
//   try (SelfArchive archive = SelfArchive.open()) {
//       SelfArchive.Entry entry = archive.entry("models/nano.gguf");
//       byte[] header = archive.readAt(entry.offset(), 65536);
//       // tensor data maps straight from archive.channel() at entry.offset() + ...
//   }
//
// Entries must be STORED (uncompressed) so their bytes can be mapped in place. Commons Compress is
// the dependency here for one reason: ZipArchiveEntry.getDataOffset() gives the exact byte where an
// entry's data begins, which the JDK's java.util.zip does not expose and which cannot be derived
// without parsing the central directory by hand.
package com.qxotic.jinfer.x.examples.inflect2;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipFile;

final class SelfArchive implements AutoCloseable {

    /** How many symlinks may be followed before assuming the archive is malformed. */
    private static final int MAX_SYMLINK_DEPTH = 8;

    private final ZipFile zip;
    private final FileChannel channel;

    private SelfArchive(ZipFile zip, FileChannel channel) {
        this.zip = zip;
        this.channel = channel;
    }

    /** Open the running executable as an archive. */
    static SelfArchive open() throws IOException {
        Path self =
                ProcessHandle.current()
                        .info()
                        .command()
                        .map(Path::of)
                        .orElseThrow(() -> new IOException("cannot find the current executable"));
        return open(self);
    }

    /** Open an arbitrary file that has a ZIP overlay appended. */
    static SelfArchive open(Path file) throws IOException {
        FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
        try {
            return new SelfArchive(new ZipFile.Builder().setPath(file).get(), channel);
        } catch (IOException e) {
            channel.close();
            throw e;
        }
    }

    /** The channel over the archive file, for mapping entry data in place. */
    FileChannel channel() {
        return channel;
    }

    /** One STORED entry: its name, the absolute offset of its data, and its length. */
    record Entry(String name, long offset, int size) {}

    /** Look up an entry by name, following Unix symlinks. */
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
                target = new String(in.readAllBytes(), StandardCharsets.UTF_8).trim();
            }
            return resolve(target, as, seen);
        }
        if (entry.getMethod() != ZipArchiveEntry.STORED)
            throw new IOException("entry must be STORED to be mapped: " + name);
        // Keep the name the caller asked for, so a listing shows the link, not its target.
        return new Entry(as, entry.getDataOffset(), (int) entry.getSize());
    }

    /**
     * Every usable entry, by name. Symlinks appear under their own name, resolved to the target's
     * data; entries that are compressed, broken or dangling are left out - a listing should not
     * fail because one entry is unusable.
     */
    List<Entry> entries() {
        List<Entry> usable = new ArrayList<>();
        for (Enumeration<ZipArchiveEntry> e = zip.getEntries(); e.hasMoreElements(); ) {
            String name = e.nextElement().getName();
            try {
                usable.add(entry(name));
            } catch (IOException unusable) {
                // not mappable: skipped rather than reported, see above
            }
        }
        usable.sort(Comparator.comparing(Entry::name));
        return usable;
    }

    /** Read exactly {@code count} bytes at {@code offset}. */
    byte[] readAt(long offset, int count) throws IOException {
        byte[] bytes = new byte[count];
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        long at = offset;
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, at);
            if (read < 0)
                throw new EOFException(
                        "archive ends after "
                                + buffer.position()
                                + " of "
                                + count
                                + " bytes at offset "
                                + offset);
            at += read;
        }
        return bytes;
    }

    @Override
    public void close() throws IOException {
        try (zip;
                channel) {
            // both closed, first failure wins, second still attempted
        }
    }
}
