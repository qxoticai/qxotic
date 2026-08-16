package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * One place a {@link ModelStore} can fetch refs from. A source answers two questions - "what files
 * does this repository have" and "give me that file" - and nothing else: selection, caching, layout
 * and format policy all live in the store.
 *
 * <p>The contract:
 *
 * <ul>
 *   <li>{@link #supports} decides from the ref alone (typically its host), never from the network.
 *   <li>{@link #list} returns FILES only, paths repository-relative; {@code dir} is the
 *       repository-relative folder to list, {@code ""} for the root. Listing a path that is not a
 *       folder fails - the store turns that failure into a better message.
 *   <li>{@link #fetch} writes {@code file} to {@code into}, staged and renamed, so a partial
 *       download never appears as a complete one. When {@code file.sha256()} is non-null the bytes
 *       MUST match it; a mismatch fails the fetch and leaves nothing behind.
 *   <li>A source NEVER writes into the store beyond {@code into}, and never reads the store at all.
 *       A read-only store therefore still serves every cache hit; a source works against any root.
 *   <li>Failures are {@link IOException} (the source is at fault: network, refused bytes) or {@link
 *       IllegalArgumentException} caused by a {@code Fetch.HttpStatusException} (the repository
 *       answered, and the answer was "no"). The store falls through to the next source for both,
 *       and lets every other runtime exception propagate - the ref itself is at fault there.
 * </ul>
 *
 * <p>Implementations should carry a readable {@link #toString()}: it appears in the store's logs
 * when a fallback happens.
 */
public interface ModelSource {

    /** Whether this source can serve {@code ref} - from the ref alone, never the network. */
    boolean supports(ModelRef ref);

    /** The files in {@code dir} ("" = the repository root) at the ref's revision. */
    List<RemoteFile> list(ModelRef ref, String dir) throws IOException;

    /** Writes {@code file} to {@code into}, staged and renamed; verifies its sha256 when set. */
    void fetch(ModelRef ref, RemoteFile file, Path into) throws IOException;
}
