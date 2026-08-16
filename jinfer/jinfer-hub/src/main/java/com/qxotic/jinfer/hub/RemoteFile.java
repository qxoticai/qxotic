package com.qxotic.jinfer.hub;

/**
 * A file as a {@link ModelSource} lists it. {@code path} is REPOSITORY-RELATIVE, so it carries any
 * subfolder; {@code sha256} is null when the host does not publish one - the only integrity check
 * for such a file is the size the server states, and {@link ModelSource#fetch} verifies what it is
 * given and nothing more.
 */
public record RemoteFile(String path, long sizeBytes, String sha256) {}
