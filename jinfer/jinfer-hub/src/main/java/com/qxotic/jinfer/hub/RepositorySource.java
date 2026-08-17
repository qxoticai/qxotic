package com.qxotic.jinfer.hub;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The protocol of one repository host: listing, fetch URLs, credentials. Both shipped hosts serve a
 * repository the same way - {@code <base><prefix>/owner/repo/resolve/<revision>/<file>},
 * redirecting to a signed CDN URL - so only the listing API and the {@link ModelRef.Host} row
 * differ. Package-visible: the public door is a named subclass ({@link HuggingFaceSource}, {@link
 * ModelScopeSource}), because the name is what the store's fallback log should say.
 */
class RepositorySource implements ModelSource {

    private final ModelRef.Host host;
    private final String base;

    RepositorySource(ModelRef.Host host, URI endpoint) {
        this.host = host;
        if (endpoint == null) {
            this.base = host.base(); // the env mirror variable, read at construction
        } else {
            String trimmed = endpoint.toASCIIString().strip();
            this.base =
                    trimmed.endsWith("/") ? trimmed.substring(0, trimmed.length() - 1) : trimmed;
        }
    }

    @Override
    public boolean supports(ModelRef ref) {
        return ref.host().equals(host.name);
    }

    @Override
    public List<RemoteFile> list(ModelRef ref, String dir) throws IOException {
        return switch (host) {
            case HF -> listHuggingFace(ref, dir);
            case MODELSCOPE -> listModelScope(ref, dir);
        };
    }

    @Override
    public void fetch(ModelRef ref, RemoteFile file, Path into) throws IOException {
        Fetch.download(
                fileUrl(ref, file.path()), into, file.sizeBytes(), file.sha256(), headers(host));
    }

    /** {@code <base><prefix>/owner/repo/resolve/<revision>/<file>} - a 302 to a signed CDN URL. */
    String fileUrl(ModelRef ref, String file) {
        return base
                + host.prefix
                + "/"
                + ref.repoId()
                + "/resolve/"
                + ref.revisionOrDefault()
                + "/"
                + file;
    }

    /**
     * The commit this ref's revision names, or null when it cannot be learned. The hub layout keys
     * snapshots by commit, so joining it starts here; a ref already pinned to a commit needs no
     * request. HuggingFace only - ModelScope has no shared cache layout to join.
     */
    String commitFor(ModelRef ref) {
        if (host != ModelRef.Host.HF) {
            return null;
        }
        try {
            Map<String, Object> body =
                    Json.parseMap(get(ref, base + "/api/models/" + ref.repoId() + "/refs"));
            for (String kind : new String[] {"branches", "tags"}) {
                for (Object entry : Json.queryList(body, kind).orElse(List.of())) {
                    if (entry instanceof Map<?, ?> map
                            && ref.revisionOrDefault().equals(map.get("name"))
                            && map.get("targetCommit") instanceof String commit
                            && Hub.isCommit(commit)) {
                        return commit;
                    }
                }
            }
            return null;
        } catch (IOException unreachable) {
            return null;
        }
    }

    /**
     * {@code /api/models/owner/repo/tree/<rev>/<path>}: a JSON array, sha256 under {@code lfs.oid}.
     */
    private List<RemoteFile> listHuggingFace(ModelRef ref, String path) throws IOException {
        String url =
                base
                        + "/api/models/"
                        + ref.repoId()
                        + "/tree/"
                        + ref.revisionOrDefault()
                        + (path.isEmpty() ? "" : "/" + path);
        List<RemoteFile> files = new ArrayList<>();
        for (Object entry : Json.parseList(get(ref, url))) {
            if (!(entry instanceof Map<?, ?> map)) {
                continue;
            }
            if (!"file".equals(map.get("type"))) {
                continue; // directories are not files; the store asks again with the folder as dir
            }
            // a plain file's "oid" is a git blob sha1, not content: only LFS entries carry a
            // sha256, and every GGUF worth downloading is one
            String sha256 =
                    map.get("lfs") instanceof Map<?, ?> lfs && lfs.get("oid") instanceof String oid
                            ? oid
                            : null;
            files.add(new RemoteFile(str(map.get("path")), num(map.get("size")), sha256));
        }
        return files;
    }

    /** {@code /api/v1/models/owner/repo/repo/files}: {@code Data.Files}, sha256 on every entry. */
    private List<RemoteFile> listModelScope(ModelRef ref, String path) throws IOException {
        String url =
                base
                        + "/api/v1/models/"
                        + ref.repoId()
                        + "/repo/files?Revision="
                        + ref.revisionOrDefault()
                        + "&Root="
                        + path;
        Map<String, Object> body = Json.parseMap(get(ref, url));
        List<Object> entries =
                Json.queryList(body, "Data", "Files")
                        .orElseThrow(
                                () ->
                                        new IOException(
                                                "unexpected listing for "
                                                        + ref.repoId()
                                                        + " from "
                                                        + url
                                                        + " - API changed?"));
        List<RemoteFile> files = new ArrayList<>();
        for (Object entry : entries) {
            if (!(entry instanceof Map<?, ?> map)) {
                continue;
            }
            if ("blob".equals(map.get("Type"))) {
                files.add(
                        new RemoteFile(
                                str(map.get("Path")),
                                num(map.get("Size")),
                                map.get("Sha256") instanceof String s && !s.isBlank() ? s : null));
            }
        }
        return files;
    }

    /**
     * A listing GET, translating the failures a user can actually fix: a gated repository, and one
     * that is not there. Both surface as {@link IllegalArgumentException} caused by the status, so
     * the store can tell "this source answered no" from "this source is down".
     */
    private String get(ModelRef ref, String url) throws IOException {
        try {
            return Fetch.getString(url, headers(host));
        } catch (Fetch.HttpStatusException e) {
            if (e.status == 401 || e.status == 403) {
                throw new IllegalArgumentException(
                        ref.repoId()
                                + " is gated or private. Accept its licence at "
                                + ref.repoUrl()
                                + " then set "
                                + host.tokenEnv
                                + (host == ModelRef.Host.HF
                                        ? " (or log in once with: hf auth login)"
                                        : ""),
                        e);
            }
            if (e.status == 404) {
                throw new IllegalArgumentException(
                        "no repository "
                                + ref.repoId()
                                + " at revision "
                                + ref.revisionOrDefault()
                                + " on "
                                + host.name
                                + " ("
                                + ref.repoUrl()
                                + ")",
                        e);
            }
            throw e;
        }
    }

    /**
     * Credentials for {@code host}: its token variable, and for HuggingFace the token {@code hf
     * auth login} leaves behind - someone who has logged in once should not have to export
     * anything.
     */
    static Map<String, String> headers(ModelRef.Host host) {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("User-Agent", "jinfer-hub");
        String token = System.getenv(host.tokenEnv);
        if ((token == null || token.isBlank()) && host == ModelRef.Host.HF) {
            token = huggingFaceTokenFile();
        }
        if (token != null && !token.isBlank()) {
            headers.put("Authorization", "Bearer " + token.strip());
        }
        return headers;
    }

    /** {@code $HF_TOKEN_PATH} &gt; {@code $HF_HOME/token} &gt; the default, as every HF client. */
    private static String huggingFaceTokenFile() {
        String explicit = System.getenv("HF_TOKEN_PATH");
        String home = System.getenv("HF_HOME");
        Path token;
        if (explicit != null && !explicit.isBlank()) {
            token = Path.of(explicit);
        } else if (home != null && !home.isBlank()) {
            token = Path.of(home, "token");
        } else {
            token = Path.of(System.getProperty("user.home"), ".cache", "huggingface", "token");
        }
        try {
            return Files.isRegularFile(token)
                    ? Files.readString(token, StandardCharsets.UTF_8)
                    : null;
        } catch (IOException e) {
            return null; // an unreadable token file is not a reason to fail a public download
        }
    }

    private static String str(Object value) {
        return value instanceof String s ? s : "";
    }

    private static long num(Object value) {
        return value instanceof Number n ? n.longValue() : -1;
    }

    @Override
    public String toString() {
        return host.name + " via " + base;
    }
}
