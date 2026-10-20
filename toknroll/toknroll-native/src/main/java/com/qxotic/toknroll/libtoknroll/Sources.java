package com.qxotic.toknroll.libtoknroll;

import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * toknroll_load's source grammar, the one the toknroll CLI speaks: a local tokenizer.json, its
 * directory or a GGUF file; otherwise a hub reference {@code owner/repo}, with {@code :QUANT} (or
 * {@code /QUANT}) naming a GGUF whose tokenizer is read from its header alone, behind an optional
 * {@code hf:} or {@code ms:} hub prefix.
 */
final class Sources {
    private static final GGUFTokenizerLoader GGUF =
            GGUFTokenizerLoader.createBuilderWithBuiltins().build();

    static Tokenizer load(String source) {
        if (source.isBlank()) throw Errors.invalid("source is empty");
        boolean modelScope = false;
        String ref = source;
        for (String prefix : new String[] {"ms:", "modelscope:"}) {
            if (ref.startsWith(prefix)) {
                modelScope = true;
                ref = ref.substring(prefix.length());
            }
        }
        for (String prefix : new String[] {"hf:", "huggingface:"}) {
            if (ref.startsWith(prefix)) ref = ref.substring(prefix.length());
        }
        if (ref == source && looksLocal(source)) return local(Path.of(source));
        return hub(modelScope, ref);
    }

    private static boolean looksLocal(String source) {
        String lower = source.toLowerCase(Locale.ROOT);
        return source.startsWith("/")
                || source.startsWith("./")
                || source.startsWith("../")
                || source.startsWith("~")
                || lower.endsWith(".gguf")
                || lower.endsWith(".json")
                || Files.exists(Path.of(source));
    }

    private static Tokenizer local(Path path) {
        if (!Files.exists(path)) throw Errors.notFound(path + " does not exist");
        if (path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf"))
            return GGUF.fromLocal(path);
        return HuggingFaceTokenizerLoader.fromLocal(path);
    }

    private static Tokenizer hub(boolean modelScope, String ref) {
        String quant = null;
        int colon = ref.indexOf(':');
        if (colon >= 0) {
            quant = ref.substring(colon + 1);
            ref = ref.substring(0, colon);
        }
        String[] parts = ref.split("/");
        if (parts.length == 3 && quant == null) {
            quant = parts[2];
            parts = new String[] {parts[0], parts[1]};
        }
        if (parts.length != 2
                || parts[0].isEmpty()
                || parts[1].isEmpty()
                || (quant != null && quant.isEmpty()))
            throw Errors.invalid(
                    "expected a tokenizer.json, a directory, a GGUF file or owner/repo[:QUANT], got"
                            + " '"
                            + ref
                            + (quant != null ? ":" + quant : "")
                            + "'");
        String owner = parts[0], repo = parts[1];
        if (quant == null) {
            return modelScope
                    ? HuggingFaceTokenizerLoader.fromModelScope(owner, repo)
                    : HuggingFaceTokenizerLoader.fromHuggingFace(owner, repo);
        }
        // ponytail: the CLI's file-name convention, <repo without -GGUF>-<QUANT>.gguf; a repository
        // that names its files differently needs the hub's file listing, which lives in jinfer-hub
        String base = repo.endsWith("-GGUF") ? repo.substring(0, repo.length() - 5) : repo;
        String file = base + "-" + quant + ".gguf";
        return modelScope
                ? GGUF.fromModelScope(owner, repo, file)
                : GGUF.fromHuggingFace(owner, repo, file);
    }

    private Sources() {}
}
