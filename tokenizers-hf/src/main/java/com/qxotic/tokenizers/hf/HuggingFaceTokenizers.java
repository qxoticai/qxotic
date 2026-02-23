package com.qxotic.tokenizers.hf;

import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.hf.impl.HfJTokkitFactory;
import com.qxotic.tokenizers.hf.impl.HfSpecialTokensResolver;
import com.qxotic.tokenizers.hf.impl.HfTokenizerJsonParser;
import com.qxotic.tokenizers.hf.impl.HuggingFaceFileCache;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Public API for loading tokenizers from HuggingFace format.
 *
 * <p>Supports:
 *
 * <ul>
 *   <li>Local tokenizer.json files
 *   <li>Local directories with HF tokenizer files
 *   <li>Remote HF repositories with automatic caching
 * </ul>
 *
 * <p>Features:
 *
 * <ul>
 *   <li>JTokkit backend for high performance
 *   <li>BPE tokenizer support only
 *   <li>Explicit regex requirement
 *   <li>Hard-fail on unsupported configurations
 *   <li>File caching for remote repositories
 * </ul>
 *
 * <p>Environment variables:
 *
 * <ul>
 *   <li>QXOTIC_TOKENIZERS_HF_CACHE_DIR - override cache directory
 *   <li>QXOTIC_TOKENIZERS_HF_OFFLINE - offline mode (use cache only)
 *   <li>QXOTIC_HF_TOKEN - authentication token for private repos
 * </ul>
 */
public final class HuggingFaceTokenizers {

    private static final HuggingFaceFileCache cache = new HuggingFaceFileCache();

    private HuggingFaceTokenizers() {
        // Utility class
    }

    /**
     * Loads a tokenizer from a HuggingFace repository.
     *
     * <p>Downloads and caches the required files from HuggingFace Hub.
     *
     * @param repoId the repository ID (e.g., "Qwen/Qwen3-0.6B")
     * @return the tokenizer
     * @throws HuggingFaceTokenizerException if loading fails
     */
    public static Tokenizer fromRepository(String repoId) {
        return fromRepository(repoId, "main");
    }

    /**
     * Loads a tokenizer from a HuggingFace repository with specific revision.
     *
     * @param repoId the repository ID (e.g., "Qwen/Qwen3-0.6B")
     * @param revision the revision (e.g., "main", "v1.0", commit hash)
     * @return the tokenizer
     * @throws HuggingFaceTokenizerException if loading fails
     */
    public static Tokenizer fromRepository(String repoId, String revision) {
        // Download required file
        Path tokenizerJson = cache.getOrDownload(repoId, revision, "tokenizer.json");

        // Download optional files
        Path tokenizerConfig = null;
        Path mergesTxt = null;

        try {
            tokenizerConfig = cache.getOrDownload(repoId, revision, "tokenizer_config.json");
        } catch (HuggingFaceTokenizerException e) {
            if (!isOptionalFileMissing(e)) {
                throw e;
            }
        }

        try {
            mergesTxt = cache.getOrDownload(repoId, revision, "merges.txt");
        } catch (HuggingFaceTokenizerException e) {
            if (!isOptionalFileMissing(e)) {
                throw e;
            }
        }

        return fromFiles(tokenizerJson, mergesTxt, tokenizerConfig);
    }

    /**
     * Loads a tokenizer from a tokenizer.json file.
     *
     * @param tokenizerJson path to tokenizer.json
     * @return the tokenizer
     * @throws HuggingFaceTokenizerException if loading fails
     */
    public static Tokenizer fromTokenizerJson(Path tokenizerJson) {
        return fromFiles(tokenizerJson, null, null);
    }

    /**
     * Loads a tokenizer from a directory containing HF tokenizer files.
     *
     * <p>Looks for tokenizer.json (required), and optionally tokenizer_config.json and merges.txt.
     *
     * @param dir path to directory
     * @return the tokenizer
     * @throws HuggingFaceTokenizerException if loading fails
     */
    public static Tokenizer fromDirectory(Path dir) {
        if (!Files.isDirectory(dir)) {
            throw new HuggingFaceTokenizerException("Not a directory: " + dir);
        }

        Path tokenizerJson = dir.resolve("tokenizer.json");
        if (!Files.exists(tokenizerJson)) {
            throw new HuggingFaceTokenizerException(
                    "tokenizer.json not found in directory: " + dir);
        }

        Path tokenizerConfig = dir.resolve("tokenizer_config.json");
        Path mergesTxt = dir.resolve("merges.txt");

        return fromFiles(
                tokenizerJson,
                Files.exists(mergesTxt) ? mergesTxt : null,
                Files.exists(tokenizerConfig) ? tokenizerConfig : null);
    }

    /**
     * Loads a tokenizer from explicit file paths.
     *
     * @param tokenizerJson path to tokenizer.json (required)
     * @param mergesTxt path to merges.txt (optional)
     * @param tokenizerConfig path to tokenizer_config.json (optional)
     * @return the tokenizer
     * @throws HuggingFaceTokenizerException if loading fails
     */
    public static Tokenizer fromFiles(Path tokenizerJson, Path mergesTxt, Path tokenizerConfig) {

        // Validate required file
        if (tokenizerJson == null) {
            throw new HuggingFaceTokenizerException("tokenizer.json path is required");
        }

        if (!Files.exists(tokenizerJson)) {
            throw new HuggingFaceTokenizerException(
                    tokenizerJson.toString(), "tokenizer.json not found");
        }

        // Parse tokenizer.json (with optional merges.txt)
        HfTokenizerJsonParser parser = new HfTokenizerJsonParser(tokenizerJson, mergesTxt);
        HfTokenizerJsonParser.TokenizerData data = parser.parse();

        // Resolve special tokens
        HfSpecialTokensResolver resolver =
                new HfSpecialTokensResolver(tokenizerJson, tokenizerConfig);
        Map<String, Integer> specialTokens = resolver.resolve(data.addedTokens());

        // Build the tokenizer
        HfJTokkitFactory factory = new HfJTokkitFactory(data, specialTokens);
        return factory.build();
    }

    private static boolean isOptionalFileMissing(HuggingFaceTokenizerException e) {
        String message = e.getMessage();
        if (message == null) {
            return false;
        }
        String lower = message.toLowerCase();
        return lower.contains("404") || lower.contains("not found");
    }
}
