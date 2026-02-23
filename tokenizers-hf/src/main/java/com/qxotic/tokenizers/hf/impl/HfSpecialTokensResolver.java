package com.qxotic.tokenizers.hf.impl;

import com.qxotic.format.json.JSON;
import com.qxotic.tokenizers.hf.HuggingFaceTokenizerException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Resolves special tokens from multiple HuggingFace tokenizer configuration sources.
 *
 * <p>Merges special tokens from:
 *
 * <ul>
 *   <li>tokenizer.json added_tokens
 *   <li>tokenizer_config.json added_tokens_decoder
 * </ul>
 *
 * <p>Hard-fails on conflicting token-to-ID mappings.
 */
public class HfSpecialTokensResolver {

    private final Path tokenizerJsonPath;
    private final Path tokenizerConfigPath;

    /**
     * Creates a resolver for the given configuration files.
     *
     * @param tokenizerJsonPath path to tokenizer.json (required)
     * @param tokenizerConfigPath path to tokenizer_config.json (optional, may be null)
     */
    public HfSpecialTokensResolver(Path tokenizerJsonPath, Path tokenizerConfigPath) {
        this.tokenizerJsonPath = tokenizerJsonPath;
        this.tokenizerConfigPath = tokenizerConfigPath;
    }

    /**
     * Resolves all special tokens from available sources.
     *
     * @param addedTokensFromTokenizer tokens already extracted from tokenizer.json
     * @return map of special token string to token ID
     * @throws HuggingFaceTokenizerException on conflict or invalid configuration
     */
    public Map<String, Integer> resolve(Map<String, Integer> addedTokensFromTokenizer) {
        Map<String, Integer> result = new HashMap<>();

        // Add tokens from tokenizer.json (already parsed)
        mergeTokens(result, addedTokensFromTokenizer, "tokenizer.json added_tokens");

        // Add tokens from tokenizer_config.json if available
        if (tokenizerConfigPath != null && Files.exists(tokenizerConfigPath)) {
            Map<String, Integer> configTokens = extractFromTokenizerConfig();
            mergeTokens(result, configTokens, "tokenizer_config.json");
        }

        return Map.copyOf(result);
    }

    /** Merges tokens into the result map, checking for conflicts. */
    private void mergeTokens(
            Map<String, Integer> result, Map<String, Integer> source, String sourceName) {
        for (Map.Entry<String, Integer> entry : source.entrySet()) {
            String token = entry.getKey();
            Integer newId = entry.getValue();

            Integer existingId = result.get(token);
            if (existingId != null) {
                if (!existingId.equals(newId)) {
                    throw new HuggingFaceTokenizerException(
                            tokenizerJsonPath.toString(),
                            "Conflicting token-to-ID mapping for special token '"
                                    + token
                                    + "': "
                                    + "existing ID "
                                    + existingId
                                    + " from another source, but "
                                    + sourceName
                                    + " specifies ID "
                                    + newId);
                }
                // Same ID, no conflict
                continue;
            }

            // Check for ID conflict (different tokens, same ID)
            for (Map.Entry<String, Integer> existing : result.entrySet()) {
                if (existing.getValue().equals(newId)) {
                    throw new HuggingFaceTokenizerException(
                            tokenizerJsonPath.toString(),
                            "Duplicate token ID "
                                    + newId
                                    + ": tokens '"
                                    + existing.getKey()
                                    + "' and '"
                                    + token
                                    + "' both map to ID "
                                    + newId
                                    + "\n"
                                    + "Token '"
                                    + token
                                    + "' from "
                                    + sourceName);
                }
            }

            result.put(token, newId);
        }
    }

    /** Extracts special tokens from tokenizer_config.json. */
    @SuppressWarnings("unchecked")
    private Map<String, Integer> extractFromTokenizerConfig() {
        Map<String, Integer> tokens = new HashMap<>();

        try {
            String configContent = Files.readString(tokenizerConfigPath);
            Object parsed = JSON.parse(configContent);

            if (!(parsed instanceof Map)) {
                throw new HuggingFaceTokenizerException(
                        tokenizerConfigPath.toString(),
                        "Invalid tokenizer_config.json: expected JSON object");
            }

            Map<String, Object> root = (Map<String, Object>) parsed;

            // Extract from added_tokens_decoder
            Object decoderObj = root.get("added_tokens_decoder");
            if (decoderObj instanceof Map) {
                Map<String, Object> decoder = (Map<String, Object>) decoderObj;
                for (Map.Entry<String, Object> entry : decoder.entrySet()) {
                    try {
                        int id = Integer.parseInt(entry.getKey());
                        Object tokenInfo = entry.getValue();
                        if (tokenInfo instanceof Map) {
                            Map<String, Object> info = (Map<String, Object>) tokenInfo;
                            Object content = info.get("content");
                            if (content instanceof String) {
                                tokens.put((String) content, id);
                            }
                        }
                    } catch (NumberFormatException e) {
                        // Skip non-numeric keys
                    }
                }
            }

            return tokens;
        } catch (HuggingFaceTokenizerException e) {
            throw e;
        } catch (Exception e) {
            throw new HuggingFaceTokenizerException(
                    tokenizerConfigPath.toString(), "Failed to parse tokenizer_config.json", e);
        }
    }
}
