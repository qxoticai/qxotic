package com.qxotic.jinfer.models.gemmaembedding2;

import static com.qxotic.jinfer.models.gemmaembedding2.GemmaEmbedding2.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/**
 * Each task method frames text with the card's prompt, character for character as {@code
 * config_sentence_transformers.json} spells it; GemmaEmbedding2ParityTest holds the same framing
 * against the official pipeline's tokens and vectors.
 */
class GemmaEmbedding2PromptsTest {

    @Test
    void everyTaskFramesTextWithTheCardsPrompt() {
        assertEquals("task: search result | query: x", searchQuery("x"));
        assertEquals("task: question answering | query: x", question("x"));
        assertEquals("task: fact checking | query: x", claim("x"));
        assertEquals("task: code retrieval | query: x", codeQuery("x"));
        assertEquals("task: classification | query: x", classification("x"));
        assertEquals("task: clustering | query: x", clustering("x"));
        assertEquals("task: sentence similarity | query: x", similarity("x"));
        assertEquals("title: Paris | text: x", document("Paris", "x"));
    }

    @Test
    void aDocumentWithoutATitleIsTitledNone() {
        assertEquals("title: none | text: x", document(null, "x"));
        assertEquals("title: none | text: x", document(" ", "x"));
    }

    @Test
    void theIntegrationsSearchFramingIsTheSameStrings() {
        assertEquals(searchQuery(""), SEARCH_QUERY);
        assertEquals(document(null, ""), DOCUMENT);
    }

    @Test
    void textIsRequired() {
        assertThrows(NullPointerException.class, () -> question(null));
        assertThrows(NullPointerException.class, () -> document("Paris", null));
    }
}
