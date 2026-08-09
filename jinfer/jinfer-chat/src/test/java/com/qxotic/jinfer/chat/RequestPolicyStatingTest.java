package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * A JSON schema must reach the model as WORDS, not only as a grammar mask - see {@link
 * RequestPolicy#stating}. The E2E proof is the langchain4j provider's AiServices extraction test;
 * these pin the placement rules the prompt depends on.
 */
final class RequestPolicyStatingTest {

    private static final Map<String, Object> SCHEMA =
            Map.of("type", "object", "properties", Map.of("name", Map.of("type", "string")));

    @Test
    void theSchemaLandsOnTheLastUserMessage() {
        List<Message> in =
                List.of(
                        Message.system("be brief"),
                        Message.user("first"),
                        Message.assistant("ok"),
                        Message.user("Johann is 42"));
        List<Message> out = RequestPolicy.stating(in, SCHEMA);

        assertEquals(in.size(), out.size(), "stating adds no message of its own");
        assertEquals(in.get(0), out.get(0));
        assertEquals(in.get(1), out.get(1), "an earlier user message is left alone");
        assertEquals(in.get(2), out.get(2));

        Message last = out.get(3);
        assertEquals(Role.USER, last.role());
        assertTrue(last.text().startsWith("Johann is 42"), last.text());
        assertTrue(last.text().contains("\"name\""), "the schema itself must appear: " + last);
    }

    @Test
    void noSchemaIsNoChange() {
        List<Message> in = List.of(Message.user("hi"));
        assertSame(in, RequestPolicy.stating(in, null));
        assertSame(in, RequestPolicy.stating(in, Map.of()));
    }

    @Test
    void aSchemaWithNoUserMessageToStateItToIsNotSmuggledElsewhere() {
        // silently mutating a system message (or inventing a user turn) would change a cached
        // prefix's bytes - the request simply carries the grammar alone
        List<Message> in = List.of(Message.system("be brief"));
        assertSame(in, RequestPolicy.stating(in, SCHEMA));
    }

    @Test
    void theMapShapeStatesTheSameStatementInTheSamePlace() {
        // the Jinja fallback path renders OpenAI maps, not typed messages; BOTH shapes of one
        // request must state identically or unported models regress to grammar-only
        List<Message> typed = List.of(Message.user("Johann is 42"));
        List<Object> maps =
                List.of(
                        Map.of("role", "system", "content", "be brief"),
                        Map.of("role", "user", "content", "Johann is 42"));
        String typedText = RequestPolicy.stating(typed, SCHEMA).get(0).text();

        List<Object> stated = RequestPolicy.statingMaps(maps, SCHEMA);
        assertEquals(maps.size(), stated.size(), "stating adds no message of its own");
        assertEquals(maps.get(0), stated.get(0), "the system map is left alone");
        assertEquals(
                typedText,
                ((Map<?, ?>) stated.get(1)).get("content"),
                "both shapes must produce the same stated bytes");

        assertSame(maps, RequestPolicy.statingMaps(maps, null));
        // the LAST user turn decides: non-string content there returns unchanged, never an
        // EARLIER user turn (a mid-history mutation the javadoc forbids)
        List<Object> structuredLast =
                List.of(
                        Map.of("role", "user", "content", "first"),
                        Map.of("role", "user", "content", List.of(Map.of("type", "image"))));
        assertSame(structuredLast, RequestPolicy.statingMaps(structuredLast, SCHEMA));
        List<Object> noUser = List.of(Map.of("role", "system", "content", "be brief"));
        assertSame(noUser, RequestPolicy.statingMaps(noUser, SCHEMA));
    }
}
