package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.chat.Content;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class MiniCpmToolSyntaxTest {

    @Test
    void parsesRawAndCdataArguments() {
        String payload =
                " name=\"search\"><param name=\"query\"><![CDATA[a < b & c\nnext]]></param>"
                        + "<param name=\"limit\">3</param>";

        Content.ToolCall call = MiniCpmToolSyntax.parsePayload(payload).getFirst();

        assertEquals("search", call.name());
        assertEquals(Map.of("query", "a < b & c\nnext", "limit", "3"), call.arguments());
    }
}
