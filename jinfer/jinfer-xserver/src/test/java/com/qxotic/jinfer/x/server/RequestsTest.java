package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class RequestsTest {

    @Test
    void responsesInputPreservesMediaAndNormalizesTextParts() {
        Map<String, Object> image =
                Map.of(
                        "type",
                        "input_image",
                        "image_url",
                        "data:image/png;base64,AA==");
        Map<String, Object> request =
                Map.of(
                        "input",
                        List.of(
                                Map.of(
                                        "role",
                                        "user",
                                        "content",
                                        List.of(
                                                image,
                                                Map.of(
                                                        "type",
                                                        "input_text",
                                                        "text",
                                                        "describe")))));

        Map<?, ?> message = (Map<?, ?>) Requests.responseInputMessages(request).getFirst();
        List<?> content = (List<?>) message.get("content");

        assertEquals(image, content.getFirst());
        assertEquals(Map.of("type", "text", "text", "describe"), content.get(1));
    }
}
