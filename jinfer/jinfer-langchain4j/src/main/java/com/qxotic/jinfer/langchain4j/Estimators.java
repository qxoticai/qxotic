package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.toknroll.Tokenizer;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.TokenCountEstimator;
import java.util.List;
import java.util.function.ToIntFunction;

/**
 * Token counting over the model's OWN tokenizer: text counts are exact (toknroll {@code
 * countTokens} - the real vocabulary, not a heuristic); media counts come from the model's
 * preprocessing PLAN ({@code mediaPositions} - image tiers, audio frames; never an encoder run),
 * exact for plan-determined encodings. Message counts sum a message's parts through the same
 * converter the chat path uses. Deliberately scaffold-exclusive: chat-template markers add a few
 *
 * <p>Message-level counting DECODES media (base64 / local file) to size its position plan - cheap
 * for text, a real read for large media histories. Deliberate: a header-only probe would have to
 * ride the media codecs' backend seam (ImageIO/AudioSystem on the JVM, ffmpeg under native-image) -
 * machinery unearned until per-turn media counting measurably hurts.
 */
final class Estimators implements TokenCountEstimator {

    private final Tokenizer tokenizer;
    private final ToIntFunction<Media> mediaPositions; // null = this model cannot ingest media

    Estimators(Tokenizer tokenizer, ToIntFunction<Media> mediaPositions) {
        this.tokenizer = tokenizer;
        this.mediaPositions = mediaPositions;
    }

    @Override
    public int estimateTokenCountInText(String text) {
        return tokenizer.countTokens(text);
    }

    @Override
    public int estimateTokenCountInMessage(ChatMessage message) {
        int sum = 0;
        for (Message m : Mappings.toMessages(List.of(message))) {
            sum += countParts(m.content());
        }
        return sum;
    }

    @Override
    public int estimateTokenCountInMessages(Iterable<ChatMessage> messages) {
        int sum = 0;
        for (ChatMessage m : messages) sum += estimateTokenCountInMessage(m);
        return sum;
    }

    private int countParts(List<Part> parts) {
        int sum = 0;
        for (Part part : parts) {
            sum +=
                    switch (part) {
                        case Part.Text t -> estimateTokenCountInText(t.text());
                        case Part.Blob b -> {
                            if (mediaPositions == null) {
                                throw new UnsupportedOperationException(
                                        "this model cannot ingest media, so media tokens cannot"
                                                + " be counted");
                            }
                            yield mediaPositions.applyAsInt(b.media());
                        }
                        case Part.ToolCall c ->
                                estimateTokenCountInText(c.name())
                                        + estimateTokenCountInText(
                                                JsonCodec.stringify(c.arguments()));
                        case Part.ToolResult r -> estimateTokenCountInText(r.text());
                        case Part.Reasoning ignored -> 0; // not re-prompted by default
                    };
        }
        return sum;
    }
}
