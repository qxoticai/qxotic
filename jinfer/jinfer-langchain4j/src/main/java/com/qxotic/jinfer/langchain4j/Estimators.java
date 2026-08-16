package com.qxotic.jinfer.langchain4j;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.boundary.Media;
import com.qxotic.jinfer.boundary.media.VideoSampler;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.toknroll.Tokenizer;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
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
    private final VideoSampler videoSampler; // the MODEL's sampler: counts what chat ingests

    Estimators(
            Tokenizer tokenizer, ToIntFunction<Media> mediaPositions, VideoSampler videoSampler) {
        this.tokenizer = tokenizer;
        this.mediaPositions = mediaPositions;
        this.videoSampler = videoSampler;
    }

    @Override
    public int estimateTokenCountInText(String text) {
        return tokenizer.countTokens(text);
    }

    @Override
    public int estimateTokenCountInMessage(ChatMessage message) {
        // a model that cannot ingest media refuses BEFORE any decode: the converter below
        // decodes eagerly (a video would even consult this estimator's absent sampler), so
        // waiting for countParts would pay hashing and decode I/O just to throw - or NPE first
        if (mediaPositions == null && message instanceof UserMessage u) {
            for (dev.langchain4j.data.message.Content c : u.contents()) {
                if (!(c instanceof TextContent)) {
                    throw new UnsupportedFeatureException(
                            "this model cannot ingest media, so media tokens cannot be counted");
                }
            }
        }
        int sum = 0;
        for (Message m : Mappings.toMessages(List.of(message), videoSampler)) {
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

    private int countParts(List<com.qxotic.jinfer.chat.Content> parts) {
        int sum = 0;
        for (com.qxotic.jinfer.chat.Content part : parts) {
            sum +=
                    switch (part) {
                        case com.qxotic.jinfer.chat.Content.Text t ->
                                estimateTokenCountInText(t.text());
                        case com.qxotic.jinfer.chat.Content.Media b -> {
                            if (mediaPositions == null) {
                                throw new UnsupportedFeatureException(
                                        "this model cannot ingest media, so media tokens cannot"
                                                + " be counted");
                            }
                            yield mediaPositions.applyAsInt(b.value());
                        }
                        case com.qxotic.jinfer.chat.Content.ToolCall c ->
                                estimateTokenCountInText(c.name())
                                        + estimateTokenCountInText(Json.stringify(c.arguments()));
                        case com.qxotic.jinfer.chat.Content.ToolResult r ->
                                estimateTokenCountInText(r.text());
                        case com.qxotic.jinfer.chat.Content.Reasoning ignored ->
                                0; // not re-prompted by default
                    };
        }
        return sum;
    }
}
