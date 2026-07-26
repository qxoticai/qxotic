package com.qxotic.jinfer.langchain4j;

import com.qxotic.toknroll.Tokenizer;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.TokenCountEstimator;

/**
 * Token counting over the model's OWN tokenizer: text counts are exact (toknroll {@code
 * countTokens} - the real vocabulary, not a heuristic); message counts sum each message's visible
 * text. Deliberately scaffold- and media-exclusive: chat-template markers and media positions add a
 * few percent that the output-headroom margin every consumer holds (memory budgets, splitter
 * ceilings) absorbs - exact-including-scaffold counting was designed and parked; see the project
 * log for the reasoning and the additive upgrade path.
 */
final class Estimators implements TokenCountEstimator {

    private final Tokenizer tokenizer;

    Estimators(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public int estimateTokenCountInText(String text) {
        return tokenizer.countTokens(text);
    }

    @Override
    public int estimateTokenCountInMessage(ChatMessage message) {
        return switch (message) {
            case SystemMessage s -> estimateTokenCountInText(s.text());
            case UserMessage u -> {
                int sum = 0;
                for (Content c : u.contents()) {
                    if (c instanceof TextContent t) sum += estimateTokenCountInText(t.text());
                }
                yield sum;
            }
            case AiMessage a -> {
                int sum = a.text() == null ? 0 : estimateTokenCountInText(a.text());
                if (a.hasToolExecutionRequests()) {
                    for (var call : a.toolExecutionRequests()) {
                        sum += estimateTokenCountInText(call.name());
                        if (call.arguments() != null) {
                            sum += estimateTokenCountInText(call.arguments());
                        }
                    }
                }
                yield sum;
            }
            case ToolExecutionResultMessage t -> estimateTokenCountInText(t.text());
            default -> 0;
        };
    }

    @Override
    public int estimateTokenCountInMessages(Iterable<ChatMessage> messages) {
        int sum = 0;
        for (ChatMessage m : messages) sum += estimateTokenCountInMessage(m);
        return sum;
    }
}
