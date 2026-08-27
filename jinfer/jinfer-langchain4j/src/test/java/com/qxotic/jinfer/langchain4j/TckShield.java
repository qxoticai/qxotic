package com.qxotic.jinfer.langchain4j;

import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.Set;

/**
 * Non-AutoCloseable views over a shared model for the compliance kits: JUnit's params
 * autoCloseArguments closes AutoCloseable parameterized-test arguments after EVERY invocation,
 * which would kill a shared model after its first test - and a fresh 8B load per collection-time
 * {@code models()} call OOMs the fork. The AiServices kits additionally read {@code
 * supportedCapabilities()} to choose native structured output over prompt-based JSON, so the views
 * delegate it.
 */
final class TckShield {

    private TckShield() {}

    static ChatModel chat(JinferChatModel m) {
        return new ChatModel() {
            @Override
            public ChatResponse chat(ChatRequest request) {
                return m.chat(request);
            }

            @Override
            public ChatResponse doChat(ChatRequest request) {
                return m.chat(request);
            }

            @Override
            public ChatRequestParameters defaultRequestParameters() {
                return m.defaultRequestParameters();
            }

            @Override
            public Set<Capability> supportedCapabilities() {
                return m.supportedCapabilities();
            }
        };
    }

    static StreamingChatModel streaming(JinferStreamingChatModel m) {
        return new StreamingChatModel() {
            @Override
            public void chat(ChatRequest request, StreamingChatResponseHandler handler) {
                m.chat(request, handler);
            }

            @Override
            public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
                m.chat(request, handler);
            }

            @Override
            public ChatRequestParameters defaultRequestParameters() {
                return m.defaultRequestParameters();
            }

            @Override
            public Set<Capability> supportedCapabilities() {
                return m.supportedCapabilities();
            }
        };
    }
}
