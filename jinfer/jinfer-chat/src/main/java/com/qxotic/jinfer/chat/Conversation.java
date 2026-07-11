package com.qxotic.jinfer.chat;

import java.util.ArrayList;
import java.util.List;

/**
 * A whole conversation as the chat codec sees it: the ordered messages plus every other input that
 * shapes the prompt - the offered tools, the {@code thinking} toggle, and the reasoning {@code
 * effort}. {@link ChatTemplate#encode} is a pure function of this value.
 *
 * <p>{@code thinking} toggles the model's reasoning scaffold in the assistant tail (matching the
 * reference template's {@code enable_thinking}); models without one ignore it. {@code effort} is a
 * free-form string rendered into the system preamble by templates that support one (e.g. GPT-OSS
 * {@code Reasoning: {effort}}); blank means the model's default, and templates without an effort
 * knob ignore it. A string, not an enum, so a model adding a level never churns this type.
 */
public record Conversation(
        List<Message> messages, List<Tool> tools, boolean thinking, String effort) {

    public Conversation {
        messages = List.copyOf(messages);
        tools = List.copyOf(tools);
        effort = effort == null ? "" : effort;
    }

    /** Plain chat: no tools, thinking on (the reference templates' default), default effort. */
    public Conversation(List<Message> messages) {
        this(messages, List.of(), true, "");
    }

    /** This conversation with one more message appended. */
    public Conversation append(Message message) {
        List<Message> extended = new ArrayList<>(messages);
        extended.add(message);
        return new Conversation(extended, tools, thinking, effort);
    }
}
