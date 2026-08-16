package com.qxotic.jinfer.chat;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** One role-tagged message with ordered, interleaved content. */
public record Message(Role role, List<Content> content) {
    public Message {
        Objects.requireNonNull(role, "role");
        content = List.copyOf(content);
    }

    public Message(Role role, String text) {
        this(role, List.of(new Content.Text(text)));
    }

    public static Message system(String text) {
        return new Message(Role.SYSTEM, text);
    }

    public static Message user(String text, com.qxotic.jinfer.boundary.Media... media) {
        ArrayList<Content> content = new ArrayList<>(media.length + 1);
        content.add(new Content.Text(text));
        for (var value : media) content.add(new Content.Media(value));
        return new Message(Role.USER, content);
    }

    public static Message assistant(String text) {
        return new Message(Role.ASSISTANT, text);
    }

    /** Concatenated text projection; structured and media content is omitted. */
    public String text() {
        StringBuilder out = new StringBuilder();
        for (Content part : content) if (part instanceof Content.Text t) out.append(t.text());
        return out.toString();
    }
}
