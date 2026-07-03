package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Media;

import java.util.ArrayList;
import java.util.List;

/** One conversation turn: a role plus ordered content parts (text and media, interleaved). The
 *  high-level, portable representation; templates lower it to batches. */
public record Message(Role role, List<Part> content) {

    public Message {
        if (role == null) throw new IllegalArgumentException("null role");
        content = List.copyOf(content);
    }

    public Message(Role role, String text) {
        this(role, List.of(new Part.Text(text)));
    }

    public static Message system(String text) {
        return new Message(Role.SYSTEM, text);
    }

    public static Message user(String text, Media... media) {
        List<Part> parts = new ArrayList<>();
        parts.add(new Part.Text(text));
        for (Media m : media) parts.add(new Part.Blob(m));
        return new Message(Role.USER, parts);
    }

    public static Message assistant(String text) {
        return new Message(Role.ASSISTANT, text);
    }

    /** The concatenated text parts — the display/projection view (media excluded). */
    public String text() {
        StringBuilder sb = new StringBuilder();
        for (Part p : content) {
            if (p instanceof Part.Text t) sb.append(t.text());
        }
        return sb.toString();
    }
}
