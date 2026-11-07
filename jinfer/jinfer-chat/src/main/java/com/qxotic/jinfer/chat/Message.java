package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.media.Media;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * One role-tagged message with ordered, interleaved content. A reply the engine generated also
 * carries how it was said, its {@link Verbatim} record; a message anyone else wrote carries {@link
 * Verbatim#NONE}.
 */
public record Message(Role role, List<Content> content, Verbatim verbatim) {
    public Message {
        Objects.requireNonNull(role, "role");
        content = List.copyOf(content);
        Objects.requireNonNull(verbatim, "verbatim");
    }

    public Message(Role role, List<Content> content) {
        this(role, content, Verbatim.NONE);
    }

    public Message(Role role, String text) {
        this(role, List.of(new Content.Text(text)));
    }

    public static Message system(String text) {
        return new Message(Role.SYSTEM, text);
    }

    public static Message user(String text, Media... media) {
        ArrayList<Content> content = new ArrayList<>(media.length + 1);
        content.add(new Content.Text(text));
        for (var value : media) content.add(new Content.Media(value));
        return new Message(Role.USER, content);
    }

    public static Message assistant(String text) {
        return new Message(Role.ASSISTANT, text);
    }

    /** The model's speech, if this reply spoke: the audio the message holds. */
    public Optional<Media.Audio> speech() {
        for (Content part : content)
            if (part instanceof Content.Media m && m.media() instanceof Media.Audio audio)
                return Optional.of(audio);
        return Optional.empty();
    }

    /** Concatenated text projection; structured and media content is omitted. */
    public String text() {
        StringBuilder out = new StringBuilder();
        for (Content part : content) if (part instanceof Content.Text t) out.append(t.text());
        return out.toString();
    }

    /** The reasoning runs joined, in order; empty when the message holds none. */
    public String reasoning() {
        StringBuilder out = new StringBuilder();
        for (Content part : content) if (part instanceof Content.Reasoning r) out.append(r.text());
        return out.toString();
    }
}
