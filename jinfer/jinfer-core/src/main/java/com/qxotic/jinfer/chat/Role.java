package com.qxotic.jinfer.chat;

/** A conversation role. String-backed on purpose: the model's template is the authority on role
 *  names ("model" vs "assistant", tool roles), so this is a label, not an enum. */
public record Role(String name) {
    public static final Role SYSTEM = new Role("system");
    public static final Role USER = new Role("user");
    public static final Role ASSISTANT = new Role("assistant");
    public static final Role TOOL = new Role("tool");

    public Role {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("empty role");
    }

    @Override
    public String toString() {
        return name;
    }
}
