package com.qxotic.jota.stablehlo.attr;

/** Shared spelling helpers for attribute printing. */
public final class AttrSyntax {

    private AttrSyntax() {}

    /** A dictionary key: bare identifier when possible, quoted string otherwise. */
    public static String key(String name) {
        if (!name.isEmpty() && isBareIdentifier(name)) {
            return name;
        }
        return quote(name);
    }

    public static boolean isBareIdentifier(String s) {
        char c0 = s.charAt(0);
        if (!(Character.isLetter(c0) || c0 == '_')) {
            return false;
        }
        for (int i = 1; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!(Character.isLetterOrDigit(c) || c == '_' || c == '$' || c == '.')) {
                return false;
            }
        }
        return true;
    }

    /** MLIR string literal; characters are bytes (0-255), non-printables use {@code \XX}. */
    public static String quote(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' || c == '\\') {
                sb.append('\\').append(c);
            } else if (c == '\n') {
                sb.append("\\n");
            } else if (c == '\t') {
                sb.append("\\t");
            } else if (c < 0x20 || c >= 0x7F) {
                sb.append('\\').append(String.format("%02X", (int) c & 0xFF));
            } else {
                sb.append(c);
            }
        }
        return sb.append('"').toString();
    }
}
