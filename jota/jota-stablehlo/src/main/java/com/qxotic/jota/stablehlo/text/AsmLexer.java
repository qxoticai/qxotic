package com.qxotic.jota.stablehlo.text;

import java.math.BigInteger;

/**
 * Character-level scanner for MLIR assembly. Parsing is done directly on characters because parts
 * of the syntax are context sensitive (tensor dimensions {@code 2x3xf32}, dense literals, attribute
 * bodies kept verbatim).
 */
class AsmLexer {

    protected final String src;
    protected final String sourceName;
    protected int pos;

    AsmLexer(String src, String sourceName) {
        this.src = src;
        this.sourceName = sourceName;
    }

    // ------------------------------------------------------------------ errors

    ParseException error(String message) {
        return errorAt(pos, message);
    }

    ParseException errorAt(int at, String message) {
        int line = 1;
        int lineStart = 0;
        for (int i = 0; i < at && i < src.length(); i++) {
            if (src.charAt(i) == '\n') {
                line++;
                lineStart = i + 1;
            }
        }
        int lineEnd = src.indexOf('\n', lineStart);
        if (lineEnd < 0) {
            lineEnd = src.length();
        }
        String text = src.substring(lineStart, Math.min(lineEnd, lineStart + 240));
        int col = at - lineStart + 1;
        String caret = " ".repeat(Math.max(0, Math.min(col - 1, 240))) + "^";
        return new ParseException(message, sourceName, line, col, text + "\n" + caret);
    }

    // ------------------------------------------------------------------ whitespace

    /** Skips whitespace and {@code //} comments. */
    void ws() {
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos++;
            } else if (c == '/' && pos + 1 < src.length() && src.charAt(pos + 1) == '/') {
                int nl = src.indexOf('\n', pos);
                pos = nl < 0 ? src.length() : nl + 1;
            } else {
                break;
            }
        }
    }

    boolean atEnd() {
        ws();
        return pos >= src.length();
    }

    char peekChar() {
        ws();
        return pos < src.length() ? src.charAt(pos) : '\0';
    }

    /** Raw character at {@code pos + offset} (no whitespace skipping). */
    char charAt(int offset) {
        int i = pos + offset;
        return i < src.length() ? src.charAt(i) : '\0';
    }

    boolean peek(String s) {
        ws();
        return src.startsWith(s, pos);
    }

    /** True when the next token is the keyword {@code kw} (not a prefix of a longer word). */
    boolean peekKeyword(String kw) {
        ws();
        if (!src.startsWith(kw, pos)) {
            return false;
        }
        int end = pos + kw.length();
        return end >= src.length() || !isIdentChar(src.charAt(end));
    }

    boolean tryConsume(String s) {
        if (peek(s)) {
            pos += s.length();
            return true;
        }
        return false;
    }

    boolean tryKeyword(String kw) {
        if (peekKeyword(kw)) {
            pos += kw.length();
            return true;
        }
        return false;
    }

    void expect(String s) {
        if (!tryConsume(s)) {
            throw error("expected '" + s + "'");
        }
    }

    void expectKeyword(String kw) {
        if (!tryKeyword(kw)) {
            throw error("expected '" + kw + "'");
        }
    }

    // ------------------------------------------------------------------ identifiers

    static boolean isIdentStart(char c) {
        return Character.isLetter(c) || c == '_';
    }

    static boolean isIdentChar(char c) {
        return Character.isLetterOrDigit(c) || c == '_' || c == '$' || c == '.';
    }

    boolean peekIdent() {
        ws();
        return pos < src.length() && isIdentStart(src.charAt(pos));
    }

    /** Bare identifier {@code [A-Za-z_][A-Za-z0-9_$.]*}. */
    String ident() {
        ws();
        int start = pos;
        if (pos >= src.length() || !isIdentStart(src.charAt(pos))) {
            throw error("expected identifier");
        }
        while (pos < src.length() && isIdentChar(src.charAt(pos))) {
            pos++;
        }
        return src.substring(start, pos);
    }

    /**
     * Identifier after a sigil ({@code %}, {@code ^}, {@code @}, {@code #}): may start with a
     * digit.
     */
    String suffixId() {
        int start = pos;
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (Character.isLetterOrDigit(c) || c == '_' || c == '$' || c == '.' || c == '-') {
                pos++;
            } else {
                break;
            }
        }
        if (start == pos) {
            throw error("expected identifier");
        }
        return src.substring(start, pos);
    }

    /** {@code @name} or {@code @"quoted name"}; returns the name without the sigil. */
    String symbolName() {
        ws();
        expect("@");
        if (pos < src.length() && src.charAt(pos) == '"') {
            return string();
        }
        return suffixId();
    }

    /** SSA name without the {@code %} sigil. */
    String valueName() {
        ws();
        expect("%");
        return suffixId();
    }

    // ------------------------------------------------------------------ literals

    /** MLIR string literal; escapes {@code \n \t \\ \" \XX}; chars hold bytes. */
    String string() {
        ws();
        if (pos >= src.length() || src.charAt(pos) != '"') {
            throw error("expected string literal");
        }
        pos++;
        StringBuilder sb = new StringBuilder();
        while (true) {
            if (pos >= src.length()) {
                throw error("unterminated string");
            }
            char c = src.charAt(pos++);
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                char e = src.charAt(pos++);
                switch (e) {
                    case 'n' -> sb.append('\n');
                    case 't' -> sb.append('\t');
                    case '\\' -> sb.append('\\');
                    case '"' -> sb.append('"');
                    default -> {
                        int hi = Character.digit(e, 16);
                        int lo = Character.digit(src.charAt(pos++), 16);
                        if (hi < 0 || lo < 0) {
                            throw error("bad escape sequence");
                        }
                        sb.append((char) (hi * 16 + lo));
                    }
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    boolean peekNumber() {
        ws();
        char c = charAt(0);
        if (Character.isDigit(c)) {
            return true;
        }
        return (c == '-' || c == '+') && Character.isDigit(charAt(1));
    }

    /**
     * Raw numeric token: decimal integer, decimal float ({@code 1.5e-3}), or hexadecimal ({@code
     * 0x7FC00000}), with optional sign.
     */
    String numberToken() {
        ws();
        int start = pos;
        if (charAt(0) == '-' || charAt(0) == '+') {
            pos++;
        }
        if (charAt(0) == '0' && (charAt(1) == 'x' || charAt(1) == 'X')) {
            pos += 2;
            while (Character.digit(charAt(0), 16) >= 0) {
                pos++;
            }
            return src.substring(start, pos);
        }
        int digitsStart = pos;
        while (Character.isDigit(charAt(0))) {
            pos++;
        }
        if (pos == digitsStart) {
            throw errorAt(start, "expected number");
        }
        if (charAt(0) == '.') {
            pos++;
            while (Character.isDigit(charAt(0))) {
                pos++;
            }
        }
        if ((charAt(0) == 'e' || charAt(0) == 'E')
                && (Character.isDigit(charAt(1))
                        || ((charAt(1) == '-' || charAt(1) == '+')
                                && Character.isDigit(charAt(2))))) {
            pos += 2;
            while (Character.isDigit(charAt(0))) {
                pos++;
            }
        }
        return src.substring(start, pos);
    }

    static boolean isFloatToken(String tok) {
        String t = tok.startsWith("-") || tok.startsWith("+") ? tok.substring(1) : tok;
        if (t.startsWith("0x") || t.startsWith("0X")) {
            return false;
        }
        return t.contains(".") || t.contains("e") || t.contains("E");
    }

    static BigInteger parseInteger(String tok) {
        String t = tok;
        boolean negative = false;
        if (t.startsWith("-") || t.startsWith("+")) {
            negative = t.charAt(0) == '-';
            t = t.substring(1);
        }
        BigInteger v =
                t.startsWith("0x") || t.startsWith("0X")
                        ? new BigInteger(t.substring(2), 16)
                        : new BigInteger(t);
        return negative ? v.negate() : v;
    }

    long integer() {
        String tok = numberToken();
        if (isFloatToken(tok)) {
            throw error("expected integer, got " + tok);
        }
        return parseInteger(tok).longValue();
    }

    /**
     * Captures a balanced region starting at the current {@code open} character up to its matching
     * close, honouring nested brackets, strings and the {@code ->} arrow; returns the text
     * including the delimiters.
     */
    String balanced(char open, char close) {
        ws();
        int start = pos;
        if (charAt(0) != open) {
            throw error("expected '" + open + "'");
        }
        int depth = 0;
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == '"') {
                string();
                continue;
            }
            if (c == '-' && charAt(1) == '>') {
                pos += 2;
                continue;
            }
            if (c == '<' || c == '(' || c == '[' || c == '{') {
                depth++;
            } else if (c == '>' || c == ')' || c == ']' || c == '}') {
                depth--;
                if (depth == 0) {
                    pos++;
                    if (c != close) {
                        throw errorAt(pos - 1, "unbalanced '" + open + "'");
                    }
                    return src.substring(start, pos);
                }
            }
            pos++;
        }
        throw errorAt(start, "unterminated '" + open + "'");
    }
}
