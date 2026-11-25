package com.qxotic.jota.stablehlo.text;

/** A syntax or semantic error in MLIR text, with the 1-based source position. */
public final class ParseException extends RuntimeException {

    private final String sourceName;
    private final int line;
    private final int column;

    public ParseException(String message, String sourceName, int line, int column, String excerpt) {
        super(sourceName + ":" + line + ":" + column + ": " + message + "\n" + excerpt);
        this.sourceName = sourceName;
        this.line = line;
        this.column = column;
    }

    public String sourceName() {
        return sourceName;
    }

    public int line() {
        return line;
    }

    public int column() {
        return column;
    }
}
