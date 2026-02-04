package ai.qxotic.jota.ir.lir;

import java.util.Objects;

/**
 * A tiled loop that splits iteration into outer tiles and inner elements. Expands to: for outerName
 * in 0..(totalBound/tileSize): for innerName in outerName*tileSize..min((outerName+1)*tileSize,
 * totalBound): body
 */
public final class TiledLoop extends LIRExprNode {
    private final String outerName;
    private final String innerName;
    private final long tileSize;
    private final Block body;

    TiledLoop(
            int id, String outerName, String innerName, LIRExprNode totalBound, long tileSize, Block body) {
        super(
                id,
                LIRExprKind.TILED_LOOP,
                null,
                new LIRExprNode[] {
                    Objects.requireNonNull(totalBound, "totalBound cannot be null"),
                    Objects.requireNonNull(body, "body cannot be null")
                },
                false,
                false);
        Objects.requireNonNull(outerName, "outerName cannot be null");
        if (outerName.isEmpty()) {
            throw new IllegalArgumentException("outerName cannot be empty");
        }
        Objects.requireNonNull(innerName, "innerName cannot be null");
        if (innerName.isEmpty()) {
            throw new IllegalArgumentException("innerName cannot be empty");
        }
        if (outerName.equals(innerName)) {
            throw new IllegalArgumentException(
                    "outerName and innerName must be different: " + outerName);
        }
        if (tileSize <= 0) {
            throw new IllegalArgumentException("tileSize must be positive, got: " + tileSize);
        }
        this.outerName = outerName;
        this.innerName = innerName;
        this.tileSize = tileSize;
        this.body = body;
    }

    public String outerName() {
        return outerName;
    }

    public String innerName() {
        return innerName;
    }

    public LIRExprNode totalBound() {
        return inputs()[0];
    }

    public long tileSize() {
        return tileSize;
    }

    public Block body() {
        return body;
    }

    @Override
    public LIRExprNode canonicalize(LIRExprGraph graph) {
        return this;
    }
}
