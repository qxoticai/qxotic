package ai.qxotic.jota.ir.lir;

import java.util.Objects;

/**
 * A tiled loop that splits iteration into outer tiles and inner elements. Expands to: for outerName
 * in 0..(totalBound/tileSize): for innerName in outerName*tileSize..min((outerName+1)*tileSize,
 * totalBound): body
 */
public record TiledLoop(
        String outerName, String innerName, IndexExpr totalBound, long tileSize, LIRNode body)
        implements LIRNode {

    public TiledLoop {
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
        Objects.requireNonNull(totalBound, "totalBound cannot be null");
        if (tileSize <= 0) {
            throw new IllegalArgumentException("tileSize must be positive, got: " + tileSize);
        }
        Objects.requireNonNull(body, "body cannot be null");
    }

    /** Creates a tiled loop with constant total bound. */
    public static TiledLoop of(
            String outerName, String innerName, long totalBound, long tileSize, LIRNode body) {
        return new TiledLoop(outerName, innerName, new IndexConst(totalBound), tileSize, body);
    }
}
