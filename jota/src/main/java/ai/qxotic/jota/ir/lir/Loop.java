package ai.qxotic.jota.ir.lir;

import java.util.Objects;

/**
 * A loop iterating from 0 to bound-1. Can be marked as parallel (independent iterations) or
 * sequential.
 */
public record Loop(String indexName, IndexExpr bound, boolean isParallel, LIRNode body)
        implements LIRNode {

    public Loop {
        Objects.requireNonNull(indexName, "indexName cannot be null");
        if (indexName.isEmpty()) {
            throw new IllegalArgumentException("indexName cannot be empty");
        }
        Objects.requireNonNull(bound, "bound cannot be null");
        Objects.requireNonNull(body, "body cannot be null");
    }

    /** Creates a parallel loop (iterations are independent). */
    public static Loop parallel(String indexName, long bound, LIRNode body) {
        return new Loop(indexName, new IndexConst(bound), true, body);
    }

    /** Creates a parallel loop with an index expression bound. */
    public static Loop parallel(String indexName, IndexExpr bound, LIRNode body) {
        return new Loop(indexName, bound, true, body);
    }

    /** Creates a sequential loop (iterations may depend on each other). */
    public static Loop sequential(String indexName, long bound, LIRNode body) {
        return new Loop(indexName, new IndexConst(bound), false, body);
    }

    /** Creates a sequential loop with an index expression bound. */
    public static Loop sequential(String indexName, IndexExpr bound, LIRNode body) {
        return new Loop(indexName, bound, false, body);
    }
}
