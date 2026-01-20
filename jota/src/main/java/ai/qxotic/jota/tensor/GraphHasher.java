package ai.qxotic.jota.tensor;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.IdentityHashMap;
import java.util.Map;

public final class GraphHasher {

    private GraphHasher() {}

    public static KernelCacheKey hash(ExpressionGraph graph) {
        MessageDigest digest = messageDigest();
        Map<ExprNode, Integer> ids = new IdentityHashMap<>();
        writeNode(graph.root(), digest, ids);
        return KernelCacheKey.of(toHex(digest.digest()));
    }

    private static void writeNode(
            ExprNode node, MessageDigest digest, Map<ExprNode, Integer> ids) {
        int nodeId = ids.computeIfAbsent(node, key -> ids.size());
        update(digest, node.getClass().getSimpleName());
        update(digest, Integer.toString(nodeId));
        update(digest, node.dataType().name());
        update(digest, node.layout().toString());
        update(digest, node.device().name());

        if (node instanceof InputNode input) {
            update(digest, "input");
            update(digest, Integer.toString(input.index()));
            return;
        }
        if (node instanceof ScalarNode scalar) {
            update(digest, "scalar");
            update(digest, String.valueOf(scalar.value()));
            return;
        }
        if (node instanceof UnaryNode unary) {
            update(digest, "unary");
            update(digest, unary.op().name());
            writeNode(unary.input(), digest, ids);
            return;
        }
        if (node instanceof BinaryNode binary) {
            update(digest, "binary");
            update(digest, binary.op().name());
            writeNode(binary.left(), digest, ids);
            writeNode(binary.right(), digest, ids);
            return;
        }
        if (node instanceof CastNode cast) {
            update(digest, "cast");
            update(digest, cast.targetType().name());
            writeNode(cast.input(), digest, ids);
            return;
        }
        if (node instanceof ReductionNode reduction) {
            update(digest, "reduction");
            update(digest, reduction.op().name());
            update(digest, Integer.toString(reduction.axis()));
            update(digest, Boolean.toString(reduction.keepDims()));
            writeNode(reduction.input(), digest, ids);
        }
    }

    private static void update(MessageDigest digest, String value) {
        digest.update(value.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
    }

    private static MessageDigest messageDigest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private static String toHex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            builder.append(String.format("%02x", value));
        }
        return builder.toString();
    }
}
