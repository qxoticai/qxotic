package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.Layout;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class TIRStepGraphBuilder {

    private final TIRNode root;
    private final Map<TIRNode, ValueId> produced;
    private final IdentityHashMap<TIRNode, TIRNode> memo = new IdentityHashMap<>();
    private final LinkedHashMap<ScheduleInputRef, TIRNode> placeholders = new LinkedHashMap<>();
    private final BoundaryRewriter rewriter = new BoundaryRewriter();

    private int nextTensorInputId;
    private int nextScalarInputId;

    private TIRStepGraphBuilder(TIRNode root, Map<TIRNode, ValueId> produced) {
        this.root = Objects.requireNonNull(root, "root");
        this.produced = Objects.requireNonNull(produced, "produced");
    }

    static Result build(TIRNode root, Map<TIRNode, ValueId> produced) {
        return new TIRStepGraphBuilder(root, produced).build();
    }

    private Result build() {
        TIRNode output = cloneNode(root);
        List<ScheduleInputRef> inputRefs = new ArrayList<>(placeholders.size());
        List<TIRNode> inputNodes = new ArrayList<>(placeholders.size());
        for (Map.Entry<ScheduleInputRef, TIRNode> e : placeholders.entrySet()) {
            inputRefs.add(e.getKey());
            inputNodes.add(e.getValue());
        }
        return new Result(new TIRGraph(inputNodes, output), inputRefs);
    }

    private TIRNode cloneNode(TIRNode node) {
        TIRNode existing = memo.get(node);
        if (existing != null) {
            return existing;
        }

        ValueId producedValue = produced.get(node);
        if (producedValue != null && node != root) {
            TIRNode placeholder = placeholderFor(new ScheduleInputRef.ProducedValueRef(producedValue), node);
            memo.put(node, placeholder);
            return placeholder;
        }

        TIRNode cloned = node.accept(rewriter);

        memo.put(node, cloned);
        return cloned;
    }

    private TIRNode placeholderFor(ScheduleInputRef ref, TIRNode source) {
        TIRNode existing = placeholders.get(ref);
        if (existing != null) {
            return existing;
        }

        TIRNode placeholder =
                switch (ref) {
                    case ScheduleInputRef.TensorInputRef __ ->
                            new TensorInput(nextTensorInputId++, source.dataType(), layoutOf(source));
                    case ScheduleInputRef.ScalarInputRef __ ->
                            new ScalarInput(nextScalarInputId++, source.dataType(), source.shape());
                    case ScheduleInputRef.ProducedValueRef __ ->
                            new TensorInput(
                                    nextTensorInputId++,
                                    source.dataType(),
                                    Layout.rowMajor(source.shape()));
                };

        placeholders.put(ref, placeholder);
        return placeholder;
    }

    private TIRNode inputPlaceholder(TensorInput input) {
        return placeholderFor(new ScheduleInputRef.TensorInputRef(input.id()), input);
    }

    private TIRNode scalarPlaceholder(ScalarInput input) {
        return placeholderFor(new ScheduleInputRef.ScalarInputRef(input.id()), input);
    }

    private static Layout layoutOf(TIRNode node) {
        if (node instanceof TensorInput input) {
            return input.layout();
        }
        if (node instanceof ViewTransform view) {
            return view.layout();
        }
        return Layout.rowMajor(node.shape());
    }

    record Result(TIRGraph graph, List<ScheduleInputRef> inputs) {}

    private final class BoundaryRewriter extends TIRRewriter {
        @Override
        public TIRNode visitTensorInput(TensorInput node) {
            return inputPlaceholder(node);
        }

        @Override
        public TIRNode visitScalarInput(ScalarInput node) {
            return scalarPlaceholder(node);
        }

        @Override
        protected TIRNode rewriteChild(TIRNode node) {
            return cloneNode(node);
        }
    }
}
