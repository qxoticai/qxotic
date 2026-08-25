package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ViewTransforms;
import java.util.Optional;

/** A semantic view operation whose physical layout is derived from its input. */
public record ViewTransform(TIRNode input, ViewOperation operation) implements TIRNode {

    public ViewTransform {
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        if (operation == null) {
            throw new IllegalArgumentException("operation cannot be null");
        }
        transform(operation, inputLayout(input));
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }

    @Override
    public Shape shape() {
        return layout().shape();
    }

    public Layout layout() {
        Layout inputLayout = inputLayout();
        return transform(inputLayout)
                .map(ViewTransforms.Result::layout)
                .orElseGet(() -> Layout.rowMajor(targetShape()));
    }

    Optional<ViewTransforms.Result> transform(Layout inputLayout) {
        return transform(operation, inputLayout);
    }

    private static Optional<ViewTransforms.Result> transform(
            ViewOperation operation, Layout inputLayout) {
        return switch (operation) {
            case ViewOperation.Transpose transpose ->
                    Optional.of(ViewTransforms.permute(inputLayout, transpose.permutation()));
            case ViewOperation.Reshape reshape ->
                    ViewTransforms.reshape(inputLayout, reshape.shape());
            case ViewOperation.Unsqueeze unsqueeze ->
                    Optional.of(ViewTransforms.unsqueeze(inputLayout, unsqueeze.axis()));
            case ViewOperation.Broadcast broadcast ->
                    Optional.of(ViewTransforms.broadcast(inputLayout, broadcast.shape()));
            case ViewOperation.Expand expand ->
                    Optional.of(ViewTransforms.expand(inputLayout, expand.shape()));
            case ViewOperation.Slice slice ->
                    Optional.of(
                            ViewTransforms.slice(
                                    inputLayout,
                                    slice.axis(),
                                    slice.start(),
                                    slice.end(),
                                    slice.step()));
        };
    }

    private Layout inputLayout() {
        return inputLayout(input);
    }

    private static Layout inputLayout(TIRNode input) {
        return switch (input) {
            case TensorInput tensorInput -> tensorInput.layout();
            case ViewTransform viewTransform -> viewTransform.layout();
            default -> Layout.rowMajor(input.shape());
        };
    }

    private Shape targetShape() {
        return switch (operation) {
            case ViewOperation.Reshape reshape -> reshape.shape();
            default -> transform(Layout.rowMajor(input.shape())).orElseThrow().layout().shape();
        };
    }

    /** Returns a hint string for debugging and display. */
    public String hint() {
        return switch (operation) {
            case ViewOperation.Transpose __ -> "transpose";
            case ViewOperation.Reshape __ -> "view";
            case ViewOperation.Unsqueeze __ -> "unsqueeze";
            case ViewOperation.Broadcast __ -> "broadcast";
            case ViewOperation.Expand __ -> "expand";
            case ViewOperation.Slice __ -> "slice";
        };
    }
}
