package com.qxotic.jota.ir.tir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.List;
import org.junit.jupiter.api.Test;

class TIRConstantFoldingPassTest {

    @Test
    void foldsConstantsThroughViewOperations() {
        TIRNode left = slicedBroadcast(2);
        TIRNode right = slicedBroadcast(3);
        TIRNode product = new BinaryOp(BinaryOperator.MULTIPLY, left, right);
        TIRNode sum =
                new ReductionOp(
                        ReductionOperator.SUM, product, new int[] {0}, false, DataType.FP32);

        TIRNode result =
                new TIRConstantFoldingPass()
                        .run(new TIRGraph(List.of(), List.of(sum)))
                        .outputs()
                        .getFirst();

        ScalarConstant constant = assertInstanceOf(ScalarConstant.class, result);
        assertEquals(Shape.scalar(), constant.shape());
        assertEquals(12.0f, Float.intBitsToFloat((int) constant.rawBits()));
    }

    private static TIRNode slicedBroadcast(float value) {
        TIRNode scalar = ScalarConstant.of(Float.floatToRawIntBits(value), DataType.FP32);
        TIRNode broadcast = new ViewTransform(scalar, new ViewOperation.Broadcast(Shape.of(4)));
        return new ViewTransform(broadcast, new ViewOperation.Slice(0, 1, 4, 2));
    }
}
