package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.ScalarLiteral;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import org.junit.jupiter.api.Test;

class LIRExprGraphTest {

    @Test
    void addUniqueReusesNodes() {
        LIRExprGraph graph = new LIRExprGraph();
        LIRExprNode left = graph.scalarConst(ScalarLiteral.ofInt(2).rawBits(), DataType.I32);
        LIRExprNode right = graph.scalarConst(ScalarLiteral.ofInt(3).rawBits(), DataType.I32);

        LIRExprNode first = graph.scalarBinary(BinaryOperator.ADD, left, right);
        LIRExprNode second = graph.scalarBinary(BinaryOperator.ADD, right, left);

        assertSame(first, second);
    }

    @Test
    void constantFoldingUsesWorklist() {
        LIRExprGraph graph = new LIRExprGraph();
        LIRExprNode left = graph.scalarConst(ScalarLiteral.ofInt(2).rawBits(), DataType.I32);
        LIRExprNode right = graph.scalarConst(ScalarLiteral.ofInt(3).rawBits(), DataType.I32);

        LIRExprNode add = graph.scalarBinary(BinaryOperator.ADD, left, right);
        graph.processWorklist();

        LIRExprNode result = graph.resolve(add);
        assertTrue(result instanceof SConst);
    }
}
