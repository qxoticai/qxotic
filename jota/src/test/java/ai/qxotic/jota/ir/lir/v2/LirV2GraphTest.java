package ai.qxotic.jota.ir.lir.v2;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.ScalarLiteral;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import org.junit.jupiter.api.Test;

class LirV2GraphTest {

    @Test
    void addUniqueReusesNodes() {
        LirV2Graph graph = new LirV2Graph();
        V2Node left = graph.scalarConst(ScalarLiteral.ofInt(2).rawBits(), DataType.I32);
        V2Node right = graph.scalarConst(ScalarLiteral.ofInt(3).rawBits(), DataType.I32);

        V2Node first = graph.scalarBinary(BinaryOperator.ADD, left, right);
        V2Node second = graph.scalarBinary(BinaryOperator.ADD, right, left);

        assertSame(first, second);
    }

    @Test
    void constantFoldingUsesWorklist() {
        LirV2Graph graph = new LirV2Graph();
        V2Node left = graph.scalarConst(ScalarLiteral.ofInt(2).rawBits(), DataType.I32);
        V2Node right = graph.scalarConst(ScalarLiteral.ofInt(3).rawBits(), DataType.I32);

        V2Node add = graph.scalarBinary(BinaryOperator.ADD, left, right);
        graph.processWorklist();

        V2Node result = graph.resolve(add);
        assertTrue(result instanceof SConst);
    }
}
