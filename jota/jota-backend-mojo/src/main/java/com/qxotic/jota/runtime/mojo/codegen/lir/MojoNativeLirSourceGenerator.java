package com.qxotic.jota.runtime.mojo.codegen.lir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.Block;
import com.qxotic.jota.ir.lir.BufferRef;
import com.qxotic.jota.ir.lir.IBinary;
import com.qxotic.jota.ir.lir.IConst;
import com.qxotic.jota.ir.lir.IFromScalar;
import com.qxotic.jota.ir.lir.IVar;
import com.qxotic.jota.ir.lir.IndexBinaryOp;
import com.qxotic.jota.ir.lir.LIRExprGraph;
import com.qxotic.jota.ir.lir.LIRExprKind;
import com.qxotic.jota.ir.lir.LIRExprNode;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRInput;
import com.qxotic.jota.ir.lir.LoopIterArg;
import com.qxotic.jota.ir.lir.SBinary;
import com.qxotic.jota.ir.lir.SCast;
import com.qxotic.jota.ir.lir.SConst;
import com.qxotic.jota.ir.lir.SFromIndex;
import com.qxotic.jota.ir.lir.SInput;
import com.qxotic.jota.ir.lir.SLoad;
import com.qxotic.jota.ir.lir.SRef;
import com.qxotic.jota.ir.lir.STernary;
import com.qxotic.jota.ir.lir.SUnary;
import com.qxotic.jota.ir.lir.ScalarInput;
import com.qxotic.jota.ir.lir.Store;
import com.qxotic.jota.ir.lir.StructuredFor;
import com.qxotic.jota.ir.lir.Yield;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.runtime.clike.CLikeExprSupport;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/** Native Mojo syntax emitter for LIR graphs. */
final class MojoNativeLirSourceGenerator {

    private final LIRGraph graph;
    private final ScratchLayout scratchLayout;
    private final String kernelName;
    private final LIRExprGraph exprGraph;
    private final List<String> lines = new ArrayList<>();
    private final Map<BufferRef, String> buffers = new IdentityHashMap<>();
    private final Map<Integer, String> scalarInputs = new HashMap<>();
    private final Map<LIRExprNode, String> cachedScalars = new IdentityHashMap<>();
    private int indent;
    private int tempId;

    MojoNativeLirSourceGenerator(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        this.graph = graph;
        this.scratchLayout = scratchLayout;
        this.kernelName = kernelName;
        this.exprGraph = graph.exprGraph();
    }

    String generate() {
        lines.clear();
        indent = 0;
        tempId = 0;
        buffers.clear();
        scalarInputs.clear();
        cachedScalars.clear();

        line("from std.gpu import block_idx, block_dim, thread_idx");
        line("from math import sqrt");
        line("");
        emitHeader();
        indent++;
        emitProlog();
        // Try to emit parallel version first, fall back to single-thread serial execution.
        if (!tryEmitParallelBody(graph.body())) {
            line("if block_idx.x == 0 and thread_idx.x == 0:");
            indent++;
            emitNode(graph.body());
            indent--;
        }
        indent--;
        line("");
        return String.join("\n", lines);
    }

    private void emitHeader() {
        line("fn " + kernelName + "(");
        List<String> args = new ArrayList<>();
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            if (input instanceof BufferRef ref) {
                args.add("    input" + i + ": " + mojoPointerType(ref.dataType()));
                buffers.put(ref, "input" + i);
            } else if (input instanceof ScalarInput scalar) {
                args.add("    scalar" + i + ": " + mojoType(scalar.dataType()));
                scalarInputs.put(scalar.id(), "scalar" + i);
            }
        }
        for (int i = 0; i < graph.outputs().size(); i++) {
            BufferRef out = graph.outputs().get(i);
            args.add("    output" + i + ": " + mojoPointerType(out.dataType()));
            buffers.put(out, "output" + i);
        }
        args.add("    scratch_ptr: UInt64");
        for (int i = 0; i < args.size(); i++) {
            String suffix = i + 1 == args.size() ? "" : ",";
            line(args.get(i) + suffix);
        }
        line("):");
    }

    private void emitProlog() {
        line("gid = (Int(block_idx.x) * Int(block_dim.x)) + Int(thread_idx.x)");
        if (scratchLayout.requiresScratch()) {
            line("scratch_u8 = UnsafePointer[UInt8, MutAnyOrigin].from_address(scratch_ptr)");
            int slot = 0;
            for (Map.Entry<BufferRef, Long> e : scratchLayout.offsets().entrySet()) {
                BufferRef ref = e.getKey();
                long offset = e.getValue();
                String name = "scratch" + slot++;
                String ptr =
                        "UnsafePointer["
                                + mojoScalarType(ref.dataType())
                                + ", MutAnyOrigin].from_address(scratch_ptr + UInt64("
                                + offset
                                + " ))";
                line(name + " = " + ptr);
                buffers.put(ref, name);
            }
        }
    }

    /**
     * Attempts to emit GPU-parallel body using strided loop pattern. Returns true if parallel
     * emission succeeded, false to fall back to serial.
     */
    private boolean tryEmitParallelBody(LIRExprNode body) {
        // Check if this is a perfect loop nest we can parallelize
        StructuredFor loop = extractTopLevelLoop(body);
        if (loop == null) {
            return false;
        }

        // Collect perfect nest of loops
        List<StructuredFor> loops = new ArrayList<>();
        StructuredFor current = loop;
        while (true) {
            if (!canParallelizeLoop(current)) {
                return false;
            }
            loops.add(current);
            Block innerBody = current.body();
            StructuredFor nested = extractNestedLoop(innerBody);
            if (nested != null) {
                current = nested;
            } else {
                break;
            }
        }

        // Get the innermost body
        Block innermostBody = loops.get(loops.size() - 1).body();

        // Calculate total iterations
        String totalVar = "total_" + tempId++;
        line("");
        line("# Calculate total elements");
        line(totalVar + " = 1");
        for (StructuredFor forLoop : loops) {
            String ub = emitIndexExpr(forLoop.upperBound());
            line(totalVar + " *= " + ub);
        }

        // Emit single-lane mapping (launch config covers total work elements).
        String idxVar = "linear_" + tempId++;
        line("");
        line("# GPU-parallel linear mapping");
        line(idxVar + " = gid");
        line("if " + idxVar + " < " + totalVar + ":");
        indent++;

        // De-linearize index into loop indices
        String tempIdx = "temp_" + tempId++;
        line(tempIdx + " = " + idxVar);
        for (int i = loops.size() - 1; i >= 0; i--) {
            StructuredFor forLoop = loops.get(i);
            String extent = emitIndexExpr(forLoop.upperBound());
            String loopIdx = forLoop.indexName();
            line(loopIdx + " = " + tempIdx + " % " + extent);
            line(tempIdx + " = " + tempIdx + " // " + extent);
        }

        // Emit the innermost body (excluding yield)
        for (LIRExprNode stmt : innermostBody.statements()) {
            if (stmt instanceof Yield) {
                continue;
            }
            emitNode(stmt);
        }

        indent--;
        return true;
    }

    private StructuredFor extractTopLevelLoop(LIRExprNode body) {
        if (body instanceof StructuredFor forLoop) {
            return forLoop;
        }
        if (body instanceof Block block
                && block.statements().size() == 1
                && block.statements().get(0) instanceof StructuredFor forLoop) {
            return forLoop;
        }
        return null;
    }

    private StructuredFor extractNestedLoop(Block body) {
        if (body.statements().isEmpty()) {
            return null;
        }
        if (body.statements().size() == 1
                && body.statements().getFirst() instanceof StructuredFor nested) {
            return nested;
        }
        if (body.statements().size() == 2
                && body.statements().getFirst() instanceof StructuredFor nested
                && body.statements().getLast() instanceof Yield yield
                && yield.values().isEmpty()) {
            return nested;
        }
        return null;
    }

    private boolean canParallelizeLoop(StructuredFor loop) {
        // Check no iter args (reductions not supported in parallel mode)
        if (!loop.iterArgs().isEmpty()) {
            return false;
        }
        // Check bounds start at 0 and step by 1
        if (!(loop.lowerBound() instanceof IConst lb) || lb.value() != 0) {
            return false;
        }
        if (!(loop.step() instanceof IConst step) || step.value() != 1) {
            return false;
        }
        return true;
    }

    private void emitNode(LIRExprNode node) {
        LIRExprNode resolved = exprGraph.resolve(node);
        switch (resolved.kind()) {
            case BLOCK -> {
                for (LIRExprNode stmt : ((Block) resolved).statements()) {
                    emitNode(stmt);
                }
            }
            case STORE -> emitStore((Store) resolved);
            case STRUCTURED_FOR -> emitStructuredFor((StructuredFor) resolved);
            case YIELD -> {}
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported node kind: " + resolved.kind());
        }
    }

    private void emitStructuredFor(StructuredFor loop) {
        for (LoopIterArg arg : loop.iterArgs()) {
            line(arg.name() + " = " + emitScalarExpr(arg.init()));
        }
        String idx = loop.indexName();
        String lb = emitIndexExpr(loop.lowerBound());
        String ub = emitIndexExpr(loop.upperBound());
        String step = emitIndexExpr(loop.step());
        line("for " + idx + " in range(" + lb + ", " + ub + ", " + step + "):");
        indent++;
        Block body = loop.body();
        Yield yield = (Yield) body.statements().getLast();
        for (int i = 0; i < body.statements().size() - 1; i++) {
            emitNode(body.statements().get(i));
        }
        for (int i = 0; i < loop.iterArgs().size(); i++) {
            LoopIterArg arg = loop.iterArgs().get(i);
            String tmp = arg.name() + "_next" + (tempId++);
            line(tmp + " = " + emitScalarExpr(yield.values().get(i)));
            line(arg.name() + " = " + tmp);
        }
        indent--;
    }

    private void emitStore(Store store) {
        String bufferName = requireBuffer(store.buffer());
        String idx = byteOffsetToElementIndex(store.buffer(), emitIndexExpr(store.offset()));
        String value = emitScalarExpr(store.value());
        line(bufferName + "[" + idx + "] = " + value);
    }

    private String emitScalarExpr(LIRExprNode node) {
        LIRExprNode resolved = exprGraph.resolve(node);
        if (resolved.kind() == LIRExprKind.S_CONST
                || resolved.kind() == LIRExprKind.S_INPUT
                || resolved.kind() == LIRExprKind.S_REF
                || resolved.kind() == LIRExprKind.S_LOAD
                || resolved.kind() == LIRExprKind.S_FROM_INDEX) {
            return emitScalarInline(resolved);
        }
        if (resolved.useCount() > 1
                && (resolved.kind() == LIRExprKind.S_UNARY
                        || resolved.kind() == LIRExprKind.S_BINARY
                        || resolved.kind() == LIRExprKind.S_TERNARY
                        || resolved.kind() == LIRExprKind.S_CAST)) {
            String cached = cachedScalars.get(resolved);
            if (cached != null) {
                return cached;
            }
            String expr = emitScalarInline(resolved);
            String tmp = "t" + tempId++;
            line(tmp + " = " + expr);
            cachedScalars.put(resolved, tmp);
            return tmp;
        }
        return emitScalarInline(resolved);
    }

    private String emitScalarInline(LIRExprNode node) {
        return switch (node.kind()) {
            case S_CONST -> scalarLiteral((SConst) node);
            case S_INPUT -> requireScalar(((SInput) node).inputId());
            case S_REF -> ((SRef) node).name();
            case S_LOAD -> {
                SLoad load = (SLoad) node;
                String b = requireBuffer(load.buffer());
                String idx = byteOffsetToElementIndex(load.buffer(), emitIndexExpr(load.offset()));
                yield b + "[" + idx + "]";
            }
            case S_FROM_INDEX -> {
                SFromIndex fromIndex = (SFromIndex) node;
                String indexExpr = emitIndexExpr(fromIndex.indexExpr());
                yield castExpr(DataType.I64, fromIndex.dataType(), indexExpr);
            }
            case S_UNARY -> emitUnary((SUnary) node);
            case S_BINARY -> emitBinary((SBinary) node);
            case S_TERNARY -> emitTernary((STernary) node);
            case S_CAST -> emitCast((SCast) node);
            default -> throw new IllegalStateException("Expected scalar node, got: " + node.kind());
        };
    }

    private String emitUnary(SUnary unary) {
        String in = emitScalarExpr(unary.input());
        return switch (unary.op()) {
            case NEGATE -> "-(" + in + ")";
            case ABS -> "(" + in + " if " + in + " >= 0 else -" + in + ")";
            case EXP -> "(2.718281828459045 ** " + in + ")"; // Approximate e^x
            case LOG -> "(log2(" + in + ") * 0.6931471805599453)"; // Approximate ln(x)
            case SQRT -> "sqrt(" + in + ")";
            case SQUARE -> "(" + in + " * " + in + ")";
            case SIN -> "sin(" + in + ")";
            case COS -> "cos(" + in + ")";
            case TAN -> "tan(" + in + ")";
            case TANH -> "tanh(" + in + ")";
            case RECIPROCAL -> {
                String one = unary.input().dataType() == DataType.FP32 ? "Float32(1.0)" : "1.0";
                yield "(" + one + " / " + in + ")";
            }
            case LOGICAL_NOT -> "(not (" + truthy(in, unary.input().dataType()) + "))";
            case BITWISE_NOT -> "~(" + in + ")";
        };
    }

    private String emitBinary(SBinary binary) {
        String l = emitScalarExpr(binary.left());
        String r = emitScalarExpr(binary.right());
        BinaryOperator op = binary.op();
        return switch (op) {
            case ADD -> "(" + l + " + " + r + ")";
            case SUBTRACT -> "(" + l + " - " + r + ")";
            case MULTIPLY -> "(" + l + " * " + r + ")";
            case DIVIDE -> "(" + l + " / " + r + ")";
            case MIN -> "(" + r + " if " + l + " > " + r + " else " + l + ")";
            case MAX -> "(" + r + " if " + l + " < " + r + " else " + l + ")";
            case POW -> "(" + l + " ** " + r + ")";
            case LOGICAL_AND ->
                    "("
                            + truthy(l, binary.left().dataType())
                            + " and "
                            + truthy(r, binary.right().dataType())
                            + ")";
            case LOGICAL_OR ->
                    "("
                            + truthy(l, binary.left().dataType())
                            + " or "
                            + truthy(r, binary.right().dataType())
                            + ")";
            case LOGICAL_XOR ->
                    "("
                            + truthy(l, binary.left().dataType())
                            + " != "
                            + truthy(r, binary.right().dataType())
                            + ")";
            case BITWISE_AND -> "(" + l + " & " + r + ")";
            case BITWISE_OR -> "(" + l + " | " + r + ")";
            case BITWISE_XOR -> "(" + l + " ^ " + r + ")";
            case SHIFT_LEFT -> "(" + l + " << " + normalizedShift(binary.dataType(), r) + ")";
            case SHIFT_RIGHT -> "(" + l + " >> " + normalizedShift(binary.dataType(), r) + ")";
            case SHIFT_RIGHT_UNSIGNED ->
                    "(UInt64(" + l + ") >> " + normalizedShift(binary.dataType(), r) + ")";
            case EQUAL -> "(" + l + " == " + r + ")";
            case LESS_THAN -> "(" + l + " < " + r + ")";
        };
    }

    private String emitTernary(STernary ternary) {
        String c = emitScalarExpr(ternary.condition());
        String t = emitScalarExpr(ternary.trueValue());
        String f = emitScalarExpr(ternary.falseValue());
        return "(" + t + " if " + truthy(c, ternary.condition().dataType()) + " else " + f + ")";
    }

    private String emitCast(SCast cast) {
        return castExpr(cast.input().dataType(), cast.targetType(), emitScalarExpr(cast.input()));
    }

    private String emitIndexExpr(LIRExprNode node) {
        LIRExprNode resolved = exprGraph.resolve(node);
        return switch (resolved.kind()) {
            case I_CONST -> Long.toString(((IConst) resolved).value());
            case I_VAR -> ((IVar) resolved).name();
            case I_BINARY -> {
                IBinary b = (IBinary) resolved;
                yield "("
                        + emitIndexExpr(b.left())
                        + " "
                        + indexOp(b.op())
                        + " "
                        + emitIndexExpr(b.right())
                        + ")";
            }
            case I_FROM_SCALAR ->
                    "Int(" + emitScalarExpr(((IFromScalar) resolved).scalarExpr()) + ")";
            default ->
                    throw new IllegalStateException("Expected index node, got: " + resolved.kind());
        };
    }

    private String indexOp(IndexBinaryOp op) {
        return switch (op) {
            case ADD -> "+";
            case SUBTRACT -> "-";
            case MULTIPLY -> "*";
            case DIVIDE -> "/";
            case MODULO -> "%";
            case BITWISE_AND -> "&";
            case BITWISE_XOR -> "^";
            case SHIFT_LEFT -> "<<";
            case SHIFT_RIGHT -> ">>";
            case UNSIGNED_SHIFT_RIGHT -> ">>";
        };
    }

    private String requireBuffer(BufferRef ref) {
        String name = buffers.get(ref);
        if (name == null) {
            throw new IllegalStateException("Missing buffer mapping for " + ref);
        }
        return name;
    }

    private String requireScalar(int id) {
        String name = scalarInputs.get(id);
        if (name == null) {
            throw new IllegalStateException("Missing scalar input " + id);
        }
        return name;
    }

    private String byteOffsetToElementIndex(BufferRef buffer, String offsetExpr) {
        long byteSize = buffer.dataType().byteSize();
        if (byteSize == 1) {
            return "Int(" + offsetExpr + ")";
        }
        // Use regular division and let Int() truncate, or use // for floor division
        // Both should work for positive offsets
        return "Int((" + offsetExpr + ") / " + byteSize + ")";
    }

    private String scalarLiteral(SConst c) {
        DataType t = c.dataType();
        long bits = c.rawBits();
        if (t == DataType.BOOL) {
            return bits == 0 ? "False" : "True";
        }
        if (t == DataType.I8 || t == DataType.I16 || t == DataType.I32 || t == DataType.I64) {
            return Long.toString(bits);
        }
        if (t == DataType.FP32) {
            String literal = CLikeExprSupport.formatFloatLiteral(Float.intBitsToFloat((int) bits));
            if (literal.endsWith("f")) {
                literal = literal.substring(0, literal.length() - 1);
            }
            return "Float32(" + literal + ")";
        }
        if (t == DataType.FP64) {
            return CLikeExprSupport.formatDoubleLiteral(Double.longBitsToDouble(bits));
        }
        return "0";
    }

    private String castExpr(DataType source, DataType target, String expr) {
        if (source == target) {
            return expr;
        }
        if (target == DataType.BOOL) {
            return "(" + truthy(expr, source) + ")";
        }
        if (target == DataType.I8) {
            return "Int8(" + expr + ")";
        }
        if (target == DataType.I16) {
            return "Int16(" + expr + ")";
        }
        if (target == DataType.I32) {
            return "Int32(" + expr + ")";
        }
        if (target == DataType.FP32) {
            return "Float32(" + expr + ")";
        }
        if (target == DataType.FP64) {
            return "Float64(" + expr + ")";
        }
        if (target == DataType.I64) {
            return "Int(" + expr + ")";
        }
        return expr;
    }

    private String truthy(String expr, DataType type) {
        if (type == DataType.BOOL) {
            return expr;
        }
        return "(" + expr + " != 0)";
    }

    private String normalizedShift(DataType type, String right) {
        return CLikeExprSupport.normalizedShift(type, right);
    }

    private String mojoType(DataType type) {
        if (type == DataType.BOOL) {
            return "Bool";
        }
        if (type == DataType.I8) {
            return "Int8";
        }
        if (type == DataType.I16) {
            return "Int16";
        }
        if (type == DataType.I32) {
            return "Int32";
        }
        if (type == DataType.I64) {
            return "Int";
        }
        if (type == DataType.FP32) {
            return "Float32";
        }
        if (type == DataType.FP64) {
            return "Float64";
        }
        if (type == DataType.FP16) {
            return "Float16";
        }
        if (type == DataType.BF16) {
            return "BFloat16";
        }
        return "Int";
    }

    private String mojoScalarType(DataType type) {
        return switch (mojoType(type)) {
            case "Int" -> "Int64";
            default -> mojoType(type);
        };
    }

    private String mojoPointerType(DataType type) {
        return "UnsafePointer[" + mojoScalarType(type) + ", MutAnyOrigin]";
    }

    private void line(String text) {
        lines.add("    ".repeat(Math.max(0, indent)) + text);
    }
}
