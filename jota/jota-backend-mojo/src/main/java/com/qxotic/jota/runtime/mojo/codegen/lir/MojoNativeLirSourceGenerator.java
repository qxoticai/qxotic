package com.qxotic.jota.runtime.mojo.codegen.lir;

import com.qxotic.jota.BFloat16;
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
import com.qxotic.jota.runtime.mojo.codegen.MojoDialect;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Native Mojo syntax emitter for LIR graphs.
 *
 * <p>Locals are declared with {@code var}, which Mojo 1.x requires and earlier releases accept. A
 * {@code var} is scoped to its indented block, so the emitter tracks declared names per block
 * (redeclaring a visible name in the same block is a compile error) and forgets cached scalar
 * temporaries when the block that declared them ends.
 */
final class MojoNativeLirSourceGenerator {

    private final MojoDialect dialect;
    private final LIRGraph graph;
    private final ScratchLayout scratchLayout;
    private final String kernelName;
    private final LIRExprGraph exprGraph;
    private final List<String> lines = new ArrayList<>();
    private final Map<BufferRef, String> buffers = new IdentityHashMap<>();
    private final Map<Integer, String> scalarInputs = new HashMap<>();
    private final Map<LIRExprNode, String> cachedScalars = new IdentityHashMap<>();
    private final Deque<Scope> scopes = new ArrayDeque<>();
    private int indent;
    private int tempId;

    /** Names declared in one indented block, and the cached temporaries it introduced. */
    private record Scope(Set<String> names, List<LIRExprNode> cachedNodes) {
        Scope() {
            this(new HashSet<>(), new ArrayList<>());
        }
    }

    MojoNativeLirSourceGenerator(
            MojoDialect dialect, LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        this.dialect = Objects.requireNonNull(dialect, "dialect");
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
        scopes.clear();

        line("from " + dialect.gpuModule() + " import block_idx, block_dim, thread_idx");
        line("from std.math import cos, log2, sin, sqrt, tan, tanh");
        line("");
        emitHeader();
        enterBlock();
        emitProlog();
        // Try to emit parallel version first, fall back to single-thread serial execution.
        if (!tryEmitParallelBody(graph.body())) {
            line("if block_idx.x == 0 and thread_idx.x == 0:");
            enterBlock();
            emitNode(graph.body());
            exitBlock();
        }
        exitBlock();
        line("");
        return String.join("\n", lines);
    }

    private void emitHeader() {
        line(dialect.kernelKeyword() + " " + kernelName + "(");
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
        declare("gid", "(Int(block_idx.x) * Int(block_dim.x)) + Int(thread_idx.x)");
        if (scratchLayout.requiresScratch()) {
            int slot = 0;
            for (Map.Entry<BufferRef, Long> e : scratchLayout.offsets().entrySet()) {
                BufferRef ref = e.getKey();
                long offset = e.getValue();
                String name = "scratch" + slot++;
                // UnsafePointer has no from_address(); every supported Mojo builds one from an
                // Int address with the unsafe_from_address= constructor.
                String ptr =
                        mojoPointerType(ref.dataType())
                                + "(unsafe_from_address=Int(scratch_ptr + UInt64("
                                + offset
                                + ")))";
                declare(name, ptr);
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
        declare(totalVar, "1");
        for (StructuredFor forLoop : loops) {
            String ub = emitIndexExpr(forLoop.upperBound());
            line(totalVar + " *= " + ub);
        }

        // Emit single-lane mapping (launch config covers total work elements).
        String idxVar = "linear_" + tempId++;
        line("");
        line("# GPU-parallel linear mapping");
        declare(idxVar, "gid");
        line("if " + idxVar + " < " + totalVar + ":");
        enterBlock();

        // De-linearize index into loop indices
        String tempIdx = "temp_" + tempId++;
        declare(tempIdx, idxVar);
        for (int i = loops.size() - 1; i >= 0; i--) {
            StructuredFor forLoop = loops.get(i);
            String extent = emitIndexExpr(forLoop.upperBound());
            String loopIdx = forLoop.indexName();
            declare(loopIdx, tempIdx + " % " + extent);
            line(tempIdx + " = " + tempIdx + " // " + extent);
        }

        // Emit the innermost body (excluding yield)
        for (LIRExprNode stmt : innermostBody.statements()) {
            if (stmt instanceof Yield) {
                continue;
            }
            emitNode(stmt);
        }

        exitBlock();
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
            declare(arg.name(), forceCastExpr(arg.dataType(), emitScalarExpr(arg.init())));
        }
        String idx = loop.indexName();
        String lb = emitIndexExpr(loop.lowerBound());
        String ub = emitIndexExpr(loop.upperBound());
        String step = emitIndexExpr(loop.step());
        line("for " + idx + " in range(" + lb + ", " + ub + ", " + step + "):");
        enterBlock();
        scopes.peek().names().add(idx);
        Block body = loop.body();
        Yield yield = (Yield) body.statements().getLast();
        for (int i = 0; i < body.statements().size() - 1; i++) {
            emitNode(body.statements().get(i));
        }
        for (int i = 0; i < loop.iterArgs().size(); i++) {
            LoopIterArg arg = loop.iterArgs().get(i);
            String tmp = arg.name() + "_next" + (tempId++);
            declare(tmp, forceCastExpr(arg.dataType(), emitScalarExpr(yield.values().get(i))));
            line(arg.name() + " = " + tmp);
        }
        exitBlock();
    }

    private void emitStore(Store store) {
        String bufferName = requireBuffer(store.buffer());
        String idx = byteOffsetToElementIndex(store.buffer(), emitIndexExpr(store.offset()));
        String value = emitScalarExpr(store.value());
        if (store.value().dataType() != store.buffer().dataType()) {
            value = castExpr(store.value().dataType(), store.buffer().dataType(), value);
        }
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
            declare(tmp, expr);
            cachedScalars.put(resolved, tmp);
            scopes.peek().cachedNodes().add(resolved);
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
                yield forceCastExpr(fromIndex.dataType(), indexExpr);
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
        // Literals are wrapped in the operand's type: Mojo 1.x turns a bare float literal mixed
        // with Float32 into Float64 and then rejects the mismatched operands.
        String t = mojoScalarType(unary.input().dataType());
        return switch (unary.op()) {
            case NEGATE -> "-(" + in + ")";
            case ABS -> "(" + in + " if " + in + " >= 0 else -" + in + ")";
            case EXP -> "(" + t + "(2.718281828459045) ** " + in + ")"; // Approximate e^x
            case LOG -> "(log2(" + in + ") * " + t + "(0.6931471805599453))"; // Approximate ln(x)
            case SQRT -> "sqrt(" + in + ")";
            case SIN -> emitTrig("sin", in, unary.input().dataType());
            case COS -> emitTrig("cos", in, unary.input().dataType());
            case TAN -> emitTrig("tan", in, unary.input().dataType());
            case TANH -> emitTrig("tanh", in, unary.input().dataType());
            case RECIPROCAL -> {
                String one = unary.input().dataType().isFloatingPoint() ? t + "(1.0)" : "1.0";
                yield "(" + one + " / " + in + ")";
            }
            case LOGICAL_NOT -> "(not (" + truthy(in, unary.input().dataType()) + "))";
            case BITWISE_NOT -> "~(" + in + ")";
        };
    }

    private String emitTrig(String fn, String in, DataType type) {
        if (type == DataType.FP64) {
            return "Float64(" + fn + "(Float32(" + in + ")))";
        }
        return fn + "(" + in + ")";
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
            case SHIFT_LEFT ->
                    forceCastExpr(
                            binary.dataType(),
                            "("
                                    + l
                                    + " << "
                                    + mojoScalarType(binary.dataType())
                                    + normalizedShift(binary.dataType(), r)
                                    + ")");
            case SHIFT_RIGHT ->
                    forceCastExpr(
                            binary.dataType(),
                            "("
                                    + l
                                    + " >> "
                                    + mojoScalarType(binary.dataType())
                                    + normalizedShift(binary.dataType(), r)
                                    + ")");
            case SHIFT_RIGHT_UNSIGNED ->
                    emitUnsignedRightShift(
                            binary.dataType(), l, normalizedShift(binary.dataType(), r));
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
            case DIVIDE -> "//";
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
        return "Int((" + offsetExpr + ") // " + byteSize + ")";
    }

    private String scalarLiteral(SConst c) {
        DataType t = c.dataType();
        long bits = c.rawBits();
        if (t == DataType.BOOL) {
            return bits == 0 ? "False" : "True";
        }
        if (t == DataType.I8) {
            return "Int8(" + Byte.toString((byte) bits) + ")";
        }
        if (t == DataType.I16) {
            return "Int16(" + Short.toString((short) bits) + ")";
        }
        if (t == DataType.I32) {
            return "Int32(" + Integer.toString((int) bits) + ")";
        }
        if (t == DataType.I64) {
            return "Int64(" + Long.toString(bits) + ")";
        }
        if (t == DataType.FP32) {
            float value = Float.intBitsToFloat((int) bits);
            if (Float.isNaN(value)) {
                return "Float32(0.0)";
            }
            if (Float.isInfinite(value)) {
                return value < 0.0f ? "Float32(-3.4028235e38)" : "Float32(3.4028235e38)";
            }
            String literal = CLikeExprSupport.formatFloatLiteral(value);
            if (literal.endsWith("f")) {
                literal = literal.substring(0, literal.length() - 1);
            }
            return "Float32(" + literal + ")";
        }
        if (t == DataType.FP64) {
            double value = Double.longBitsToDouble(bits);
            if (Double.isNaN(value)) {
                return "Float64(0.0)";
            }
            if (Double.isInfinite(value)) {
                return value < 0.0
                        ? "Float64(-1.7976931348623157e308)"
                        : "Float64(1.7976931348623157e308)";
            }
            return "Float64(" + CLikeExprSupport.formatDoubleLiteral(value) + ")";
        }
        if (t == DataType.FP16) {
            float value = Float.float16ToFloat((short) bits);
            if (Float.isNaN(value)) {
                return "Float16(0.0)";
            }
            if (Float.isInfinite(value)) {
                return value < 0.0f ? "Float16(-65504.0)" : "Float16(65504.0)";
            }
            String literal = CLikeExprSupport.formatFloatLiteral(value);
            if (literal.endsWith("f")) {
                literal = literal.substring(0, literal.length() - 1);
            }
            return "Float16(" + literal + ")";
        }
        if (t == DataType.BF16) {
            float value = BFloat16.toFloat((short) bits);
            if (Float.isNaN(value)) {
                return "BFloat16(0.0)";
            }
            if (Float.isInfinite(value)) {
                return value < 0.0f ? "BFloat16(-3.3895314e38)" : "BFloat16(3.3895314e38)";
            }
            String literal = CLikeExprSupport.formatFloatLiteral(value);
            if (literal.endsWith("f")) {
                literal = literal.substring(0, literal.length() - 1);
            }
            return "BFloat16(" + literal + ")";
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
        if (source == DataType.BOOL && target.isFloatingPoint()) {
            // Mojo 1.x only builds a float from an integral value, not from a Bool.
            return mojoScalarType(target) + "(Int(" + expr + "))";
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
            return "Int64(" + expr + ")";
        }
        if (target == DataType.FP16) {
            return "Float16(" + expr + ")";
        }
        if (target == DataType.BF16) {
            return "BFloat16(" + expr + ")";
        }
        return expr;
    }

    private String forceCastExpr(DataType target, String expr) {
        if (target == DataType.BOOL) {
            return "Bool(" + expr + ")";
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
        if (target == DataType.I64) {
            return "Int64(" + expr + ")";
        }
        if (target == DataType.FP32) {
            return "Float32(" + expr + ")";
        }
        if (target == DataType.FP64) {
            return "Float64(" + expr + ")";
        }
        if (target == DataType.FP16) {
            return "Float16(" + expr + ")";
        }
        if (target == DataType.BF16) {
            return "BFloat16(" + expr + ")";
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
        int mask = 31;
        if (type == DataType.I8) {
            mask = 7;
        } else if (type == DataType.I16) {
            mask = 15;
        } else if (type == DataType.I32) {
            mask = 31;
        } else if (type == DataType.I64) {
            mask = 63;
        }
        return "(Int(" + right + ") & " + mask + ")";
    }

    /** Shift amounts must have the shifted value's type (Mojo 1.x rejects Int on the right). */
    private String emitUnsignedRightShift(DataType type, String left, String shiftExpr) {
        String u32 = "UInt32" + shiftExpr;
        String u64 = "UInt64" + shiftExpr;
        if (type == DataType.I8) {
            return "Int8((UInt32(Int32(" + left + ") & 255) >> " + u32 + "))";
        }
        if (type == DataType.I16) {
            return "Int16((UInt32(Int32(" + left + ") & 65535) >> " + u32 + "))";
        }
        if (type == DataType.I32) {
            return "Int32((UInt32(" + left + ") >> " + u32 + "))";
        }
        if (type == DataType.I64) {
            return "Int64((UInt64(" + left + ") >> " + u64 + "))";
        }
        return forceCastExpr(type, "(UInt64(" + left + ") >> " + u64 + ")");
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
            return "Int64";
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

    /** Opens an indented block: its {@code var} names and cached temporaries end with it. */
    private void enterBlock() {
        indent++;
        scopes.push(new Scope());
    }

    private void exitBlock() {
        Scope scope = scopes.pop();
        for (LIRExprNode node : scope.cachedNodes()) {
            cachedScalars.remove(node);
        }
        indent--;
    }

    /**
     * Emits {@code var name = expr}, or a plain assignment when {@code name} is already visible.
     */
    private void declare(String name, String expr) {
        boolean visible = scopes.stream().anyMatch(s -> s.names().contains(name));
        line((visible ? "" : "var ") + name + " = " + expr);
        if (!visible) {
            scopes.peek().names().add(name);
        }
    }

    private void line(String text) {
        lines.add("    ".repeat(Math.max(0, indent)) + text);
    }
}
