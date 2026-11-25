package com.qxotic.jota.stablehlo.op;

/**
 * Visitor over the typed op records. Every method defaults to {@link #visitDefault(Op)}, so a
 * visitor implements only the ops it supports (e.g. a converter handling a subset of the opset).
 *
 * @param <R> result type of a visit
 */
public interface OpVisitor<R> {

    /** Called for every op without a dedicated override; throws by default. */
    default R visitDefault(Op op) {
        throw new UnsupportedOperationException("unsupported operation: " + op.name());
    }

    default R visitGeneric(GenericOp op) {
        return visitDefault(op);
    }

    default R visitAbs(AbsOp op) {
        return visitDefault(op);
    }

    default R visitCeil(CeilOp op) {
        return visitDefault(op);
    }

    default R visitFloor(FloorOp op) {
        return visitDefault(op);
    }

    default R visitConvert(ConvertOp op) {
        return visitDefault(op);
    }

    default R visitCountLeadingZeros(CountLeadingZerosOp op) {
        return visitDefault(op);
    }

    default R visitImag(ImagOp op) {
        return visitDefault(op);
    }

    default R visitReal(RealOp op) {
        return visitDefault(op);
    }

    default R visitIsFinite(IsFiniteOp op) {
        return visitDefault(op);
    }

    default R visitNot(NotOp op) {
        return visitDefault(op);
    }

    default R visitNegate(NegateOp op) {
        return visitDefault(op);
    }

    default R visitPopcnt(PopcntOp op) {
        return visitDefault(op);
    }

    default R visitRoundNearestAfz(RoundNearestAfzOp op) {
        return visitDefault(op);
    }

    default R visitRoundNearestEven(RoundNearestEvenOp op) {
        return visitDefault(op);
    }

    default R visitSign(SignOp op) {
        return visitDefault(op);
    }

    default R visitBitcastConvert(BitcastConvertOp op) {
        return visitDefault(op);
    }

    default R visitReshape(ReshapeOp op) {
        return visitDefault(op);
    }

    default R visitUniformQuantize(UniformQuantizeOp op) {
        return visitDefault(op);
    }

    default R visitUniformDequantize(UniformDequantizeOp op) {
        return visitDefault(op);
    }

    default R visitCbrt(CbrtOp op) {
        return visitDefault(op);
    }

    default R visitCosine(CosineOp op) {
        return visitDefault(op);
    }

    default R visitExponential(ExponentialOp op) {
        return visitDefault(op);
    }

    default R visitExponentialMinusOne(ExponentialMinusOneOp op) {
        return visitDefault(op);
    }

    default R visitExp2(Exp2Op op) {
        return visitDefault(op);
    }

    default R visitLog(LogOp op) {
        return visitDefault(op);
    }

    default R visitLogPlusOne(LogPlusOneOp op) {
        return visitDefault(op);
    }

    default R visitLog2(Log2Op op) {
        return visitDefault(op);
    }

    default R visitLogistic(LogisticOp op) {
        return visitDefault(op);
    }

    default R visitRsqrt(RsqrtOp op) {
        return visitDefault(op);
    }

    default R visitSine(SineOp op) {
        return visitDefault(op);
    }

    default R visitSqrt(SqrtOp op) {
        return visitDefault(op);
    }

    default R visitTan(TanOp op) {
        return visitDefault(op);
    }

    default R visitTanh(TanhOp op) {
        return visitDefault(op);
    }

    default R visitAdd(AddOp op) {
        return visitDefault(op);
    }

    default R visitAtan2(Atan2Op op) {
        return visitDefault(op);
    }

    default R visitDivide(DivideOp op) {
        return visitDefault(op);
    }

    default R visitMaximum(MaximumOp op) {
        return visitDefault(op);
    }

    default R visitMinimum(MinimumOp op) {
        return visitDefault(op);
    }

    default R visitMultiply(MultiplyOp op) {
        return visitDefault(op);
    }

    default R visitPower(PowerOp op) {
        return visitDefault(op);
    }

    default R visitRemainder(RemainderOp op) {
        return visitDefault(op);
    }

    default R visitShiftLeft(ShiftLeftOp op) {
        return visitDefault(op);
    }

    default R visitShiftRightArithmetic(ShiftRightArithmeticOp op) {
        return visitDefault(op);
    }

    default R visitShiftRightLogical(ShiftRightLogicalOp op) {
        return visitDefault(op);
    }

    default R visitSubtract(SubtractOp op) {
        return visitDefault(op);
    }

    default R visitAnd(AndOp op) {
        return visitDefault(op);
    }

    default R visitOr(OrOp op) {
        return visitDefault(op);
    }

    default R visitXor(XorOp op) {
        return visitDefault(op);
    }

    default R visitComplex(ComplexOp op) {
        return visitDefault(op);
    }

    default R visitConstant(ConstantOp op) {
        return visitDefault(op);
    }

    default R visitIota(IotaOp op) {
        return visitDefault(op);
    }

    default R visitDynamicIota(DynamicIotaOp op) {
        return visitDefault(op);
    }

    default R visitCreateToken(CreateTokenOp op) {
        return visitDefault(op);
    }

    default R visitAfterAll(AfterAllOp op) {
        return visitDefault(op);
    }

    default R visitReplicaId(ReplicaIdOp op) {
        return visitDefault(op);
    }

    default R visitPartitionId(PartitionIdOp op) {
        return visitDefault(op);
    }

    default R visitCompare(CompareOp op) {
        return visitDefault(op);
    }

    default R visitSelect(SelectOp op) {
        return visitDefault(op);
    }

    default R visitClamp(ClampOp op) {
        return visitDefault(op);
    }

    default R visitFma(FmaOp op) {
        return visitDefault(op);
    }

    default R visitBroadcast(BroadcastOp op) {
        return visitDefault(op);
    }

    default R visitBroadcastInDim(BroadcastInDimOp op) {
        return visitDefault(op);
    }

    default R visitDynamicBroadcastInDim(DynamicBroadcastInDimOp op) {
        return visitDefault(op);
    }

    default R visitTranspose(TransposeOp op) {
        return visitDefault(op);
    }

    default R visitReverse(ReverseOp op) {
        return visitDefault(op);
    }

    default R visitSlice(SliceOp op) {
        return visitDefault(op);
    }

    default R visitDynamicSlice(DynamicSliceOp op) {
        return visitDefault(op);
    }

    default R visitDynamicUpdateSlice(DynamicUpdateSliceOp op) {
        return visitDefault(op);
    }

    default R visitRealDynamicSlice(RealDynamicSliceOp op) {
        return visitDefault(op);
    }

    default R visitConcatenate(ConcatenateOp op) {
        return visitDefault(op);
    }

    default R visitPad(PadOp op) {
        return visitDefault(op);
    }

    default R visitDynamicPad(DynamicPadOp op) {
        return visitDefault(op);
    }

    default R visitDynamicReshape(DynamicReshapeOp op) {
        return visitDefault(op);
    }

    default R visitGetDimensionSize(GetDimensionSizeOp op) {
        return visitDefault(op);
    }

    default R visitSetDimensionSize(SetDimensionSizeOp op) {
        return visitDefault(op);
    }

    default R visitGather(GatherOp op) {
        return visitDefault(op);
    }

    default R visitDynamicGather(DynamicGatherOp op) {
        return visitDefault(op);
    }

    default R visitTorchIndexSelect(TorchIndexSelectOp op) {
        return visitDefault(op);
    }

    default R visitScatter(ScatterOp op) {
        return visitDefault(op);
    }

    default R visitSelectAndScatter(SelectAndScatterOp op) {
        return visitDefault(op);
    }

    default R visitReduce(ReduceOp op) {
        return visitDefault(op);
    }

    default R visitReduceWindow(ReduceWindowOp op) {
        return visitDefault(op);
    }

    default R visitReducePrecision(ReducePrecisionOp op) {
        return visitDefault(op);
    }

    default R visitMap(MapOp op) {
        return visitDefault(op);
    }

    default R visitSort(SortOp op) {
        return visitDefault(op);
    }

    default R visitDot(DotOp op) {
        return visitDefault(op);
    }

    default R visitDotGeneral(DotGeneralOp op) {
        return visitDefault(op);
    }

    default R visitEinsum(EinsumOp op) {
        return visitDefault(op);
    }

    default R visitUnaryEinsum(UnaryEinsumOp op) {
        return visitDefault(op);
    }

    default R visitConvolution(ConvolutionOp op) {
        return visitDefault(op);
    }

    default R visitDynamicConv(DynamicConvOp op) {
        return visitDefault(op);
    }

    default R visitFft(FftOp op) {
        return visitDefault(op);
    }

    default R visitRng(RngOp op) {
        return visitDefault(op);
    }

    default R visitRngBitGenerator(RngBitGeneratorOp op) {
        return visitDefault(op);
    }

    default R visitCholesky(CholeskyOp op) {
        return visitDefault(op);
    }

    default R visitTriangularSolve(TriangularSolveOp op) {
        return visitDefault(op);
    }

    default R visitBatchNormInference(BatchNormInferenceOp op) {
        return visitDefault(op);
    }

    default R visitBatchNormTraining(BatchNormTrainingOp op) {
        return visitDefault(op);
    }

    default R visitBatchNormGrad(BatchNormGradOp op) {
        return visitDefault(op);
    }

    default R visitTuple(TupleOp op) {
        return visitDefault(op);
    }

    default R visitGetTupleElement(GetTupleElementOp op) {
        return visitDefault(op);
    }

    default R visitOptimizationBarrier(OptimizationBarrierOp op) {
        return visitDefault(op);
    }

    default R visitIf(IfOp op) {
        return visitDefault(op);
    }

    default R visitCase(CaseOp op) {
        return visitDefault(op);
    }

    default R visitWhile(WhileOp op) {
        return visitDefault(op);
    }

    default R visitCustomCall(CustomCallOp op) {
        return visitDefault(op);
    }

    default R visitComposite(CompositeOp op) {
        return visitDefault(op);
    }

    default R visitInfeed(InfeedOp op) {
        return visitDefault(op);
    }

    default R visitOutfeed(OutfeedOp op) {
        return visitDefault(op);
    }

    default R visitSend(SendOp op) {
        return visitDefault(op);
    }

    default R visitRecv(RecvOp op) {
        return visitDefault(op);
    }

    default R visitAllGather(AllGatherOp op) {
        return visitDefault(op);
    }

    default R visitAllReduce(AllReduceOp op) {
        return visitDefault(op);
    }

    default R visitReduceScatter(ReduceScatterOp op) {
        return visitDefault(op);
    }

    default R visitAllToAll(AllToAllOp op) {
        return visitDefault(op);
    }

    default R visitCollectivePermute(CollectivePermuteOp op) {
        return visitDefault(op);
    }

    default R visitCollectiveBroadcast(CollectiveBroadcastOp op) {
        return visitDefault(op);
    }

    default R visitCollectiveReduce(CollectiveReduceOp op) {
        return visitDefault(op);
    }

    default R visitCrossReplicaSum(CrossReplicaSumOp op) {
        return visitDefault(op);
    }

    default R visitAsyncStart(AsyncStartOp op) {
        return visitDefault(op);
    }

    default R visitAsyncDone(AsyncDoneOp op) {
        return visitDefault(op);
    }

    default R visitReturn(ReturnOp op) {
        return visitDefault(op);
    }

    default R visitFunc(FuncOp op) {
        return visitDefault(op);
    }

    default R visitCall(CallOp op) {
        return visitDefault(op);
    }

    default R visitFuncReturn(FuncReturnOp op) {
        return visitDefault(op);
    }

    default R visitModule(ModuleOp op) {
        return visitDefault(op);
    }
}
