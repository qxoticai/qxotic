package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.List;

/**
 * Typed view of an {@link Operation}: one record per StableHLO op (plus {@code builtin.module},
 * {@code func.func}, {@code func.call}, {@code func.return}), {@link GenericOp} for anything else
 * ({@code chlo.*}, test dialects...).
 *
 * <p>Records only wrap the generic operation and decode operands / attributes / regions on access,
 * so wrapping is free and the generic form stays the single source of truth. Dispatch to per-op
 * logic with {@link #accept(OpVisitor)} or with a pattern-matching {@code switch} over this sealed
 * interface.
 */
public sealed interface Op
        permits AbsOp,
                CeilOp,
                FloorOp,
                ConvertOp,
                CountLeadingZerosOp,
                ImagOp,
                RealOp,
                IsFiniteOp,
                NotOp,
                NegateOp,
                PopcntOp,
                RoundNearestAfzOp,
                RoundNearestEvenOp,
                SignOp,
                BitcastConvertOp,
                ReshapeOp,
                UniformQuantizeOp,
                UniformDequantizeOp,
                CbrtOp,
                CosineOp,
                ExponentialOp,
                ExponentialMinusOneOp,
                Exp2Op,
                LogOp,
                LogPlusOneOp,
                Log2Op,
                LogisticOp,
                RsqrtOp,
                SineOp,
                SqrtOp,
                TanOp,
                TanhOp,
                AddOp,
                Atan2Op,
                DivideOp,
                MaximumOp,
                MinimumOp,
                MultiplyOp,
                PowerOp,
                RemainderOp,
                ShiftLeftOp,
                ShiftRightArithmeticOp,
                ShiftRightLogicalOp,
                SubtractOp,
                AndOp,
                OrOp,
                XorOp,
                ComplexOp,
                ConstantOp,
                IotaOp,
                DynamicIotaOp,
                CreateTokenOp,
                AfterAllOp,
                ReplicaIdOp,
                PartitionIdOp,
                CompareOp,
                SelectOp,
                ClampOp,
                FmaOp,
                BroadcastOp,
                BroadcastInDimOp,
                DynamicBroadcastInDimOp,
                TransposeOp,
                ReverseOp,
                SliceOp,
                DynamicSliceOp,
                DynamicUpdateSliceOp,
                RealDynamicSliceOp,
                ConcatenateOp,
                PadOp,
                DynamicPadOp,
                DynamicReshapeOp,
                GetDimensionSizeOp,
                SetDimensionSizeOp,
                GatherOp,
                DynamicGatherOp,
                TorchIndexSelectOp,
                ScatterOp,
                SelectAndScatterOp,
                ReduceOp,
                ReduceWindowOp,
                ReducePrecisionOp,
                MapOp,
                SortOp,
                DotOp,
                DotGeneralOp,
                EinsumOp,
                UnaryEinsumOp,
                ConvolutionOp,
                DynamicConvOp,
                FftOp,
                RngOp,
                RngBitGeneratorOp,
                CholeskyOp,
                TriangularSolveOp,
                BatchNormInferenceOp,
                BatchNormTrainingOp,
                BatchNormGradOp,
                TupleOp,
                GetTupleElementOp,
                OptimizationBarrierOp,
                IfOp,
                CaseOp,
                WhileOp,
                CustomCallOp,
                CompositeOp,
                InfeedOp,
                OutfeedOp,
                SendOp,
                RecvOp,
                AllGatherOp,
                AllReduceOp,
                ReduceScatterOp,
                AllToAllOp,
                CollectivePermuteOp,
                CollectiveBroadcastOp,
                CollectiveReduceOp,
                CrossReplicaSumOp,
                AsyncStartOp,
                AsyncDoneOp,
                ReturnOp,
                FuncOp,
                CallOp,
                FuncReturnOp,
                ModuleOp,
                GenericOp {

    /** The wrapped generic operation. */
    Operation operation();

    /** Double dispatch to {@code visitor.visitXxx(this)}. */
    <R> R accept(OpVisitor<R> visitor);

    default String name() {
        return operation().name();
    }

    default List<Value> operands() {
        return operation().operands();
    }

    default List<Value> results() {
        return operation().results();
    }

    /** The single result. */
    default Value result() {
        if (operation().numResults() != 1) {
            throw new IllegalStateException(
                    name() + " has " + operation().numResults() + " results");
        }
        return operation().result(0);
    }

    default List<Type> resultTypes() {
        return operation().resultTypes();
    }

    default List<Region> regions() {
        return operation().regions();
    }

    /** Wraps a generic operation into its typed record. */
    static Op wrap(Operation operation) {
        return switch (operation.name()) {
            case AbsOp.NAME -> new AbsOp(operation);
            case CeilOp.NAME -> new CeilOp(operation);
            case FloorOp.NAME -> new FloorOp(operation);
            case ConvertOp.NAME -> new ConvertOp(operation);
            case CountLeadingZerosOp.NAME -> new CountLeadingZerosOp(operation);
            case ImagOp.NAME -> new ImagOp(operation);
            case RealOp.NAME -> new RealOp(operation);
            case IsFiniteOp.NAME -> new IsFiniteOp(operation);
            case NotOp.NAME -> new NotOp(operation);
            case NegateOp.NAME -> new NegateOp(operation);
            case PopcntOp.NAME -> new PopcntOp(operation);
            case RoundNearestAfzOp.NAME -> new RoundNearestAfzOp(operation);
            case RoundNearestEvenOp.NAME -> new RoundNearestEvenOp(operation);
            case SignOp.NAME -> new SignOp(operation);
            case BitcastConvertOp.NAME -> new BitcastConvertOp(operation);
            case ReshapeOp.NAME -> new ReshapeOp(operation);
            case UniformQuantizeOp.NAME -> new UniformQuantizeOp(operation);
            case UniformDequantizeOp.NAME -> new UniformDequantizeOp(operation);
            case CbrtOp.NAME -> new CbrtOp(operation);
            case CosineOp.NAME -> new CosineOp(operation);
            case ExponentialOp.NAME -> new ExponentialOp(operation);
            case ExponentialMinusOneOp.NAME -> new ExponentialMinusOneOp(operation);
            case Exp2Op.NAME -> new Exp2Op(operation);
            case LogOp.NAME -> new LogOp(operation);
            case LogPlusOneOp.NAME -> new LogPlusOneOp(operation);
            case Log2Op.NAME -> new Log2Op(operation);
            case LogisticOp.NAME -> new LogisticOp(operation);
            case RsqrtOp.NAME -> new RsqrtOp(operation);
            case SineOp.NAME -> new SineOp(operation);
            case SqrtOp.NAME -> new SqrtOp(operation);
            case TanOp.NAME -> new TanOp(operation);
            case TanhOp.NAME -> new TanhOp(operation);
            case AddOp.NAME -> new AddOp(operation);
            case Atan2Op.NAME -> new Atan2Op(operation);
            case DivideOp.NAME -> new DivideOp(operation);
            case MaximumOp.NAME -> new MaximumOp(operation);
            case MinimumOp.NAME -> new MinimumOp(operation);
            case MultiplyOp.NAME -> new MultiplyOp(operation);
            case PowerOp.NAME -> new PowerOp(operation);
            case RemainderOp.NAME -> new RemainderOp(operation);
            case ShiftLeftOp.NAME -> new ShiftLeftOp(operation);
            case ShiftRightArithmeticOp.NAME -> new ShiftRightArithmeticOp(operation);
            case ShiftRightLogicalOp.NAME -> new ShiftRightLogicalOp(operation);
            case SubtractOp.NAME -> new SubtractOp(operation);
            case AndOp.NAME -> new AndOp(operation);
            case OrOp.NAME -> new OrOp(operation);
            case XorOp.NAME -> new XorOp(operation);
            case ComplexOp.NAME -> new ComplexOp(operation);
            case ConstantOp.NAME -> new ConstantOp(operation);
            case IotaOp.NAME -> new IotaOp(operation);
            case DynamicIotaOp.NAME -> new DynamicIotaOp(operation);
            case CreateTokenOp.NAME -> new CreateTokenOp(operation);
            case AfterAllOp.NAME -> new AfterAllOp(operation);
            case ReplicaIdOp.NAME -> new ReplicaIdOp(operation);
            case PartitionIdOp.NAME -> new PartitionIdOp(operation);
            case CompareOp.NAME -> new CompareOp(operation);
            case SelectOp.NAME -> new SelectOp(operation);
            case ClampOp.NAME -> new ClampOp(operation);
            case FmaOp.NAME -> new FmaOp(operation);
            case BroadcastOp.NAME -> new BroadcastOp(operation);
            case BroadcastInDimOp.NAME -> new BroadcastInDimOp(operation);
            case DynamicBroadcastInDimOp.NAME -> new DynamicBroadcastInDimOp(operation);
            case TransposeOp.NAME -> new TransposeOp(operation);
            case ReverseOp.NAME -> new ReverseOp(operation);
            case SliceOp.NAME -> new SliceOp(operation);
            case DynamicSliceOp.NAME -> new DynamicSliceOp(operation);
            case DynamicUpdateSliceOp.NAME -> new DynamicUpdateSliceOp(operation);
            case RealDynamicSliceOp.NAME -> new RealDynamicSliceOp(operation);
            case ConcatenateOp.NAME -> new ConcatenateOp(operation);
            case PadOp.NAME -> new PadOp(operation);
            case DynamicPadOp.NAME -> new DynamicPadOp(operation);
            case DynamicReshapeOp.NAME -> new DynamicReshapeOp(operation);
            case GetDimensionSizeOp.NAME -> new GetDimensionSizeOp(operation);
            case SetDimensionSizeOp.NAME -> new SetDimensionSizeOp(operation);
            case GatherOp.NAME -> new GatherOp(operation);
            case DynamicGatherOp.NAME -> new DynamicGatherOp(operation);
            case TorchIndexSelectOp.NAME -> new TorchIndexSelectOp(operation);
            case ScatterOp.NAME -> new ScatterOp(operation);
            case SelectAndScatterOp.NAME -> new SelectAndScatterOp(operation);
            case ReduceOp.NAME -> new ReduceOp(operation);
            case ReduceWindowOp.NAME -> new ReduceWindowOp(operation);
            case ReducePrecisionOp.NAME -> new ReducePrecisionOp(operation);
            case MapOp.NAME -> new MapOp(operation);
            case SortOp.NAME -> new SortOp(operation);
            case DotOp.NAME -> new DotOp(operation);
            case DotGeneralOp.NAME -> new DotGeneralOp(operation);
            case EinsumOp.NAME -> new EinsumOp(operation);
            case UnaryEinsumOp.NAME -> new UnaryEinsumOp(operation);
            case ConvolutionOp.NAME -> new ConvolutionOp(operation);
            case DynamicConvOp.NAME -> new DynamicConvOp(operation);
            case FftOp.NAME -> new FftOp(operation);
            case RngOp.NAME -> new RngOp(operation);
            case RngBitGeneratorOp.NAME -> new RngBitGeneratorOp(operation);
            case CholeskyOp.NAME -> new CholeskyOp(operation);
            case TriangularSolveOp.NAME -> new TriangularSolveOp(operation);
            case BatchNormInferenceOp.NAME -> new BatchNormInferenceOp(operation);
            case BatchNormTrainingOp.NAME -> new BatchNormTrainingOp(operation);
            case BatchNormGradOp.NAME -> new BatchNormGradOp(operation);
            case TupleOp.NAME -> new TupleOp(operation);
            case GetTupleElementOp.NAME -> new GetTupleElementOp(operation);
            case OptimizationBarrierOp.NAME -> new OptimizationBarrierOp(operation);
            case IfOp.NAME -> new IfOp(operation);
            case CaseOp.NAME -> new CaseOp(operation);
            case WhileOp.NAME -> new WhileOp(operation);
            case CustomCallOp.NAME -> new CustomCallOp(operation);
            case CompositeOp.NAME -> new CompositeOp(operation);
            case InfeedOp.NAME -> new InfeedOp(operation);
            case OutfeedOp.NAME -> new OutfeedOp(operation);
            case SendOp.NAME -> new SendOp(operation);
            case RecvOp.NAME -> new RecvOp(operation);
            case AllGatherOp.NAME -> new AllGatherOp(operation);
            case AllReduceOp.NAME -> new AllReduceOp(operation);
            case ReduceScatterOp.NAME -> new ReduceScatterOp(operation);
            case AllToAllOp.NAME -> new AllToAllOp(operation);
            case CollectivePermuteOp.NAME -> new CollectivePermuteOp(operation);
            case CollectiveBroadcastOp.NAME -> new CollectiveBroadcastOp(operation);
            case CollectiveReduceOp.NAME -> new CollectiveReduceOp(operation);
            case CrossReplicaSumOp.NAME -> new CrossReplicaSumOp(operation);
            case AsyncStartOp.NAME -> new AsyncStartOp(operation);
            case AsyncDoneOp.NAME -> new AsyncDoneOp(operation);
            case ReturnOp.NAME -> new ReturnOp(operation);
            case FuncOp.NAME -> new FuncOp(operation);
            case CallOp.NAME -> new CallOp(operation);
            case FuncReturnOp.NAME -> new FuncReturnOp(operation);
            case ModuleOp.NAME -> new ModuleOp(operation);
            default -> new GenericOp(operation);
        };
    }
}
