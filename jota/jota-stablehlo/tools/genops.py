"""Generates the typed StableHLO op records, Op.java and OpVisitor.java.
# Usage (from jota/): python3 jota-stablehlo/tools/genops.py jota-stablehlo/src/main/java/com/qxotic/jota/stablehlo/op && mvn -pl jota-stablehlo spotless:apply

Table: mnemonic -> (ClassName, operands, attributes, regions, doc)
  operands: 'x' single, 'xs*' variadic rest (fixed operands may follow), 'a*h'/'b*h' halves,
            'a*s','idx','b*s' scatter segments.
  attributes: 'name:kind' kinds i64 i64? i64s i64s? bool str str? sym f64 dense dense? attr? attr
              type enum:Cls enum:Cls? gather scatter dot conv
  regions: names; 'branches*' = all regions.
"""
import os, re, sys

out_dir = sys.argv[1]
ACC = 'result_accuracy:attr?'
U = lambda: (['operand'], [], [])
UA = lambda: (['operand'], [ACC], [])
B = lambda: (['lhs', 'rhs'], [], [])

T = {}
def op(mn, cls, operands, attrs=(), regions=(), doc=None, dialect='stablehlo'):
    T[f'{dialect}.{mn}'] = (cls, list(operands), list(attrs), list(regions), doc or f'{{@code {dialect}.{mn}}}')

for mn, cls in [('abs','Abs'),('ceil','Ceil'),('floor','Floor'),('convert','Convert'),
                ('count_leading_zeros','CountLeadingZeros'),('imag','Imag'),('real','Real'),
                ('is_finite','IsFinite'),('not','Not'),('negate','Negate'),('popcnt','Popcnt'),
                ('round_nearest_afz','RoundNearestAfz'),('round_nearest_even','RoundNearestEven'),
                ('sign','Sign'),('bitcast_convert','BitcastConvert'),('reshape','Reshape'),
                ('uniform_quantize','UniformQuantize'),('uniform_dequantize','UniformDequantize')]:
    op(mn, cls+'Op', ['operand'])
for mn, cls in [('cbrt','Cbrt'),('cosine','Cosine'),('exponential','Exponential'),
                ('exponential_minus_one','ExponentialMinusOne'),('exp2','Exp2'),('log','Log'),
                ('log_plus_one','LogPlusOne'),('log2','Log2'),('logistic','Logistic'),
                ('rsqrt','Rsqrt'),('sine','Sine'),('sqrt','Sqrt'),('tan','Tan'),('tanh','Tanh')]:
    op(mn, cls+'Op', ['operand'], [ACC])
for mn, cls in [('add','Add'),('atan2','Atan2'),('divide','Divide'),('maximum','Maximum'),
                ('minimum','Minimum'),('multiply','Multiply'),('power','Power'),
                ('remainder','Remainder'),('shift_left','ShiftLeft'),
                ('shift_right_arithmetic','ShiftRightArithmetic'),
                ('shift_right_logical','ShiftRightLogical'),('subtract','Subtract'),
                ('and','And'),('or','Or'),('xor','Xor'),('complex','Complex')]:
    op(mn, cls+'Op', ['lhs','rhs'])

op('constant','ConstantOp',[],['value:attr'])
op('iota','IotaOp',[],['iota_dimension:i64'])
op('dynamic_iota','DynamicIotaOp',['outputShape'],['iota_dimension:i64'])
op('create_token','CreateTokenOp',[])
op('after_all','AfterAllOp',['inputs*'])
op('replica_id','ReplicaIdOp',[])
op('partition_id','PartitionIdOp',[])
op('compare','CompareOp',['lhs','rhs'],['comparison_direction:enum:ComparisonDirection','compare_type:enum:ComparisonType?'])
op('select','SelectOp',['pred','onTrue','onFalse'])
op('clamp','ClampOp',['min','operand','max'])
op('fma','FmaOp',['a','b','c'])
op('broadcast','BroadcastOp',['operand'],['broadcast_sizes:i64s'])
op('broadcast_in_dim','BroadcastInDimOp',['operand'],['broadcast_dimensions:i64s'])
op('dynamic_broadcast_in_dim','DynamicBroadcastInDimOp',['operand','outputDimensions'],
   ['broadcast_dimensions:i64s','known_expanding_dimensions:i64s?','known_nonexpanding_dimensions:i64s?'])
op('transpose','TransposeOp',['operand'],['permutation:i64s'])
op('reverse','ReverseOp',['operand'],['dimensions:i64s'])
op('slice','SliceOp',['operand'],['start_indices:i64s','limit_indices:i64s','strides:i64s'])
op('dynamic_slice','DynamicSliceOp',['operand','startIndices*'],['slice_sizes:i64s'])
op('dynamic_update_slice','DynamicUpdateSliceOp',['operand','update','startIndices*'])
op('real_dynamic_slice','RealDynamicSliceOp',['operand','startIndices','limitIndices','strides'])
op('concatenate','ConcatenateOp',['inputs*'],['dimension:i64'])
op('pad','PadOp',['operand','paddingValue'],['edge_padding_low:i64s','edge_padding_high:i64s','interior_padding:i64s'])
op('dynamic_pad','DynamicPadOp',['operand','paddingValue','edgePaddingLow','edgePaddingHigh','interiorPadding'])
op('dynamic_reshape','DynamicReshapeOp',['operand','outputShape'])
op('get_dimension_size','GetDimensionSizeOp',['operand'],['dimension:i64'])
op('set_dimension_size','SetDimensionSizeOp',['operand','size'],['dimension:i64'])
op('gather','GatherOp',['operand','startIndices'],['dimension_numbers:gather','slice_sizes:i64s','indices_are_sorted:bool'])
op('dynamic_gather','DynamicGatherOp',['operand','startIndices','sliceSizes'],['dimension_numbers:gather','indices_are_sorted:bool'])
op('torch_index_select','TorchIndexSelectOp',['operand','index'],['dim:i64','batch_dims:i64'])
op('scatter','ScatterOp',['inputs*s','scatterIndices','updates*s'],
   ['scatter_dimension_numbers:scatter','indices_are_sorted:bool','unique_indices:bool'],['updateComputation'])
op('select_and_scatter','SelectAndScatterOp',['operand','source','initValue'],
   ['window_dimensions:i64s?','window_strides:i64s?','padding:dense?'],['select','scatter'])
op('reduce','ReduceOp',['inputs*h','initValues*h'],['dimensions:i64s'],['body'])
op('reduce_window','ReduceWindowOp',['inputs*h','initValues*h'],
   ['window_dimensions:i64s','window_strides:i64s?','base_dilations:i64s?','window_dilations:i64s?','padding:dense?'],['body'])
op('reduce_precision','ReducePrecisionOp',['operand'],['exponent_bits:i64','mantissa_bits:i64'])
op('map','MapOp',['inputs*'],['dimensions:i64s'],['computation'])
op('sort','SortOp',['inputs*'],['dimension:i64?','is_stable:bool'],['comparator'])
op('dot','DotOp',['lhs','rhs'],['precision_config:attr?'])
op('dot_general','DotGeneralOp',['lhs','rhs'],['dot_dimension_numbers:dot','precision_config:attr?','algorithm:attr?'])
op('einsum','EinsumOp',['lhs','rhs'],['einsum_config:str'])
op('unary_einsum','UnaryEinsumOp',['operand'],['einsum_config:str'])
CONV = ['window_strides:i64s?','padding:dense?','lhs_dilation:i64s?','rhs_dilation:i64s?','window_reversal:attr?',
        'dimension_numbers:conv','feature_group_count:i64','batch_group_count:i64','precision_config:attr?']
op('convolution','ConvolutionOp',['lhs','rhs'],CONV)
op('dynamic_conv','DynamicConvOp',['lhs','rhs','padding'],[c for c in CONV if not c.startswith('padding')])
op('fft','FftOp',['operand'],['fft_type:enum:FftType','fft_length:i64s'])
op('rng','RngOp',['a','b','shape'],['rng_distribution:enum:RngDistribution'])
op('rng_bit_generator','RngBitGeneratorOp',['initialState'],['rng_algorithm:enum:RngAlgorithm'])
op('cholesky','CholeskyOp',['a'],['lower:bool'])
op('triangular_solve','TriangularSolveOp',['a','b'],['left_side:bool','lower:bool','unit_diagonal:bool','transpose_a:enum:Transpose'])
op('batch_norm_inference','BatchNormInferenceOp',['operand','scale','offset','mean','variance'],['epsilon:f64','feature_index:i64'])
op('batch_norm_training','BatchNormTrainingOp',['operand','scale','offset'],['epsilon:f64','feature_index:i64'])
op('batch_norm_grad','BatchNormGradOp',['operand','scale','mean','variance','gradOutput'],['epsilon:f64','feature_index:i64'])
op('tuple','TupleOp',['val*'])
op('get_tuple_element','GetTupleElementOp',['operand'],['index:i64'])
op('optimization_barrier','OptimizationBarrierOp',['operand*'])
op('if','IfOp',['pred'],[],['trueBranch','falseBranch'])
op('case','CaseOp',['index'],[],['branches*'])
op('while','WhileOp',['operand*'],[],['cond','body'])
op('custom_call','CustomCallOp',['inputs*'],['call_target_name:str','has_side_effect:bool','backend_config:attr?',
   'api_version:attr?','called_computations:attr?','operand_layouts:attr?','result_layouts:attr?','output_operand_aliases:attr?'])
op('composite','CompositeOp',['inputs*'],['name:str','composite_attributes:attr?','decomposition:sym','version:i64?'])
op('infeed','InfeedOp',['token'],['infeed_config:str?','layout:attr?'])
op('outfeed','OutfeedOp',['inputs*','token'],['outfeed_config:str?'])
op('send','SendOp',['inputs*','token'],['channel_handle:attr','is_host_transfer:bool','source_target_pairs:dense?'])
op('recv','RecvOp',['token'],['channel_handle:attr','is_host_transfer:bool','source_target_pairs:dense?'])
COLL = ['replica_groups:dense?','channel_handle:attr?','use_global_device_ids:bool']
op('all_gather','AllGatherOp',['operands*'],['all_gather_dim:i64']+COLL)
op('all_reduce','AllReduceOp',['operands*'],COLL,['computation'])
op('reduce_scatter','ReduceScatterOp',['operand'],['scatter_dimension:i64']+COLL,['computation'])
op('all_to_all','AllToAllOp',['operands*'],['split_dimension:i64','concat_dimension:i64','split_count:i64','replica_groups:dense?','channel_handle:attr?'])
op('collective_permute','CollectivePermuteOp',['operand'],['source_target_pairs:dense','channel_handle:attr?'])
op('collective_broadcast','CollectiveBroadcastOp',['operand'],['replica_groups:dense?','channel_handle:attr?'])
op('collective_reduce','CollectiveReduceOp',['operand'],['replica_groups:dense?','channel_handle:attr?'],['computation'])
op('cross-replica-sum','CrossReplicaSumOp',['operand'],['replica_groups:dense'],doc='{@code stablehlo.cross-replica-sum} (legacy all_reduce)')
op('async_start','AsyncStartOp',['inputs*'],['called_computation:attr?'])
op('async_done','AsyncDoneOp',['future'])
op('return','ReturnOp',['results*'],doc='{@code stablehlo.return}: terminator of StableHLO regions')
op('func','FuncOp',[],['sym_name:str','function_type:type','sym_visibility:str?','arg_attrs:attr?','res_attrs:attr?'],
   ['body'],doc='{@code func.func}: a named function',dialect='func')
op('call','CallOp',['inputs*'],['callee:sym'],doc='{@code func.call}',dialect='func')
op('return','FuncReturnOp',['results*'],doc='{@code func.return}: function terminator',dialect='func')
op('module','ModuleOp',[],['sym_name:str?'],['bodyRegion'],doc='{@code builtin.module}',dialect='builtin')

def camel(s):
    p = s.split('_'); return p[0] + ''.join(x[:1].upper() + x[1:] for x in p[1:])

HEADER = 'package com.qxotic.jota.stablehlo.op;\n\n'
IMPORTS = {
 'Value':'com.qxotic.jota.stablehlo.ir.Value','List':'java.util.List','Operation':'com.qxotic.jota.stablehlo.ir.Operation',
 'Optional':'java.util.Optional','OptionalLong':'java.util.OptionalLong','Attribute':'com.qxotic.jota.stablehlo.attr.Attribute',
 'DenseElementsAttr':'com.qxotic.jota.stablehlo.attr.DenseElementsAttr','Region':'com.qxotic.jota.stablehlo.ir.Region',
 'Type':'com.qxotic.jota.stablehlo.type.Type','GatherDimensionNumbers':'com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers',
 'ScatterDimensionNumbers':'com.qxotic.jota.stablehlo.attr.ScatterDimensionNumbers','DotDimensionNumbers':'com.qxotic.jota.stablehlo.attr.DotDimensionNumbers',
 'ConvDimensionNumbers':'com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers'}
for e in ['ComparisonDirection','ComparisonType','FftType','RngDistribution','RngAlgorithm','Transpose']:
    IMPORTS[e] = 'com.qxotic.jota.stablehlo.attr.' + e

def gen_record(full, spec):
    cls, operands, attrs, regions, doc = spec
    used = {'Operation'}
    methods = []
    # operands
    fixed_before = []; var = None; fixed_after = []
    kinds = [o for o in operands]
    if any(o.endswith('*h') for o in operands):
        for i, o in enumerate(operands):
            used |= {'List', 'Value'}
            methods.append(f'    public List<Value> {o[:-2]}() {{\n        return OpSupport.half(operation, {i});\n    }}')
    elif any(o.endswith('*s') for o in operands):
        for i, o in enumerate(operands):
            name = o[:-2] if o.endswith('*s') else o
            used |= {'List', 'Value'}
            if o.endswith('*s'):
                methods.append(f'    public List<Value> {name}() {{\n        return OpSupport.scatterGroup(operation, {i});\n    }}')
            else:
                methods.append(f'    public Value {name}() {{\n        return OpSupport.scatterGroup(operation, {i}).getFirst();\n    }}')
    else:
        vi = next((i for i, o in enumerate(operands) if o.endswith('*')), None)
        for i, o in enumerate(operands):
            used.add('Value')
            if vi is None or i < vi:
                methods.append(f'    public Value {o}() {{\n        return operation.operand({i});\n    }}')
            elif i == vi:
                used.add('List')
                methods.append(f'    public List<Value> {o[:-1]}() {{\n        return OpSupport.slice(operation, {vi}, {len(operands)-vi-1});\n    }}')
            else:
                k = len(operands) - i
                methods.append(f'    public Value {o}() {{\n        return operation.operand(operation.numOperands() - {k});\n    }}')
    # attributes
    for a in attrs:
        name, kind = a.split(':', 1)
        m = camel(name)
        if m in ('name', 'operation', 'operands', 'results', 'result', 'regions', 'index') and full not in ('stablehlo.get_tuple_element',):
            m = m + 'Attr' if m != 'name' else 'compositeName'
        if full == 'stablehlo.get_tuple_element' and name == 'index':
            m = 'index'
        opt = kind.endswith('?'); k = kind.rstrip('?')
        q = f'"{name}"'
        if k == 'i64':
            if opt: used.add('OptionalLong'); r, body = 'OptionalLong', f'OpSupport.i64Opt(operation, {q})'
            else: r, body = 'long', f'OpSupport.i64(operation, {q})'
        elif k == 'i64s':
            if opt: used.add('Optional'); r, body = 'Optional<long[]>', f'OpSupport.i64sOpt(operation, {q})'
            else: r, body = 'long[]', f'OpSupport.i64s(operation, {q})'
        elif k == 'bool': r, body = 'boolean', f'OpSupport.bool(operation, {q})'
        elif k in ('str', 'sym'):
            if opt: used.add('Optional'); r, body = 'Optional<String>', f'OpSupport.strOpt(operation, {q})'
            else: r, body = 'String', f'OpSupport.str(operation, {q})'
        elif k == 'f64': r, body = 'double', f'OpSupport.f64(operation, {q})'
        elif k == 'dense':
            used.add('DenseElementsAttr')
            if opt: used.add('Optional'); r, body = 'Optional<DenseElementsAttr>', f'OpSupport.denseOpt(operation, {q})'
            else: r, body = 'DenseElementsAttr', f'OpSupport.dense(operation, {q})'
        elif k == 'attr':
            used.add('Attribute')
            if opt: used.add('Optional'); r, body = 'Optional<Attribute>', f'operation.attribute({q})'
            else: r, body = 'Attribute', f'OpSupport.required(operation, {q})'
        elif k == 'type': used.add('Type'); r, body = 'Type', f'OpSupport.type(operation, {q})'
        elif k.startswith('enum:'):
            e = k[5:]; used.add(e)
            if opt: used.add('Optional'); r, body = f'Optional<{e}>', f'operation.attribute({q}).map({e}::from)'
            else: r, body = e, f'{e}.from(OpSupport.required(operation, {q}))'
        else:
            t = {'gather':'GatherDimensionNumbers','scatter':'ScatterDimensionNumbers','dot':'DotDimensionNumbers','conv':'ConvDimensionNumbers'}[k]
            used.add(t); r, body = t, f'{t}.from(OpSupport.required(operation, {q}))'
        methods.append(f'    /** Attribute {{@code {name}}}. */\n    public {r} {m}() {{\n        return {body};\n    }}')
    # regions
    for i, rn in enumerate(regions):
        if rn.endswith('*'):
            used |= {'List', 'Region'}
            methods.append(f'    public List<Region> {rn[:-1]}() {{\n        return operation.regions();\n    }}')
        else:
            used.add('Region')
            methods.append(f'    public Region {rn}() {{\n        return operation.region({i});\n    }}')
    imports = sorted(IMPORTS[u] for u in used if u in IMPORTS)
    src = HEADER + ''.join(f'import {i};\n' for i in imports) + '\n'
    src += f'/** Typed view of {doc}. Generated from the op table; see {{@link Op}}. */\n'
    src += f'public record {cls}(Operation operation) implements Op {{\n\n'
    src += f'    public static final String NAME = "{full}";\n\n'
    src += f'    public {cls} {{\n        OpSupport.checkName(operation, NAME);\n    }}\n'
    for m in methods:
        src += '\n' + m + '\n'
    src += '\n    @Override\n    public <R> R accept(OpVisitor<R> visitor) {\n'
    src += f'        return visitor.visit{cls[:-2]}(this);\n    }}\n}}\n'
    return cls, src

classes = []
for full, spec in T.items():
    cls, src = gen_record(full, spec)
    classes.append((full, cls))
    open(os.path.join(out_dir, cls + '.java'), 'w').write(src)

permits = ',\n                '.join([c for _, c in classes] + ['GenericOp'])
cases = '\n'.join(f'            case {c}.NAME -> new {c}(operation);' for _, c in classes)
open(os.path.join(out_dir, 'Op.java'), 'w').write(HEADER + f'''import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.List;

/**
 * Typed view of an {{@link Operation}}: one record per StableHLO op (plus {{@code builtin.module}},
 * {{@code func.func}}, {{@code func.call}}, {{@code func.return}}), {{@link GenericOp}} for anything
 * else ({{@code chlo.*}}, test dialects...).
 *
 * <p>Records only wrap the generic operation and decode operands / attributes / regions on
 * access, so wrapping is free and the generic form stays the single source of truth. Dispatch to
 * per-op logic with {{@link #accept(OpVisitor)}} or with a pattern-matching {{@code switch}} over
 * this sealed interface.
 */
public sealed interface Op
        permits {permits} {{

    /** The wrapped generic operation. */
    Operation operation();

    /** Double dispatch to {{@code visitor.visitXxx(this)}}. */
    <R> R accept(OpVisitor<R> visitor);

    default String name() {{
        return operation().name();
    }}

    default List<Value> operands() {{
        return operation().operands();
    }}

    default List<Value> results() {{
        return operation().results();
    }}

    /** The single result. */
    default Value result() {{
        if (operation().numResults() != 1) {{
            throw new IllegalStateException(
                    name() + " has " + operation().numResults() + " results");
        }}
        return operation().result(0);
    }}

    default List<Type> resultTypes() {{
        return operation().resultTypes();
    }}

    default List<Region> regions() {{
        return operation().regions();
    }}

    /** Wraps a generic operation into its typed record. */
    static Op wrap(Operation operation) {{
        return switch (operation.name()) {{
{cases}
            default -> new GenericOp(operation);
        }};
    }}
}}
''')
visits = '\n\n'.join(f'    default R visit{c[:-2]}({c} op) {{\n        return visitDefault(op);\n    }}' for _, c in classes)
open(os.path.join(out_dir, 'OpVisitor.java'), 'w').write(HEADER + f'''/**
 * Visitor over the typed op records. Every method defaults to {{@link #visitDefault(Op)}}, so a
 * visitor implements only the ops it supports (e.g. a converter handling a subset of the opset).
 *
 * @param <R> result type of a visit
 */
public interface OpVisitor<R> {{

    /** Called for every op without a dedicated override; throws by default. */
    default R visitDefault(Op op) {{
        throw new UnsupportedOperationException("unsupported operation: " + op.name());
    }}

    default R visitGeneric(GenericOp op) {{
        return visitDefault(op);
    }}

{visits}
}}
''')
print(len(classes), 'op records')
