package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Writer for the MLIR bytecode format, bytecode versions 0 to 6 (the inverse of {@link
 * MlirBytecodeReader}).
 *
 * <p>{@link #write} writes VHLO programs (portable artifacts): attributes and types are encoded by
 * the dialect of the operation that uses them, {@code vhlo.*} operations with the VHLO encodings
 * ({@link VhloEncoder}), other operations ({@code builtin.module}) with the builtin encodings
 * ({@link BuiltinEncoder}).
 *
 * <p>{@link #writePlain} writes StableHLO programs as plain (non-portable) bytecode, as {@code
 * stablehlo-opt --emit-bytecode} does: each attribute and type with the encoding of its own dialect
 * ({@link StablehloEncoder}, builtin, or the textual form under its dialect), the {@code stablehlo}
 * dialect version recorded, and native properties (bytecode version 5 and later) in the layout of
 * the target StableHLO version. Plain bytecode is only readable by a StableHLO whose ops have the
 * same layouts: bytecode version 4 or older stores the attributes in a dictionary instead, which
 * every reader converts to its own properties.
 *
 * <p>An attribute dictionary is always a builtin dictionary with builtin string names. Locations
 * are written as {@code UnknownLoc}, use-list orders are not written (the reader then keeps its
 * default order).
 */
public final class MlirBytecodeWriter {

    static final String BUILTIN = "builtin";
    static final String VHLO = "vhlo";
    static final String STABLEHLO = "stablehlo";

    private final long version;
    private final String producer;

    /** The target of plain bytecode, null for VHLO programs. */
    private final StableHloVersion stablehlo;

    // strings
    private final Map<String, Integer> stringIndex = new HashMap<>();
    private final List<String> strings = new ArrayList<>();

    // dialects and op names
    private final Map<String, Integer> dialectIndex = new LinkedHashMap<>();
    private final Map<String, Integer> opNameIndex = new HashMap<>();
    private final List<String> opNamesInOrder = new ArrayList<>();

    // attribute / type tables
    private record Entry(String dialect, byte[] data, boolean custom) {}

    private record Key(String dialect, Object value) {}

    private final List<Entry> attributes = new ArrayList<>();
    private final List<Entry> types = new ArrayList<>();
    private final Map<Key, Integer> attributeIndex = new HashMap<>();
    private final Map<Key, Integer> typeIndex = new HashMap<>();

    // properties
    private final List<byte[]> properties = new ArrayList<>();

    private MlirBytecodeWriter(long version, String producer, StableHloVersion stablehlo) {
        if (version < 0 || version > MlirBytecodeReader.VERSION_CURRENT) {
            throw new IllegalArgumentException("unsupported bytecode version " + version);
        }
        this.version = version;
        this.producer = Objects.requireNonNull(producer, "producer");
        this.stablehlo = stablehlo;
    }

    /** Encodes a VHLO {@code program} as MLIR bytecode of the given format version. */
    public static byte[] write(Program program, long bytecodeVersion, String producer) {
        return new MlirBytecodeWriter(bytecodeVersion, producer, null).writeAll(program);
    }

    /**
     * Encodes a StableHLO {@code program} as plain MLIR bytecode of the given format version, for a
     * reader of StableHLO {@code target} (whose op property layouts are used from bytecode version
     * 5 on).
     */
    public static byte[] writePlain(
            Program program, long bytecodeVersion, StableHloVersion target, String producer) {
        return new MlirBytecodeWriter(
                        bytecodeVersion, producer, Objects.requireNonNull(target, "target"))
                .writeAll(program);
    }

    long version() {
        return version;
    }

    // ================================================================== tables

    int string(String s) {
        Integer i = stringIndex.get(s);
        if (i == null) {
            i = strings.size();
            strings.add(s);
            stringIndex.put(s, i);
        }
        return i;
    }

    int dialect(String name) {
        return dialectIndex.computeIfAbsent(name, n -> dialectIndex.size());
    }

    /** Index of {@code attribute} encoded with {@code dialect}'s encodings. */
    int attribute(Attribute attribute, String dialect) {
        Key key = new Key(dialect, attribute);
        Integer i = attributeIndex.get(key);
        if (i != null) {
            return i;
        }
        AttrTypeWriter w = new AttrTypeWriter(this);
        String owner =
                switch (dialect) {
                    case VHLO -> VhloEncoder.attribute(w, attribute);
                    default ->
                            stablehlo != null
                                    ? StablehloEncoder.attribute(w, attribute)
                                    : BuiltinEncoder.attribute(w, attribute);
                };
        // registered after nested entries: indices are only references, order is free
        i = attributes.size();
        attributes.add(new Entry(owner, w.bytes(), w.custom()));
        attributeIndex.put(key, i);
        dialect(owner);
        return i;
    }

    int type(Type type, String dialect) {
        Key key = new Key(dialect, type);
        Integer i = typeIndex.get(key);
        if (i != null) {
            return i;
        }
        AttrTypeWriter w = new AttrTypeWriter(this);
        String owner =
                switch (dialect) {
                    case VHLO -> VhloEncoder.type(w, type);
                    default ->
                            stablehlo != null
                                    ? StablehloEncoder.type(w, type)
                                    : BuiltinEncoder.type(w, type);
                };
        i = types.size();
        types.add(new Entry(owner, w.bytes(), w.custom()));
        typeIndex.put(key, i);
        dialect(owner);
        return i;
    }

    private static String dialectOf(String opName) {
        int dot = opName.indexOf('.');
        return dot < 0 ? opName : opName.substring(0, dot);
    }

    /** Attribute/type dialect used for an operation's attributes and types. */
    private static String codecFor(Operation op) {
        return op.name().startsWith("vhlo.") ? VHLO : BUILTIN;
    }

    // ================================================================== top level

    private byte[] writeAll(Program program) {
        dialect(BUILTIN);
        collectOpNames(program.operation());
        int unknownLoc = attribute(UNKNOWN_LOCATION, BUILTIN);

        EncodingWriter ir = new EncodingWriter();
        ValueScope scope = new ValueScope();
        scope.push(0);
        // top level: a block (no arguments) holding the module
        ir.writeVarIntWithFlag(1, false);
        writeOperation(ir, program.operation(), scope, 0, unknownLoc);

        EncodingWriter file = new EncodingWriter();
        file.writeBytes(new byte[] {'M', 'L', (byte) 0xEF, 'R'});
        file.writeVarInt(version);
        file.writeNullTerminatedString(producer);
        byte[] dialectSection = dialectSection();
        byte[] attrTypeSection = attrTypeSection();
        byte[] attrTypeOffsets = attrTypeOffsetSection();
        byte[] propertiesSection =
                version >= MlirBytecodeReader.VERSION_NATIVE_PROPERTIES
                        ? propertiesSection()
                        : null;
        // the string table is complete only after every other section was encoded
        file.writeSection(MlirBytecodeReader.SECTION_STRING, stringSection());
        file.writeSection(MlirBytecodeReader.SECTION_DIALECT, dialectSection);
        file.writeSection(MlirBytecodeReader.SECTION_ATTR_TYPE, attrTypeSection);
        file.writeSection(MlirBytecodeReader.SECTION_ATTR_TYPE_OFFSET, attrTypeOffsets);
        file.writeSection(MlirBytecodeReader.SECTION_IR, ir.toByteArray());
        if (propertiesSection != null) {
            file.writeSection(MlirBytecodeReader.SECTION_PROPERTIES, propertiesSection);
        }
        return file.toByteArray();
    }

    /** Op names are numbered grouped by dialect, as the dialect section lists them. */
    private void collectOpNames(Operation root) {
        Map<String, List<String>> byDialect = new LinkedHashMap<>();
        root.walk(
                op -> {
                    String d = dialectOf(op.name());
                    dialect(d);
                    List<String> names = byDialect.computeIfAbsent(d, k -> new ArrayList<>());
                    if (!names.contains(op.name())) {
                        names.add(op.name());
                    }
                });
        for (Map.Entry<String, Integer> d : dialectIndex.entrySet()) {
            for (String name : byDialect.getOrDefault(d.getKey(), List.of())) {
                opNameIndex.put(name, opNamesInOrder.size());
                opNamesInOrder.add(name);
            }
        }
    }

    private byte[] stringSection() {
        EncodingWriter w = new EncodingWriter();
        w.writeVarInt(strings.size());
        for (int i = strings.size() - 1; i >= 0; i--) {
            w.writeVarInt(strings.get(i).length() + 1L);
        }
        for (String s : strings) {
            w.writeNullTerminatedString(s);
        }
        return w.toByteArray();
    }

    private byte[] dialectSection() {
        // op name strings must be registered before the string section is written
        Map<String, List<Integer>> grouped = new LinkedHashMap<>();
        for (String name : opNamesInOrder) {
            String d = dialectOf(name);
            String mnemonic = name.substring(Math.min(name.length(), d.length() + 1));
            grouped.computeIfAbsent(d, k -> new ArrayList<>()).add(string(mnemonic));
        }
        EncodingWriter w = new EncodingWriter();
        w.writeVarInt(dialectIndex.size());
        for (String d : dialectIndex.keySet()) {
            if (version < MlirBytecodeReader.VERSION_DIALECT_VERSIONING) {
                w.writeVarInt(string(d));
            } else if (stablehlo != null && d.equals(STABLEHLO)) {
                w.writeVarIntWithFlag(string(d), true); // the StableHLO version
                EncodingWriter v = new EncodingWriter();
                v.writeVarInt(stablehlo.major());
                v.writeVarInt(stablehlo.minor());
                v.writeVarInt(stablehlo.patch());
                w.writeSection(MlirBytecodeReader.SECTION_DIALECT_VERSIONS, v.toByteArray());
            } else {
                w.writeVarIntWithFlag(string(d), false); // no dialect version
            }
        }
        if (version >= MlirBytecodeReader.VERSION_ELIDE_UNKNOWN_BLOCK_ARG_LOCATION) {
            w.writeVarInt(opNamesInOrder.size());
        }
        for (Map.Entry<String, List<Integer>> g : grouped.entrySet()) {
            w.writeVarInt(dialectIndex.get(g.getKey()));
            w.writeVarInt(g.getValue().size());
            for (int nameIndex : g.getValue()) {
                if (version < MlirBytecodeReader.VERSION_NATIVE_PROPERTIES) {
                    w.writeVarInt(nameIndex);
                } else {
                    w.writeVarIntWithFlag(nameIndex, true); // registered
                }
            }
        }
        return w.toByteArray();
    }

    private byte[] attrTypeSection() {
        EncodingWriter w = new EncodingWriter();
        for (Entry e : attributes) {
            w.writeBytes(e.data);
        }
        for (Entry e : types) {
            w.writeBytes(e.data);
        }
        return w.toByteArray();
    }

    private byte[] attrTypeOffsetSection() {
        EncodingWriter w = new EncodingWriter();
        w.writeVarInt(attributes.size());
        w.writeVarInt(types.size());
        writeGroups(w, attributes);
        writeGroups(w, types);
        return w.toByteArray();
    }

    /** Consecutive entries of the same dialect form one group. */
    private void writeGroups(EncodingWriter w, List<Entry> entries) {
        int i = 0;
        while (i < entries.size()) {
            String d = entries.get(i).dialect;
            int j = i;
            while (j < entries.size() && entries.get(j).dialect.equals(d)) {
                j++;
            }
            w.writeVarInt(dialectIndex.get(d));
            w.writeVarInt(j - i);
            for (int k = i; k < j; k++) {
                w.writeVarIntWithFlag(entries.get(k).data.length, entries.get(k).custom);
            }
            i = j;
        }
    }

    private byte[] propertiesSection() {
        EncodingWriter w = new EncodingWriter();
        w.writeVarInt(properties.size());
        for (byte[] p : properties) {
            w.writeBlob(p);
        }
        return w.toByteArray();
    }

    // ================================================================== IR

    /** Mirror of the reader's value numbering (per isolated-from-above scope). */
    private static final class ValueScope {
        final Map<Value, Integer> ids = new IdentityHashMap<>();
        final List<Integer> nextIds = new ArrayList<>();
        final List<Integer> reserved = new ArrayList<>();
        int size;

        void push(int numValues) {
            nextIds.add(size);
            reserved.add(numValues);
            size += numValues;
        }

        void pop() {
            size -= reserved.removeLast();
            nextIds.removeLast();
        }

        void define(List<Value> values) {
            int next = nextIds.getLast();
            for (Value v : values) {
                ids.put(v, next++);
            }
            nextIds.set(nextIds.size() - 1, next);
        }

        int id(Value v) {
            Integer id = ids.get(v);
            if (id == null) {
                throw new BytecodeException("operand used before its definition");
            }
            return id;
        }
    }

    private static final int HAS_ATTRS = 0b0000_0001;
    private static final int HAS_RESULTS = 0b0000_0010;
    private static final int HAS_OPERANDS = 0b0000_0100;
    private static final int HAS_INLINE_REGIONS = 0b0001_0000;
    private static final int HAS_PROPERTIES = 0b0100_0000;

    private void writeOperation(
            EncodingWriter w, Operation op, ValueScope scope, int numValues, int unknownLoc) {
        String codec = codecFor(op);
        boolean nativeProperties = version >= MlirBytecodeReader.VERSION_NATIVE_PROPERTIES;
        List<PropertySchemas.Field> schema;
        if (stablehlo == null) {
            schema = PropertySchemas.of(op.name());
        } else {
            List<List<PropertySchemas.Field>> layouts =
                    PropertySchemas.layouts(op.name(), stablehlo);
            schema = layouts == null ? null : layouts.getFirst();
            if (schema == null && nativeProperties && !op.allAttributes().isEmpty()) {
                // MLIR would write the properties of a registered op in its own layout
                if (op.dialect().equals(STABLEHLO) || op.dialect().equals("chlo")) {
                    throw new BytecodeException("no property layout for " + op.name());
                }
            }
        }

        List<NamedAttribute> dictEntries = new ArrayList<>();
        byte[] props = null;
        if (nativeProperties && schema != null && !schema.isEmpty()) {
            props = propertiesBlob(op, schema, codec);
            for (NamedAttribute a : op.allAttributes()) {
                if (schema.stream().noneMatch(f -> f.name().equals(a.name()))) {
                    dictEntries.add(a);
                }
            }
        } else {
            dictEntries.addAll(op.allAttributes());
        }

        int mask = 0;
        if (!dictEntries.isEmpty()) {
            mask |= HAS_ATTRS;
        }
        if (props != null) {
            mask |= HAS_PROPERTIES;
        }
        if (op.numResults() > 0) {
            mask |= HAS_RESULTS;
        }
        if (op.numOperands() > 0) {
            mask |= HAS_OPERANDS;
        }
        if (!op.regions().isEmpty()) {
            mask |= HAS_INLINE_REGIONS;
        }
        w.writeVarInt(opNameIndex.get(op.name()));
        w.writeByte(mask);
        w.writeVarInt(unknownLoc);
        if ((mask & HAS_ATTRS) != 0) {
            w.writeVarInt(dictionary(dictEntries, codec));
        }
        if (props != null) {
            w.writeVarInt(properties.size());
            properties.add(props);
        }
        if (op.numResults() > 0) {
            w.writeVarInt(op.numResults());
            for (Value r : op.results()) {
                w.writeVarInt(type(r.type(), codec));
            }
        }
        if (op.numOperands() > 0) {
            w.writeVarInt(op.numOperands());
            for (Value v : op.operands()) {
                w.writeVarInt(scope.id(v));
            }
        }
        boolean isolated = op.name().equals("builtin.module");
        if (!op.regions().isEmpty()) {
            w.writeVarIntWithFlag(op.regions().size(), isolated);
        }
        if (numValues > 0 && op.numResults() > 0) {
            scope.define(op.results());
        }
        if (op.regions().isEmpty()) {
            return;
        }
        if (isolated) {
            ValueScope inner = new ValueScope();
            if (version >= MlirBytecodeReader.VERSION_LAZY_LOADING) {
                EncodingWriter section = new EncodingWriter();
                for (Region r : op.regions()) {
                    writeRegion(section, r, inner, unknownLoc);
                }
                w.writeSection(MlirBytecodeReader.SECTION_IR, section.toByteArray());
            } else {
                for (Region r : op.regions()) {
                    writeRegion(w, r, inner, unknownLoc);
                }
            }
        } else {
            for (Region r : op.regions()) {
                writeRegion(w, r, scope, unknownLoc);
            }
        }
    }

    private void writeRegion(EncodingWriter w, Region region, ValueScope scope, int unknownLoc) {
        w.writeVarInt(region.blocks().size());
        if (region.blocks().isEmpty()) {
            return;
        }
        int numValues = 0;
        for (Block b : region.blocks()) {
            numValues += b.arguments().size();
            for (Operation op : b.operations()) {
                numValues += op.numResults();
            }
        }
        w.writeVarInt(numValues);
        scope.push(numValues);
        String codec = null;
        for (Block b : region.blocks()) {
            boolean hasArgs = !b.arguments().isEmpty();
            w.writeVarIntWithFlag(b.operations().size(), hasArgs);
            if (hasArgs) {
                if (codec == null) {
                    codec = blockArgumentCodec(region);
                }
                w.writeVarInt(b.arguments().size());
                for (Value a : b.arguments()) {
                    int t = type(a.type(), codec);
                    if (version >= MlirBytecodeReader.VERSION_ELIDE_UNKNOWN_BLOCK_ARG_LOCATION) {
                        w.writeVarIntWithFlag(t, false);
                    } else {
                        w.writeVarInt(t);
                        w.writeVarInt(unknownLoc);
                    }
                }
                scope.define(b.arguments());
                if (version >= MlirBytecodeReader.VERSION_USE_LIST_ORDERING) {
                    w.writeByte(0); // no use-list orders
                }
            }
            for (Operation op : b.operations()) {
                writeOperation(w, op, scope, numValues, unknownLoc);
            }
        }
        scope.pop();
    }

    /** Block arguments carry VHLO types inside VHLO programs. */
    private String blockArgumentCodec(Region region) {
        for (Block b : region.blocks()) {
            for (Operation op : b.operations()) {
                return codecFor(op);
            }
        }
        return stablehlo != null ? BUILTIN : VHLO;
    }

    private byte[] propertiesBlob(Operation op, List<PropertySchemas.Field> schema, String codec) {
        EncodingWriter w = new EncodingWriter();
        for (PropertySchemas.Field f : schema) {
            Attribute a = op.attribute(f.name()).orElse(null);
            if (f.optional()) {
                w.writeVarIntWithFlag(a == null ? 0 : attribute(a, codec), a != null);
            } else {
                if (a == null) {
                    throw new BytecodeException(
                            op.name() + " is missing attribute '" + f.name() + "'");
                }
                w.writeVarInt(attribute(a, codec));
            }
        }
        return w.toByteArray();
    }

    /** The location attached to every operation: {@code loc(unknown)}. */
    static final Attribute UNKNOWN_LOCATION = MlirBytecodeReader.LOCATION;

    /**
     * Index of a builtin dictionary with builtin string names whose values are encoded with {@code
     * valueCodec} (an operation attribute dictionary).
     */
    int dictionary(List<NamedAttribute> entries, String valueCodec) {
        DictionaryAttr dict = new DictionaryAttr(entries);
        if (valueCodec.equals(BUILTIN)) {
            return attribute(dict, BUILTIN);
        }
        Key key = new Key(BUILTIN + "/" + valueCodec, dict);
        Integer i = attributeIndex.get(key);
        if (i != null) {
            return i;
        }
        AttrTypeWriter w = new AttrTypeWriter(this);
        w.varInt(BuiltinEncoder.DICTIONARY);
        w.varInt(dict.entries().size());
        for (NamedAttribute e : dict.entries()) {
            w.varInt(attribute(new com.qxotic.jota.stablehlo.attr.StringAttr(e.name()), BUILTIN));
            w.varInt(attribute(e.value(), valueCodec));
        }
        i = attributes.size();
        attributes.add(new Entry(BUILTIN, w.bytes(), true));
        attributeIndex.put(key, i);
        return i;
    }
}
