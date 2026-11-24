package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.OpaqueAttr;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.text.ParseException;
import com.qxotic.jota.stablehlo.type.OpaqueType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reader for the MLIR bytecode format ({@code mlir/docs/BytecodeFormat.md}, {@code
 * mlir/lib/Bytecode/Reader/BytecodeReader.cpp}), bytecode versions 0 to 6.
 *
 * <p>The result is a generic {@link Program}: operations keep their dialect names ({@code
 * vhlo.add_v1} for portable artifacts), attributes and types of the {@code builtin} and {@code
 * vhlo} dialects are decoded into this model by {@link BuiltinDecoder} and {@link VhloDecoder};
 * other attributes/types with a custom encoding are rejected, those stored in assembly form are
 * parsed with {@link MlirParser}. Locations are dropped. {@link
 * com.qxotic.jota.stablehlo.vhlo.VhloToStableHlo} turns a VHLO program into StableHLO.
 *
 * <p>Not supported: block successors (multi-block CFGs), resource blobs referenced by {@code
 * dense_resource} (kept opaque), properties of registered ops without a known schema.
 */
public final class MlirBytecodeReader {

    static final int SECTION_STRING = 0;
    static final int SECTION_DIALECT = 1;
    static final int SECTION_ATTR_TYPE = 2;
    static final int SECTION_ATTR_TYPE_OFFSET = 3;
    static final int SECTION_IR = 4;
    static final int SECTION_RESOURCE = 5;
    static final int SECTION_RESOURCE_OFFSET = 6;
    static final int SECTION_DIALECT_VERSIONS = 7;
    static final int SECTION_PROPERTIES = 8;
    static final int NUM_SECTIONS = 9;

    static final int VERSION_DIALECT_VERSIONING = 1;
    static final int VERSION_LAZY_LOADING = 2;
    static final int VERSION_USE_LIST_ORDERING = 3;
    static final int VERSION_ELIDE_UNKNOWN_BLOCK_ARG_LOCATION = 4;
    static final int VERSION_NATIVE_PROPERTIES = 5;
    static final int VERSION_CURRENT = 6;

    private static final int HAS_ATTRS = 0b0000_0001;
    private static final int HAS_RESULTS = 0b0000_0010;
    private static final int HAS_OPERANDS = 0b0000_0100;
    private static final int HAS_SUCCESSORS = 0b0000_1000;
    private static final int HAS_INLINE_REGIONS = 0b0001_0000;
    private static final int HAS_USE_LIST_ORDERS = 0b0010_0000;
    private static final int HAS_PROPERTIES = 0b0100_0000;

    /** Placeholder for decoded location attributes, which this model does not keep. */
    static final Attribute LOCATION = new OpaqueAttr("loc(unknown)");

    /** A decoded bytecode file. */
    public record Result(
            long bytecodeVersion,
            String producer,
            Map<String, byte[]> dialectVersions,
            Program program) {}

    private record Dialect(String name, byte[] version) {}

    private record OpName(Dialect dialect, String name, Boolean registered) {
        String fullName() {
            return name.isEmpty() ? dialect.name : dialect.name + "." + name;
        }
    }

    private static final class Entry<T> {
        Dialect dialect;
        int start;
        int end;
        boolean custom;
        T value;
        boolean decoding;
    }

    private final byte[] data;
    private long version;
    private String producer;
    private String[] strings;
    private final List<Dialect> dialects = new ArrayList<>();
    private final List<OpName> opNames = new ArrayList<>();
    private final List<Entry<Attribute>> attributes = new ArrayList<>();
    private final List<Entry<Type>> types = new ArrayList<>();
    private EncodingReader attrTypeData;
    private EncodingReader propertiesData;
    private final List<Integer> propertiesOffsets = new ArrayList<>();

    private MlirBytecodeReader(byte[] data) {
        this.data = data;
    }

    /** Decodes a bytecode file. */
    public static Result read(byte[] data) {
        return new MlirBytecodeReader(data).readAll();
    }

    long version() {
        return version;
    }

    // ================================================================== top level

    private Result readAll() {
        EncodingReader r = new EncodingReader(data);
        if (data.length < 4
                || data[0] != 'M'
                || data[1] != 'L'
                || (data[2] & 0xFF) != 0xEF
                || data[3] != 'R') {
            throw new BytecodeException("not an MLIR bytecode file (bad magic)");
        }
        r.skipBytes(4);
        version = r.readVarInt();
        if (version > VERSION_CURRENT) {
            throw new BytecodeException(
                    "bytecode version "
                            + version
                            + " is newer than the supported "
                            + VERSION_CURRENT);
        }
        producer = r.readNullTerminatedString();
        EncodingReader[] sections = new EncodingReader[NUM_SECTIONS];
        while (!r.isEmpty()) {
            EncodingReader.Section s = r.readSection();
            if (s.id() >= NUM_SECTIONS) {
                throw r.error("invalid section id " + s.id());
            }
            if (sections[s.id()] != null) {
                throw r.error("duplicate top-level section " + s.id());
            }
            sections[s.id()] = s.data();
        }
        for (int id :
                new int[] {
                    SECTION_STRING,
                    SECTION_DIALECT,
                    SECTION_ATTR_TYPE,
                    SECTION_ATTR_TYPE_OFFSET,
                    SECTION_IR
                }) {
            if (sections[id] == null) {
                throw new BytecodeException("missing top-level section " + id);
            }
        }
        if (version >= VERSION_NATIVE_PROPERTIES && sections[SECTION_PROPERTIES] == null) {
            throw new BytecodeException("missing properties section");
        }
        readStrings(sections[SECTION_STRING]);
        if (sections[SECTION_PROPERTIES] != null) {
            readProperties(sections[SECTION_PROPERTIES]);
        }
        Map<String, byte[]> dialectVersions = readDialects(sections[SECTION_DIALECT]);
        readAttrTypeOffsets(sections[SECTION_ATTR_TYPE], sections[SECTION_ATTR_TYPE_OFFSET]);
        Program program = readIR(sections[SECTION_IR]);
        return new Result(version, producer, dialectVersions, program);
    }

    // ================================================================== strings

    private void readStrings(EncodingReader r) {
        int count = r.readSize();
        strings = new String[count];
        int[] sizes = new int[count];
        for (int i = count - 1; i >= 0; i--) {
            sizes[i] = r.readSize();
        }
        int offset = r.position();
        for (int i = 0; i < count; i++) {
            if (sizes[i] < 1 || offset + sizes[i] > r.end()) {
                throw r.error("string size exceeds the section");
            }
            strings[i] =
                    new String(
                            data,
                            offset,
                            sizes[i] - 1,
                            java.nio.charset.StandardCharsets.ISO_8859_1);
            offset += sizes[i];
        }
        if (offset != r.end()) {
            throw r.error("unexpected trailing data in string section");
        }
    }

    String string(long index) {
        if (index < 0 || index >= strings.length) {
            throw new BytecodeException("invalid string index " + index);
        }
        return strings[(int) index];
    }

    // ================================================================== dialects

    private Map<String, byte[]> readDialects(EncodingReader r) {
        int count = r.readSize();
        Map<String, byte[]> versions = new HashMap<>();
        for (int i = 0; i < count; i++) {
            if (version < VERSION_DIALECT_VERSIONING) {
                dialects.add(new Dialect(string(r.readVarInt()), null));
                continue;
            }
            long nameIndex = r.readVarIntWithFlag();
            boolean versioned = r.lastFlag();
            byte[] dialectVersion = null;
            if (versioned) {
                EncodingReader.Section s = r.readSection();
                if (s.id() != SECTION_DIALECT_VERSIONS) {
                    throw r.error("expected dialect version section");
                }
                dialectVersion = s.data().readBytes(s.data().remaining());
            }
            Dialect d = new Dialect(string(nameIndex), dialectVersion);
            dialects.add(d);
            if (dialectVersion != null) {
                versions.put(d.name, dialectVersion);
            }
        }
        if (version >= VERSION_ELIDE_UNKNOWN_BLOCK_ARG_LOCATION) {
            r.readVarInt(); // number of op names (reservation hint)
        }
        while (!r.isEmpty()) {
            Dialect d = dialect(r.readVarInt());
            int n = r.readSize();
            for (int i = 0; i < n; i++) {
                if (version < VERSION_NATIVE_PROPERTIES) {
                    opNames.add(new OpName(d, string(r.readVarInt()), null));
                } else {
                    long nameIndex = r.readVarIntWithFlag();
                    opNames.add(new OpName(d, string(nameIndex), r.lastFlag()));
                }
            }
        }
        return Collections.unmodifiableMap(versions);
    }

    private com.qxotic.jota.stablehlo.portable.StableHloVersion stablehloVersion;
    private boolean stablehloVersionRead;

    /**
     * The StableHLO version recorded with the {@code stablehlo} dialect (plain bytecode written by
     * StableHLO: major, minor, patch), or null.
     */
    private com.qxotic.jota.stablehlo.portable.StableHloVersion stablehloVersion() {
        if (!stablehloVersionRead) {
            stablehloVersionRead = true;
            for (Dialect d : dialects) {
                if (d.name.equals("stablehlo") && d.version != null) {
                    EncodingReader r = new EncodingReader(d.version);
                    stablehloVersion =
                            new com.qxotic.jota.stablehlo.portable.StableHloVersion(
                                    (int) r.readVarInt(),
                                    (int) r.readVarInt(),
                                    (int) r.readVarInt());
                }
            }
        }
        return stablehloVersion;
    }

    private Dialect dialect(long index) {
        if (index < 0 || index >= dialects.size()) {
            throw new BytecodeException("invalid dialect index " + index);
        }
        return dialects.get((int) index);
    }

    // ================================================================== properties

    private void readProperties(EncodingReader r) {
        if (r.isEmpty()) {
            return;
        }
        int count = r.readSize();
        propertiesData = r.slice(r.position(), r.end());
        EncodingReader offsets = r.slice(r.position(), r.end());
        for (int i = 0; i < count; i++) {
            propertiesOffsets.add(offsets.position());
            offsets.skipBytes(offsets.readVarInt());
        }
        if (!offsets.isEmpty()) {
            throw offsets.error("broken properties section");
        }
    }

    /** The properties blob of index {@code index} (a {@code size, bytes} record). */
    private EncodingReader propertiesBlob(long index) {
        if (propertiesData == null || index < 0 || index >= propertiesOffsets.size()) {
            throw new BytecodeException("invalid properties index " + index);
        }
        int at = propertiesOffsets.get((int) index);
        EncodingReader r = propertiesData.slice(at, propertiesData.end());
        long size = r.readVarInt();
        int start = r.skipBytes(size);
        return r.slice(start, start + (int) size);
    }

    // ================================================================== attributes & types

    private void readAttrTypeOffsets(EncodingReader dataSection, EncodingReader offsets) {
        attrTypeData = dataSection;
        int numAttributes = offsets.readSize();
        int numTypes = offsets.readSize();
        int[] cursor = {dataSection.position()};
        readEntryGroups(offsets, attributes, numAttributes, cursor);
        readEntryGroups(offsets, types, numTypes, cursor);
        if (!offsets.isEmpty()) {
            throw offsets.error("unexpected trailing data in attribute/type offset section");
        }
    }

    private <T> void readEntryGroups(
            EncodingReader offsets, List<Entry<T>> out, int count, int[] cursor) {
        while (out.size() < count) {
            Dialect d = dialect(offsets.readVarInt());
            int n = offsets.readSize();
            for (int i = 0; i < n; i++) {
                if (out.size() == count) {
                    throw offsets.error("attribute/type group exceeds the declared count");
                }
                long size = offsets.readVarIntWithFlag();
                Entry<T> e = new Entry<>();
                e.dialect = d;
                e.custom = offsets.lastFlag();
                e.start = cursor[0];
                e.end = Math.addExact(cursor[0], Math.toIntExact(size));
                if (e.end > attrTypeData.end()) {
                    throw offsets.error("attribute/type entry past the end of its section");
                }
                cursor[0] = e.end;
                out.add(e);
            }
        }
    }

    Attribute attribute(long index) {
        if (index < 0 || index >= attributes.size()) {
            throw new BytecodeException("invalid attribute index " + index);
        }
        Entry<Attribute> e = attributes.get((int) index);
        if (e.value == null) {
            if (e.decoding) {
                throw new BytecodeException("cyclic attribute #" + index);
            }
            e.decoding = true;
            try {
                e.value = decodeAttribute(e, index);
            } finally {
                e.decoding = false;
            }
        }
        return e.value;
    }

    Type type(long index) {
        if (index < 0 || index >= types.size()) {
            throw new BytecodeException("invalid type index " + index);
        }
        Entry<Type> e = types.get((int) index);
        if (e.value == null) {
            if (e.decoding) {
                throw new BytecodeException("cyclic type #" + index);
            }
            e.decoding = true;
            try {
                e.value = decodeType(e, index);
            } finally {
                e.decoding = false;
            }
        }
        return e.value;
    }

    private Attribute decodeAttribute(Entry<Attribute> e, long index) {
        EncodingReader r = attrTypeData.slice(e.start, e.end);
        if (!e.custom) {
            String asm = r.readNullTerminatedString();
            try {
                return MlirParser.parseAttribute(asm);
            } catch (ParseException ex) {
                return new OpaqueAttr(asm);
            }
        }
        DialectReader dr = new DialectReader(this, r);
        Attribute a =
                switch (e.dialect.name) {
                    case "builtin" -> BuiltinDecoder.attribute(dr);
                    case "vhlo" -> VhloDecoder.attribute(dr);
                    case "stablehlo" -> StablehloDecoder.attribute(dr);
                    default ->
                            throw new BytecodeException(
                                    "no bytecode decoder for attributes of dialect '"
                                            + e.dialect.name
                                            + "' (attribute #"
                                            + index
                                            + ")");
                };
        return a;
    }

    private Type decodeType(Entry<Type> e, long index) {
        EncodingReader r = attrTypeData.slice(e.start, e.end);
        if (!e.custom) {
            String asm = r.readNullTerminatedString();
            try {
                return MlirParser.parseType(asm);
            } catch (ParseException ex) {
                return new OpaqueType(asm);
            }
        }
        DialectReader dr = new DialectReader(this, r);
        return switch (e.dialect.name) {
            case "builtin" -> BuiltinDecoder.type(dr);
            case "vhlo" -> VhloDecoder.type(dr);
            case "stablehlo" -> StablehloDecoder.type(dr);
            default ->
                    throw new BytecodeException(
                            "no bytecode decoder for types of dialect '"
                                    + e.dialect.name
                                    + "' (type #"
                                    + index
                                    + ")");
        };
    }

    // ================================================================== IR

    /** Values of the current isolated-from-above scope, numbered in definition order. */
    private static final class ValueScope {
        final List<Value> values = new ArrayList<>();
        final List<Integer> nextIds = new ArrayList<>();

        void push(int numValues) {
            nextIds.add(values.size());
            for (int i = 0; i < numValues; i++) {
                values.add(null);
            }
        }

        void pop(int numValues) {
            for (int i = 0; i < numValues; i++) {
                values.removeLast();
            }
            nextIds.removeLast();
        }

        void define(List<Value> newValues, EncodingReader r) {
            int next = nextIds.getLast();
            if (next + newValues.size() > values.size()) {
                throw r.error("value index range outside of the parent region");
            }
            for (Value v : newValues) {
                values.set(next++, v);
            }
            nextIds.set(nextIds.size() - 1, next);
        }
    }

    private Program readIR(EncodingReader r) {
        Block top = new Block();
        ValueScope scope = new ValueScope();
        scope.push(0);
        readBlock(r, top, scope, 0);
        if (!r.isEmpty()) {
            throw r.error("unexpected trailing data in IR section");
        }
        List<Operation> ops = top.operations();
        if (ops.size() == 1 && ops.getFirst().name().equals("builtin.module")) {
            return new Program(ops.getFirst());
        }
        return new Program(
                Operation.create(
                        "builtin.module",
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(Region.of(top))));
    }

    /** Block header and operations; {@code numValues} > 0 means results must be numbered. */
    private void readBlock(EncodingReader r, Block block, ValueScope scope, int numValues) {
        long numOps = r.readVarIntWithFlag();
        boolean hasArgs = r.lastFlag();
        if (hasArgs) {
            int numArgs = r.readSize();
            List<Value> args = new ArrayList<>(numArgs);
            for (int i = 0; i < numArgs; i++) {
                Type t;
                if (version >= VERSION_ELIDE_UNKNOWN_BLOCK_ARG_LOCATION) {
                    long typeIndex = r.readVarIntWithFlag();
                    boolean hasLoc = r.lastFlag();
                    t = type(typeIndex);
                    if (hasLoc) {
                        r.readVarInt();
                    }
                } else {
                    t = type(r.readVarInt());
                    r.readVarInt(); // location
                }
                args.add(block.addArgument(t, null));
            }
            scope.define(args, r);
        }
        if (version >= VERSION_USE_LIST_ORDERING && hasArgs) {
            int hasUseListOrders = r.readByte();
            if (hasUseListOrders != 0) {
                skipUseListOrders(r, block.arguments().size());
            }
        }
        for (long i = 0; i < numOps; i++) {
            readOperation(r, block, scope, numValues);
        }
    }

    private void skipUseListOrders(EncodingReader r, int numValues) {
        long count = 1;
        if (numValues > 1) {
            count = r.readVarInt();
        }
        for (long v = 0; v < count; v++) {
            if (numValues > 1) {
                r.readVarInt(); // value index
            }
            long n = r.readVarIntWithFlag();
            for (long k = 0; k < n; k++) {
                r.readVarInt();
            }
        }
    }

    private void readOperation(EncodingReader r, Block block, ValueScope scope, int numValues) {
        long nameIndex = r.readVarInt();
        if (nameIndex < 0 || nameIndex >= opNames.size()) {
            throw r.error("invalid operation name index " + nameIndex);
        }
        OpName opName = opNames.get((int) nameIndex);
        String name = opName.fullName();
        int mask = r.readByte();
        r.readVarInt(); // location

        List<NamedAttribute> attrs = List.of();
        if ((mask & HAS_ATTRS) != 0) {
            Attribute dict = attribute(r.readVarInt());
            if (!(dict instanceof DictionaryAttr d)) {
                throw r.error("operation attributes are not a dictionary: " + dict);
            }
            attrs = d.entries();
        }
        List<NamedAttribute> props = List.of();
        if ((mask & HAS_PROPERTIES) != 0) {
            if (opName.registered == null) {
                throw r.error(
                        "properties without the registered flag (bytecode version "
                                + version
                                + ")");
            }
            props = readOperationProperties(r, name, opName.registered);
        }
        List<Type> resultTypes = List.of();
        if ((mask & HAS_RESULTS) != 0) {
            int n = r.readSize();
            resultTypes = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                resultTypes.add(type(r.readVarInt()));
            }
        }
        List<Value> operands = List.of();
        if ((mask & HAS_OPERANDS) != 0) {
            int n = r.readSize();
            operands = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                long id = r.readVarInt();
                if (id < 0 || id >= scope.values.size() || scope.values.get((int) id) == null) {
                    throw r.error(
                            "operand #"
                                    + i
                                    + " of "
                                    + name
                                    + " refers to an undefined value "
                                    + id
                                    + " (forward references are not supported)");
                }
                operands.add(scope.values.get((int) id));
            }
        }
        if ((mask & HAS_SUCCESSORS) != 0) {
            throw r.error("operation " + name + " has block successors, which are not supported");
        }
        if (version >= VERSION_USE_LIST_ORDERING && (mask & HAS_USE_LIST_ORDERS) != 0) {
            skipUseListOrders(r, resultTypes.size());
        }
        int numRegions = 0;
        boolean isolated = false;
        if ((mask & HAS_INLINE_REGIONS) != 0) {
            numRegions = Math.toIntExact(r.readVarIntWithFlag());
            isolated = r.lastFlag();
        }
        List<Region> regions = new ArrayList<>(numRegions);
        for (int i = 0; i < numRegions; i++) {
            regions.add(new Region());
        }
        Operation op = Operation.create(name, operands, resultTypes, props, attrs, regions);
        block.addOperation(op);
        if (numValues > 0 && op.numResults() > 0) {
            scope.define(op.results(), r);
        }
        if (numRegions > 0) {
            EncodingReader regionReader = r;
            ValueScope regionScope = scope;
            if (isolated) {
                regionScope = new ValueScope();
                if (version >= VERSION_LAZY_LOADING) {
                    EncodingReader.Section s = r.readSection();
                    if (s.id() != SECTION_IR) {
                        throw r.error("expected IR section for an isolated region");
                    }
                    regionReader = s.data();
                }
            }
            for (Region region : regions) {
                readRegion(regionReader, region, regionScope);
            }
            if (isolated && regionReader != r && !regionReader.isEmpty()) {
                throw regionReader.error("unexpected trailing data in region section");
            }
        }
    }

    private void readRegion(EncodingReader r, Region region, ValueScope scope) {
        int numBlocks = r.readSize();
        if (numBlocks == 0) {
            return;
        }
        int numValues = r.readSize();
        scope.push(numValues);
        List<Block> blocks = new ArrayList<>(numBlocks);
        for (int i = 0; i < numBlocks; i++) {
            Block b = new Block();
            blocks.add(b);
            region.addBlock(b);
        }
        for (Block b : blocks) {
            readBlock(r, b, scope, numValues);
        }
        scope.pop(numValues);
    }

    // ================================================================== op properties

    private List<NamedAttribute> readOperationProperties(
            EncodingReader r, String opName, boolean registered) {
        long index = r.readVarInt();
        List<List<PropertySchemas.Field>> layouts =
                PropertySchemas.layouts(opName, stablehloVersion());
        if (layouts == null) {
            if (registered) {
                throw new BytecodeException(
                        "no property schema for registered operation '" + opName + "'");
            }
            Attribute a = new DialectReader(this, propertiesBlob(index)).attribute();
            List<NamedAttribute> out = new ArrayList<>();
            if (a instanceof DictionaryAttr d) {
                out.addAll(d.entries());
            } else {
                out.add(new NamedAttribute("properties", a));
            }
            return out;
        }
        // an op may have gained properties across LLVM versions (func.func, builtin.module got
        // an optional dlti): each property is one reference, so the layout that consumes the
        // whole blob is the one it was written with
        BytecodeException first = null;
        for (List<PropertySchemas.Field> schema : layouts) {
            EncodingReader blob = propertiesBlob(index);
            try {
                List<NamedAttribute> out = readProperties(blob, schema);
                if (blob.isEmpty()) {
                    return out;
                }
                throw blob.error("unexpected trailing data in properties of " + opName);
            } catch (BytecodeException e) {
                if (first == null) {
                    first = e;
                }
            }
        }
        throw new BytecodeException(
                "cannot read the properties of " + opName + ": " + first.getMessage());
    }

    private List<NamedAttribute> readProperties(
            EncodingReader blob, List<PropertySchemas.Field> schema) {
        DialectReader dr = new DialectReader(this, blob);
        List<NamedAttribute> out = new ArrayList<>();
        for (PropertySchemas.Field f : schema) {
            Attribute a = f.optional() ? dr.optionalAttribute() : dr.attribute();
            if (a != null && f.name().equals(PropertySchemas.UNKNOWN)) {
                throw blob.error("unknown trailing property set: " + a);
            }
            if (a != null) {
                out.add(new NamedAttribute(f.name(), a));
            }
        }
        return out;
    }

    static NamedAttribute named(Attribute key, Attribute value) {
        if (key instanceof StringAttr s) {
            return new NamedAttribute(s.value(), value);
        }
        throw new BytecodeException("attribute name is not a string: " + key);
    }
}
