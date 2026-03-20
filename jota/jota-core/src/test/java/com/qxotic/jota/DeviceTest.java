package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class DeviceTest {

    @Test
    void testDeviceTypeFromIdNormalizesInput() {
        assertEquals(DeviceType.CUDA, DeviceType.fromId("cuda"));
        assertEquals(DeviceType.CUDA, DeviceType.fromId("CUDA"));
        assertEquals(DeviceType.CUDA, DeviceType.fromId("  CuDa  "));
    }

    @Test
    void testDeviceTypeEquality() {
        assertEquals(DeviceType.CUDA, DeviceType.CUDA);
        assertNotEquals(DeviceType.CUDA, DeviceType.HIP);
    }

    @Test
    void testDeviceTypeFromIdRejectsUnknown() {
        assertThrows(NullPointerException.class, () -> DeviceType.fromId(null));
        assertThrows(IllegalArgumentException.class, () -> DeviceType.fromId(""));
        assertEquals("unknown", DeviceType.fromId("unknown").id());
    }

    @Test
    void testDeviceTypePredefinedConstants() {
        assertEquals("panama", DeviceType.PANAMA.id());
        assertEquals("java", DeviceType.JAVA.id());
        assertEquals("cuda", DeviceType.CUDA.id());
        assertEquals("hip", DeviceType.HIP.id());
        assertEquals("c", DeviceType.C.id());
        assertEquals("opencl", DeviceType.OPENCL.id());
        assertEquals("metal", DeviceType.METAL.id());
        assertEquals("mojo", DeviceType.MOJO.id());
    }

    @Test
    void testDeviceRecordBasics() {
        Device device = DeviceType.CUDA.deviceIndex(0);
        assertEquals(DeviceType.CUDA, device.type());
        assertEquals(0, device.index());
        assertEquals("cuda", device.runtimeId());
    }

    @Test
    void testDeviceBelongsTo() {
        Device device = DeviceType.CUDA.deviceIndex(0);
        assertTrue(device.belongsTo(DeviceType.CUDA));
        assertFalse(device.belongsTo(DeviceType.HIP));

        Device panama3 = DeviceType.PANAMA.deviceIndex(3);
        assertTrue(panama3.belongsTo(DeviceType.PANAMA));
        assertFalse(panama3.belongsTo(DeviceType.CUDA));
    }

    @Test
    void testDeviceRejectsNegativeIndex() {
        assertThrows(IllegalArgumentException.class, () -> DeviceType.CUDA.deviceIndex(-1));
        assertThrows(IllegalArgumentException.class, () -> DeviceType.CUDA.deviceIndex(-2));
    }

    @Test
    void testDeviceRequiresNonNullType() {
        DeviceType type = null;
        assertThrows(NullPointerException.class, () -> type.deviceIndex(0));
    }

    @Test
    void testDeviceEqualityAndHashCode() {
        Device a = DeviceType.CUDA.deviceIndex(0);
        Device b = DeviceType.CUDA.deviceIndex(0);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());

        Device c = DeviceType.CUDA.deviceIndex(1);
        assertNotEquals(a, c);

        Device d = DeviceType.HIP.deviceIndex(0);
        assertNotEquals(a, d);
    }

    @Test
    void testDeviceToString() {
        assertEquals("cuda:0", DeviceType.CUDA.deviceIndex(0).toString());
        assertEquals("panama:3", DeviceType.PANAMA.deviceIndex(3).toString());
    }

    @Test
    void testDeviceRuntimeId() {
        assertEquals("cuda", DeviceType.CUDA.deviceIndex(0).runtimeId());
        assertEquals("panama", DeviceType.PANAMA.deviceIndex(0).runtimeId());
        assertEquals("hip", DeviceType.HIP.deviceIndex(2).runtimeId());
    }
}
