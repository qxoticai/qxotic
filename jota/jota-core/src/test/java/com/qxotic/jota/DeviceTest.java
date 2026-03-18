package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class DeviceTest {

    @Test
    void testDeviceTypeIdNormalization() {
        DeviceType upper = new DeviceType("CUDA");
        assertEquals("cuda", upper.id());

        DeviceType withWhitespace = new DeviceType("  device  ");
        assertEquals("device", withWhitespace.id());
    }

    @Test
    void testDeviceTypeEquality() {
        DeviceType a = new DeviceType("cuda");
        DeviceType b = new DeviceType("CUDA");
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());

        assertNotEquals(DeviceType.CUDA, DeviceType.HIP);
    }

    @Test
    void testDeviceTypeRejectsBlankId() {
        assertThrows(IllegalArgumentException.class, () -> new DeviceType(null));
        assertThrows(IllegalArgumentException.class, () -> new DeviceType(""));
        assertThrows(IllegalArgumentException.class, () -> new DeviceType("  "));
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
        Device device = new Device(DeviceType.CUDA, 0);
        assertEquals(DeviceType.CUDA, device.type());
        assertEquals(0, device.index());
        assertEquals("cuda", device.runtimeId());
    }

    @Test
    void testDeviceBelongsTo() {
        Device device = new Device(DeviceType.CUDA, 0);
        assertTrue(device.belongsTo(DeviceType.CUDA));
        assertFalse(device.belongsTo(DeviceType.HIP));

        Device panama3 = new Device(DeviceType.PANAMA, 3);
        assertTrue(panama3.belongsTo(DeviceType.PANAMA));
        assertFalse(panama3.belongsTo(DeviceType.CUDA));
    }

    @Test
    void testDeviceRejectsNegativeIndex() {
        assertThrows(IllegalArgumentException.class, () -> new Device(DeviceType.CUDA, -1));
        assertThrows(IllegalArgumentException.class, () -> new Device(DeviceType.CUDA, -2));
    }

    @Test
    void testDeviceRequiresNonNullType() {
        assertThrows(NullPointerException.class, () -> new Device(null, 0));
    }

    @Test
    void testDeviceEqualityAndHashCode() {
        Device a = new Device(DeviceType.CUDA, 0);
        Device b = new Device(DeviceType.CUDA, 0);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());

        Device c = new Device(DeviceType.CUDA, 1);
        assertNotEquals(a, c);

        Device d = new Device(DeviceType.HIP, 0);
        assertNotEquals(a, d);
    }

    @Test
    void testDeviceToString() {
        assertEquals("cuda:0", new Device(DeviceType.CUDA, 0).toString());
        assertEquals("panama:3", new Device(DeviceType.PANAMA, 3).toString());
    }

    @Test
    void testDeviceRuntimeId() {
        assertEquals("cuda", new Device(DeviceType.CUDA, 0).runtimeId());
        assertEquals("panama", new Device(DeviceType.PANAMA, 0).runtimeId());
        assertEquals("hip", new Device(DeviceType.HIP, 2).runtimeId());
    }
}
