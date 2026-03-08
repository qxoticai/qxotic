package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class DeviceTest {

    @Test
    void testRootDeviceCreation() {
        Device custom = Device.of("custom");
        assertEquals("custom", custom.name());
        assertEquals("custom", custom.leafName());
        assertNull(custom.parent());
        assertSame(custom, custom.root());
    }

    @Test
    void testChildDeviceCreation() {
        Device child = Device.CPU.child("test");
        assertEquals("cpu:test", child.name());
        assertEquals("test", child.leafName());
        assertSame(Device.CPU, child.parent());
        assertSame(Device.CPU, child.root());
    }

    @Test
    void testChildValidation() {
        assertThrows(IllegalArgumentException.class, () -> Device.CPU.child(null));
        assertThrows(IllegalArgumentException.class, () -> Device.CPU.child(""));
        assertThrows(IllegalArgumentException.class, () -> Device.CPU.child("  "));
        assertThrows(IllegalArgumentException.class, () -> Device.CPU.child("test:invalid"));
    }

    @Test
    void testPredefinedConstantsHierarchy() {
        assertEquals("cpu", Device.CPU.name());
        assertNull(Device.CPU.parent());
        assertSame(Device.CPU, Device.CPU.root());

        assertEquals("gpu", Device.GPU.name());
        assertNull(Device.GPU.parent());
        assertSame(Device.GPU, Device.GPU.root());

        assertEquals("cpu:panama", Device.PANAMA.name());
        assertSame(Device.CPU, Device.PANAMA.parent());
        assertSame(Device.CPU, Device.PANAMA.root());

        assertEquals("cpu:java", Device.JAVA.name());
        assertSame(Device.CPU, Device.JAVA.parent());
        assertSame(Device.CPU, Device.JAVA.root());

        assertEquals("cpu:native", Device.NATIVE.name());
        assertSame(Device.CPU, Device.NATIVE.parent());
        assertSame(Device.CPU, Device.NATIVE.root());

        assertEquals("gpu:cuda", Device.CUDA.name());
        assertSame(Device.GPU, Device.CUDA.parent());
        assertSame(Device.GPU, Device.CUDA.root());

        assertEquals("gpu:opencl", Device.OPENCL.name());
        assertSame(Device.GPU, Device.OPENCL.parent());
        assertSame(Device.GPU, Device.OPENCL.root());
    }

    @Test
    void testNamingConventions() {
        Device upperCase = Device.of("CPU");
        assertEquals("cpu", upperCase.name());

        Device withWhitespace = Device.of("  device  ");
        assertEquals("device", withWhitespace.name());

        Device hierarchical = Device.CPU.child("a").child("b").child("c");
        assertEquals("cpu:a:b:c", hierarchical.name());
    }

    @Test
    void testEqualityAndHashCode() {
        Device cpuTest1 = Device.CPU.child("test");
        Device cpuTest2 = Device.CPU.child("test");
        assertEquals(cpuTest1, cpuTest2);
        assertEquals(cpuTest1.hashCode(), cpuTest2.hashCode());

        Device cpuA = Device.CPU.child("a");
        Device cpuB = Device.CPU.child("b");
        assertNotEquals(cpuA, cpuB);

        Device cpuA2 = Device.CPU.child("a");
        Device gpuA = Device.GPU.child("a");
        assertNotEquals(cpuA2, gpuA);
    }

    @Test
    void testBelongsTo() {
        assertTrue(Device.PANAMA.belongsTo(Device.CPU));
        assertTrue(Device.JAVA.belongsTo(Device.CPU));
        assertTrue(Device.NATIVE.belongsTo(Device.CPU));
        assertTrue(Device.CUDA.belongsTo(Device.GPU));
        assertFalse(Device.CPU.belongsTo(Device.PANAMA));
        assertFalse(Device.PANAMA.belongsTo(Device.NATIVE));
        assertFalse(Device.GPU.belongsTo(Device.CPU));

        assertTrue(Device.CPU.belongsTo(Device.CPU));
        assertTrue(Device.GPU.belongsTo(Device.GPU));
        assertTrue(Device.PANAMA.belongsTo(Device.PANAMA));
        assertTrue(Device.JAVA.belongsTo(Device.JAVA));

        Device cudaDevice0 = Device.GPU.child("cuda").child("0");
        assertTrue(cudaDevice0.belongsTo(Device.GPU));
        assertTrue(cudaDevice0.belongsTo(Device.CUDA));
        assertFalse(Device.GPU.belongsTo(cudaDevice0));

        Device cpuTest = Device.CPU.child("test");
        assertFalse(Device.CPU.belongsTo(cpuTest));

        Device deepChild = Device.CPU.child("a").child("b").child("c");
        assertTrue(deepChild.belongsTo(Device.CPU));
        assertTrue(deepChild.belongsTo(Device.CPU.child("a")));
        assertFalse(Device.CPU.child("a").belongsTo(deepChild));

        Device cpuSibling = Device.CPU.child("sibling");
        assertFalse(cpuTest.belongsTo(cpuSibling));
        assertFalse(cpuSibling.belongsTo(cpuTest));

        Device hipX = Device.GPU.child("hipx");
        assertFalse(hipX.belongsTo(Device.HIP));
    }
}
