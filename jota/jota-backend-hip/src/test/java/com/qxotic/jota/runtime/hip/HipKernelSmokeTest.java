package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.random.RandomAlgorithms;
import com.qxotic.jota.random.RandomKey;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.TestKernels;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class HipKernelSmokeTest {

    @Test
    void launchesVecAddKernel() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 1024;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = n - i;
        }

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        var hostMemA = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        var hostMemB = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostA =
                MemoryView.of(hostMemA, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryView<MemorySegment> hostB =
                MemoryView.of(hostMemB, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> hostAccess = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostA, i);
            hostAccess.writeFloat(hostA.memory(), offset, a[i]);
            hostAccess.writeFloat(hostB.memory(), offset, b[i]);
        }

        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devA =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryView<HipDevicePtr> devB =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryView<HipDevicePtr> devOut =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));

        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(hostA.memory(), hostA.byteOffset(), devA.memory(), 0, byteSize);
        device.memoryOperations()
                .copyFromNative(hostB.memory(), hostB.byteOffset(), devB.memory(), 0, byteSize);

        Tensor aTensor = Tensor.of(devA);
        Tensor bTensor = Tensor.of(devB);
        MemoryView<?> output = aTensor.add(bTensor).materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) output;

        MemoryAccess<MemorySegment> access = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            float value = access.readFloat(hostView.memory(), offset);
            assertEquals(a[i] + b[i], value, 0.0001f);
        }
    }

    @Test
    void launchesLirKernel() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 32;
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = i * 0.25f - 2.0f;
        }

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostInput =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> hostAccess = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostInput, i);
            hostAccess.writeFloat(hostInput.memory(), offset, values[i]);
        }

        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devInput =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(
                        hostInput.memory(), hostInput.byteOffset(), devInput.memory(), 0, byteSize);

        Tensor inputTensor = Tensor.of(devInput);
        Tensor traced = Tracer.trace(inputTensor, t -> t.multiply(2.0f).add(1.0f));
        MemoryView<?> output = traced.materialize();

        MemoryView<MemorySegment> hostOutput = toHost(output);
        MemoryAccess<MemorySegment> access = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOutput, i);
            float value = access.readFloat(hostOutput.memory(), offset);
            float expected = values[i] * 2.0f + 1.0f;
            assertEquals(expected, value, 1e-4f);
        }
    }

    @Test
    void launchesLirGeluKernel() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 8;
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = i - 4.0f;
        }

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostInput =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> hostAccess = host.directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostInput, i);
            hostAccess.writeFloat(hostInput.memory(), offset, values[i]);
        }

        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devInput =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        long byteSize = (long) n * Float.BYTES;
        device.memoryOperations()
                .copyFromNative(
                        hostInput.memory(), hostInput.byteOffset(), devInput.memory(), 0, byteSize);

        Tensor traced = Tracer.trace(Tensor.of(devInput), Tensor::gelu);
        MemoryView<?> output = traced.materialize();
        MemoryView<MemorySegment> hostOutput = toHost(output);

        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOutput, i);
            float value = hostAccess.readFloat(hostOutput.memory(), offset);
            assertEquals(TestKernels.gelu(values[i]), value, 1e-4f);
        }
    }

    @Test
    void launchesLirRandomKernelWithGoldenParity() throws Exception {
        HipTestAssumptions.assumeHipReady();

        Environment current = Environment.current();
        Environment hipEnv =
                new Environment(Device.HIP, current.defaultFloat(), current.runtimes());
        RandomKey key = RandomKey.of(2026L);
        int n = 64;

        MemoryView<?> outFp32 =
                Environment.with(
                        hipEnv,
                        () ->
                                Tracer.trace(Tensor.rand(key, Shape.of(n), DataType.FP32), x -> x)
                                        .materialize());
        MemoryView<?> outFp64 =
                Environment.with(
                        hipEnv,
                        () ->
                                Tracer.trace(Tensor.rand(key, Shape.of(n), DataType.FP64), x -> x)
                                        .materialize());

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<MemorySegment> fp32 = toHost(outFp32);
        MemoryView<MemorySegment> fp64 = toHost(outFp64);
        MemoryAccess<MemorySegment> access = host.directAccess();

        for (int i = 0; i < n; i++) {
            long off32 = Indexing.linearToOffset(fp32, i);
            int actual32 = Float.floatToRawIntBits(access.readFloat(fp32.memory(), off32));
            int expected32 = Float.floatToRawIntBits(RandomAlgorithms.uniformFp32(i, key));
            assertEquals(expected32, actual32);

            long off64 = Indexing.linearToOffset(fp64, i);
            long actual64 = Double.doubleToRawLongBits(access.readDouble(fp64.memory(), off64));
            long expected64 = Double.doubleToRawLongBits(RandomAlgorithms.uniformFp64(i, key));
            assertEquals(expected64, actual64);
        }
    }

    @Test
    void transfersTransposedIotaWithToDeviceHip() {
        HipTestAssumptions.assumeHipReady();

        Tensor src = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4)).transpose(0, 1);
        Tensor dst = src.to(Device.HIP);

        assertEquals(Device.HIP, dst.device());
        assertEquals(src.shape(), dst.shape());
        assertEquals(src.dataType(), dst.dataType());
        assertEquals(Layout.rowMajor(src.shape()), dst.layout());

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<MemorySegment> hostDst = toHost(dst.materialize());
        MemoryView<MemorySegment> hostSrc =
                (MemoryView<MemorySegment>) src.to(Device.NATIVE).materialize();
        MemoryAccess<MemorySegment> access = host.directAccess();

        long n = src.shape().size();
        for (int i = 0; i < n; i++) {
            long srcOffset = Indexing.linearToOffset(hostSrc, i);
            long dstOffset = Indexing.linearToOffset(hostDst, i);
            int srcBits = Float.floatToRawIntBits(access.readFloat(hostSrc.memory(), srcOffset));
            int dstBits = Float.floatToRawIntBits(access.readFloat(hostDst.memory(), dstOffset));
            assertEquals(srcBits, dstBits);
        }
    }

    @Test
    void transfersBroadcastedConstantWithToDeviceHip() {
        HipTestAssumptions.assumeHipReady();

        Tensor src = Tensor.full(42L, Shape.of(1, 1)).broadcast(Shape.of(5, 7));
        Tensor dst = src.to(Device.HIP);

        assertEquals(Device.HIP, dst.device());
        assertEquals(src.shape(), dst.shape());
        assertEquals(src.dataType(), dst.dataType());
        assertEquals(Layout.rowMajor(src.shape()), dst.layout());

        MemoryDomain<MemorySegment> host = DomainFactory.ofMemorySegment();
        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<MemorySegment> hostDst = toHost(dst.materialize());
        MemoryAccess<MemorySegment> access = host.directAccess();

        long n = src.shape().size();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostDst, i);
            assertEquals(42L, access.readLong(hostDst.memory(), offset));
        }
    }

    @Test
    void fillByteSetsAllBytes() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 1024;
        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devBuf =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.I8, n),
                        DataType.I8,
                        Layout.rowMajor(Shape.flat(n)));

        byte fillValue = (byte) 0xAB;
        device.memoryOperations().fillByte(devBuf.memory(), 0, n, fillValue);

        MemoryView<MemorySegment> hostBuf = toHost(devBuf);
        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostBuf, i);
            assertEquals(fillValue, access.readByte(hostBuf.memory(), offset));
        }
    }

    @Test
    void fillIntSetsAllInts() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 256;
        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devBuf =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.I32, n),
                        DataType.I32,
                        Layout.rowMajor(Shape.flat(n)));

        int fillValue = 0xDEADBEEF;
        device.memoryOperations().fillInt(devBuf.memory(), 0, (long) n * Integer.BYTES, fillValue);

        MemoryView<MemorySegment> hostBuf = toHost(devBuf);
        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostBuf, i);
            assertEquals(fillValue, access.readInt(hostBuf.memory(), offset));
        }
    }

    @Test
    void fillLongSetsAllLongs() throws Exception {
        HipTestAssumptions.assumeHipReady();

        int n = 128;
        HipMemoryDomain device = HipMemoryDomain.instance();
        MemoryView<HipDevicePtr> devBuf =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.I64, n),
                        DataType.I64,
                        Layout.rowMajor(Shape.flat(n)));

        long fillValue = 0xDEADBEEFCAFEBABEL;
        device.memoryOperations().fillLong(devBuf.memory(), 0, (long) n * Long.BYTES, fillValue);

        MemoryView<MemorySegment> hostBuf = toHost(devBuf);
        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostBuf, i);
            assertEquals(fillValue, access.readLong(hostBuf.memory(), offset));
        }
    }

    @Test
    void copiesHeapBackedFloatMemorySegmentToDeviceAndBack() {
        HipTestAssumptions.assumeHipReady();

        float[] values = {1.5f, -2.25f, 0.0f, 9.75f, -6.5f};
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        MemoryFactory.ofMemorySegment(MemorySegment.ofArray(values)),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(values.length)));

        Tensor onHip = Tensor.of(hostView).to(Device.HIP);
        MemoryView<MemorySegment> roundTrip = toHost(onHip.materialize());
        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = Indexing.linearToOffset(roundTrip, i);
            assertEquals(values[i], access.readFloat(roundTrip.memory(), offset), 1e-6f);
        }
    }

    private static MemoryView<MemorySegment> toHost(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostView =
                (MemoryView<MemorySegment>) Tensor.of(view).to(Device.NATIVE).materialize();
        return hostView;
    }

    // HIP runtime/device assumptions are centralized in HipTestAssumptions.
}
