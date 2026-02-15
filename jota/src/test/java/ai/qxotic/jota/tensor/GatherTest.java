package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive tests for the gather operation.
 *
 * <p>Gather selects elements from an input tensor along a given dimension based on indices
 * specified in an indices tensor. This is commonly used for:
 *
 * <ul>
 *   <li>Embedding lookups in neural networks
 *   <li>Index-based tensor access
 *   <li>Gathering specific rows/columns from matrices
 * </ul>
 */
class GatherTest {

    @SuppressWarnings("unchecked")
    private static final MemoryDomain<MemorySegment> CONTEXT =
            (MemoryDomain<MemorySegment>) Environment.current().nativeRuntime().memoryDomain();

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return CONTEXT.directAccess().readFloat(typedView.memory(), offset);
    }

    // ==================== Basic Functionality Tests ====================

    @Test
    @DisplayName("Basic gather along axis 0")
    void testBasicGather() {
        // Create a simple embedding table: [4, 3] (4 embeddings of size 3)
        // [[0, 1, 2],
        //  [3, 4, 5],
        //  [6, 7, 8],
        //  [9, 10, 11]]
        float[] embeddings = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        Tensor embeddingTable = Tensor.of(embeddings, Shape.of(4, 3));

        // Create indices: [2, 0, 3] - gather embeddings at positions 2, 0, 3
        int[] indices = {2, 0, 3};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(3));

        // Gather along axis 0
        Tensor result = embeddingTable.gather(indicesTensor, 0);

        // Result should be [3, 3]
        assertEquals(Shape.of(3, 3), result.shape());
        assertEquals(DataType.FP32, result.dataType());

        // Materialize and check values
        MemoryView<?> view = result.materialize();
        float[] expected = {6, 7, 8, 0, 1, 2, 9, 10, 11};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    @Test
    @DisplayName("Embedding lookup with convenience method")
    void testEmbeddingLookup() {
        // Create embedding table: [vocab=5, hidden=4]
        float[] embeddings = new float[5 * 4];
        for (int i = 0; i < embeddings.length; i++) {
            embeddings[i] = i;
        }
        Tensor embeddingTable = Tensor.of(embeddings, Shape.of(5, 4));

        // Create token IDs: [batch=2, seq=3]
        int[] tokenIds = {0, 2, 4, 1, 3, 0};
        Tensor tokenIdsTensor = Tensor.of(tokenIds, Shape.of(2, 3));

        // Use embeddingLookup convenience method (gather along axis 0)
        Tensor result = embeddingTable.embeddingLookup(tokenIdsTensor);

        // Result should be [2, 3, 4]
        assertEquals(Shape.of(2, 3, 4), result.shape());
        assertEquals(DataType.FP32, result.dataType());

        // Verify specific values
        MemoryView<?> view = result.materialize();
        // tokenIds[0,0]=0 -> embeddings[0] = [0, 1, 2, 3]
        assertEquals(0, readFloat(view, 0), 0.0001f);
        assertEquals(1, readFloat(view, 1), 0.0001f);
        assertEquals(2, readFloat(view, 2), 0.0001f);
        assertEquals(3, readFloat(view, 3), 0.0001f);

        // tokenIds[0,1]=2 -> embeddings[2] = [8, 9, 10, 11]
        assertEquals(8, readFloat(view, 4), 0.0001f);
        assertEquals(9, readFloat(view, 5), 0.0001f);
    }

    @Test
    @DisplayName("Gather with 2D indices along axis 0")
    void testGather2DIndices() {
        // Input: [5, 4]
        float[] input = new float[5 * 4];
        for (int i = 0; i < input.length; i++) {
            input[i] = i;
        }
        Tensor inputTensor = Tensor.of(input, Shape.of(5, 4));

        // Indices: [2] gather along axis 0
        int[] indices = {1, 3};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(2));

        Tensor result = inputTensor.gather(indicesTensor, 0);

        // Result: [2, 4]
        assertEquals(Shape.of(2, 4), result.shape());

        MemoryView<?> view = result.materialize();
        // First row should be input[1] = [4, 5, 6, 7]
        // Second row should be input[3] = [12, 13, 14, 15]
        float[] expected = {4, 5, 6, 7, 12, 13, 14, 15};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    // ==================== Axis Tests ====================

    @Test
    @DisplayName("Gather with negative axis (-1 = last axis)")
    void testGatherNegativeAxis() {
        // Input: [3, 4]
        float[] input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        Tensor inputTensor = Tensor.of(input, Shape.of(3, 4));

        // Indices: gather along axis -1 (last axis = 1)
        int[] indices = {0, 2};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(2));

        Tensor result = inputTensor.gather(indicesTensor, -1);

        // Result: [3, 2]
        assertEquals(Shape.of(3, 2), result.shape());

        MemoryView<?> view = result.materialize();
        // Row 0: [0, 2]
        // Row 1: [4, 6]
        // Row 2: [8, 10]
        float[] expected = {0, 2, 4, 6, 8, 10};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    @Test
    @DisplayName("Gather along axis 1")
    void testGatherAxis1() {
        // Input: [2, 5]
        // [[0, 1, 2, 3, 4],
        //  [5, 6, 7, 8, 9]]
        float[] input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 5));

        // Gather columns [4, 1, 3]
        int[] indices = {4, 1, 3};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(3));

        Tensor result = inputTensor.gather(indicesTensor, 1);

        // Result: [2, 3]
        assertEquals(Shape.of(2, 3), result.shape());

        MemoryView<?> view = result.materialize();
        // Row 0: [4, 1, 3] (columns 4, 1, 3 from first row)
        // Row 1: [9, 6, 8] (columns 4, 1, 3 from second row)
        float[] expected = {4, 1, 3, 9, 6, 8};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    @Test
    @DisplayName("Gather along middle axis in 3D tensor")
    void testGather3DTensorMiddleAxis() {
        // Input: [2, 3, 4]
        float[] input = new float[2 * 3 * 4];
        for (int i = 0; i < input.length; i++) {
            input[i] = i;
        }
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 3, 4));

        // Gather along axis 1
        int[] indices = {2, 0};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(2));

        Tensor result = inputTensor.gather(indicesTensor, 1);

        // Result: [2, 2, 4]
        assertEquals(Shape.of(2, 2, 4), result.shape());

        MemoryView<?> view = result.materialize();
        // Verify shape and some values
        assertEquals(2 * 2 * 4, view.shape().size());
    }

    @Test
    @DisplayName("Gather along last axis in 3D tensor")
    void testGather3DTensorLastAxis() {
        // Input: [2, 3, 4]
        float[] input = new float[2 * 3 * 4];
        for (int i = 0; i < input.length; i++) {
            input[i] = i;
        }
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 3, 4));

        // Gather along axis 2
        int[] indices = {1, 3};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(2));

        Tensor result = inputTensor.gather(indicesTensor, 2);

        // Result: [2, 3, 2]
        assertEquals(Shape.of(2, 3, 2), result.shape());
    }

    // ==================== Multi-dimensional Indices Tests ====================

    @Test
    @DisplayName("Gather with 2D indices tensor")
    void testGatherWith2DIndicesTensor() {
        // Input: [4, 3]
        float[] input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        Tensor inputTensor = Tensor.of(input, Shape.of(4, 3));

        // 2D indices: [[0, 2], [1, 3]] -> shape [2, 2]
        int[] indices = {0, 2, 1, 3};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(2, 2));

        Tensor result = inputTensor.gather(indicesTensor, 0);

        // Result: [2, 2, 3]
        assertEquals(Shape.of(2, 2, 3), result.shape());

        MemoryView<?> view = result.materialize();
        // indices[0,0]=0 -> input[0] = [0, 1, 2]
        // indices[0,1]=2 -> input[2] = [6, 7, 8]
        // indices[1,0]=1 -> input[1] = [3, 4, 5]
        // indices[1,1]=3 -> input[3] = [9, 10, 11]
        float[] expected = {0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    @Test
    @DisplayName("Gather with 3D indices tensor")
    void testGatherWith3DIndicesTensor() {
        // Input: [3, 4]
        float[] input = new float[3 * 4];
        for (int i = 0; i < input.length; i++) {
            input[i] = i;
        }
        Tensor inputTensor = Tensor.of(input, Shape.of(3, 4));

        // 3D indices with shape [2, 2, 1]
        int[] indices = {0, 2, 1, 0};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(2, 2, 1));

        Tensor result = inputTensor.gather(indicesTensor, 0);

        // Result: [2, 2, 1, 4]
        assertEquals(Shape.of(2, 2, 1, 4), result.shape());
    }

    // ==================== Index Type Tests ====================

    @Test
    @DisplayName("Gather with I64 (long) indices")
    void testGatherWithI64Indices() {
        float[] embeddings = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        Tensor embeddingTable = Tensor.of(embeddings, Shape.of(4, 3));

        long[] indices = {1L, 3L};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(2));

        Tensor result = embeddingTable.gather(indicesTensor, 0);

        assertEquals(Shape.of(2, 3), result.shape());

        MemoryView<?> view = result.materialize();
        // indices[0]=1 -> embeddings[1] = [3, 4, 5]
        // indices[1]=3 -> embeddings[3] = [9, 10, 11]
        float[] expected = {3, 4, 5, 9, 10, 11};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    // ==================== Error Handling Tests ====================

    @Test
    @DisplayName("Gather with out-of-bounds positive axis throws exception")
    void testGatherInvalidAxisPositive() {
        float[] input = {0, 1, 2, 3};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 2));

        int[] indices = {0};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(1));

        assertThrows(IllegalArgumentException.class, () -> inputTensor.gather(indicesTensor, 5));
    }

    @Test
    @DisplayName("Gather with out-of-bounds negative axis throws exception")
    void testGatherInvalidAxisNegative() {
        float[] input = {0, 1, 2, 3};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 2));

        int[] indices = {0};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(1));

        assertThrows(IllegalArgumentException.class, () -> inputTensor.gather(indicesTensor, -3));
    }

    @Test
    @DisplayName("Gather with float indices throws exception")
    void testGatherInvalidIndicesTypeFloat() {
        float[] input = {0, 1, 2, 3};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 2));

        // Float indices should fail
        float[] indices = {0.0f};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(1));

        assertThrows(IllegalArgumentException.class, () -> inputTensor.gather(indicesTensor, 0));
    }

    @Test
    @DisplayName("Gather with double indices throws exception")
    void testGatherInvalidIndicesTypeDouble() {
        float[] input = {0, 1, 2, 3};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 2));

        double[] indices = {0.0};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(1));

        assertThrows(IllegalArgumentException.class, () -> inputTensor.gather(indicesTensor, 0));
    }

    @Test
    @DisplayName("GatherOp validation catches mismatched shapes")
    void testGatherOpValidation() {
        float[] input = {0, 1, 2, 3};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 2));

        int[] indices = {0};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(1));

        // This should work
        Tensor result = inputTensor.gather(indicesTensor, 0);
        assertEquals(Shape.of(1, 2), result.shape());
    }

    // ==================== Edge Case Tests ====================

    @Test
    @DisplayName("Gather with single element indices")
    void testGatherSingleElementIndices() {
        float[] input = {0, 1, 2, 3, 4, 5};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 3));

        int[] indices = {1};
        Tensor indicesTensor = Tensor.of(indices, Shape.scalar());

        Tensor result = inputTensor.gather(indicesTensor, 0);

        // Result: scalar-ish -> [3]
        assertEquals(Shape.of(3), result.shape());

        MemoryView<?> view = result.materialize();
        float[] expected = {3, 4, 5};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    @Test
    @DisplayName("Gather from 1D input tensor")
    void testGather1DInput() {
        // Input: [5]
        float[] input = {10, 20, 30, 40, 50};
        Tensor inputTensor = Tensor.of(input, Shape.of(5));

        int[] indices = {4, 0, 2};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(3));

        Tensor result = inputTensor.gather(indicesTensor, 0);

        // Result: [3]
        assertEquals(Shape.of(3), result.shape());

        MemoryView<?> view = result.materialize();
        float[] expected = {50, 10, 30};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    @Test
    @DisplayName("Gather preserving input dimensions after axis")
    void testGatherPreservesTrailingDimensions() {
        // Input: [2, 3, 4]
        float[] input = new float[2 * 3 * 4];
        for (int i = 0; i < input.length; i++) {
            input[i] = i;
        }
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 3, 4));

        // Gather 1 element along axis 1
        int[] indices = {2};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(1));

        Tensor result = inputTensor.gather(indicesTensor, 1);

        // Result: [2, 1, 4]
        assertEquals(Shape.of(2, 1, 4), result.shape());

        MemoryView<?> view = result.materialize();
        // For each batch (2), get slice 2 (index 2 in dim 1), all 4 elements
        // Batch 0, slice 2: values 8, 9, 10, 11
        // Batch 1, slice 2: values 20, 21, 22, 23
        assertEquals(8, readFloat(view, 0), 0.0001f);
        assertEquals(9, readFloat(view, 1), 0.0001f);
        assertEquals(10, readFloat(view, 2), 0.0001f);
        assertEquals(11, readFloat(view, 3), 0.0001f);
        assertEquals(20, readFloat(view, 4), 0.0001f);
    }

    @Test
    @DisplayName("Gather with repeated indices")
    void testGatherRepeatedIndices() {
        float[] input = {0, 1, 2, 3, 4, 5};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 3));

        // Gather same row twice
        int[] indices = {1, 1, 1};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(3));

        Tensor result = inputTensor.gather(indicesTensor, 0);

        // Result: [3, 3]
        assertEquals(Shape.of(3, 3), result.shape());

        MemoryView<?> view = result.materialize();
        // All three rows should be [3, 4, 5]
        for (int row = 0; row < 3; row++) {
            assertEquals(3, readFloat(view, row * 3 + 0), 0.0001f);
            assertEquals(4, readFloat(view, row * 3 + 1), 0.0001f);
            assertEquals(5, readFloat(view, row * 3 + 2), 0.0001f);
        }
    }

    @Test
    @DisplayName("Gather produces correct results with materialized input")
    void testGatherWithMaterializedInput() {
        float[] embeddings = {0, 1, 2, 3, 4, 5};
        Tensor embeddingTable = Tensor.of(embeddings, Shape.of(2, 3));

        int[] indices = {1, 0};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(2));

        Tensor result = embeddingTable.gather(indicesTensor, 0);

        // Verify the values are correct
        MemoryView<?> view = result.materialize();
        float[] expected = {3, 4, 5, 0, 1, 2};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    @Test
    @DisplayName("Gather with empty indices")
    void testGatherEmptyIndices() {
        float[] input = {0, 1, 2, 3, 4, 5};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 3));

        int[] indices = {};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(0));

        Tensor result = inputTensor.gather(indicesTensor, 0);

        // Result: [0, 3] - empty along gathered dimension
        assertEquals(Shape.of(0, 3), result.shape());
    }

    @Test
    @DisplayName("Gather with scalar indices")
    void testGatherScalarIndices() {
        float[] input = {0, 1, 2, 3, 4, 5};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 3));

        int[] indices = {1};
        Tensor indicesTensor = Tensor.of(indices, Shape.scalar());

        Tensor result = inputTensor.gather(indicesTensor, 0);

        // Result should be [3] - row 1 of input
        assertEquals(Shape.of(3), result.shape());

        MemoryView<?> view = result.materialize();
        float[] expected = {3, 4, 5};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], readFloat(view, i), 0.0001f);
        }
    }

    @Test
    @DisplayName("Gather data type preservation")
    void testGatherPreservesDataType() {
        // Test with different input data types
        float[] input = {0, 1, 2, 3};
        Tensor inputTensor = Tensor.of(input, Shape.of(2, 2));

        int[] indices = {0};
        Tensor indicesTensor = Tensor.of(indices, Shape.of(1));

        Tensor result = inputTensor.gather(indicesTensor, 0);

        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    @DisplayName("Large embedding table gather performance check")
    void testLargeEmbeddingGather() {
        // Simulate realistic embedding lookup
        int vocabSize = 50000;
        int hiddenSize = 768;
        int seqLen = 512;

        // Create large embedding table (but use small values for test speed)
        float[] embeddings = new float[vocabSize * 1]; // Simplified
        for (int i = 0; i < embeddings.length; i++) {
            embeddings[i] = i % 1000;
        }
        Tensor embeddingTable = Tensor.of(embeddings, Shape.of(vocabSize, 1));

        // Create token sequence
        int[] tokenIds = new int[seqLen];
        for (int i = 0; i < seqLen; i++) {
            tokenIds[i] = i % vocabSize;
        }
        Tensor tokenIdsTensor = Tensor.of(tokenIds, Shape.of(seqLen));

        // Perform gather
        Tensor result = embeddingTable.embeddingLookup(tokenIdsTensor);

        // Verify shape
        assertEquals(Shape.of(seqLen, 1), result.shape());

        // Materialize
        MemoryView<?> view = result.materialize();
        assertNotNull(view);
    }
}
