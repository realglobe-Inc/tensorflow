package org.tensorflow.demo.poseestimation;

import org.tensorflow.Shape;

public class Mat3F {
    private final float[] data;
    private final int rows;
    private final int cols;
    private final int depth;

    public Mat3F(float[] data, int rows, int cols, int depth) {
        this.data = data;
        this.rows = rows;
        this.cols = cols;
        this.depth = depth;
    }

    public Mat3F(float[] data, Shape s) {
        this(data, (int) s.size(1), (int) s.size(2), (int) s.size(3));
        assert (s.numDimensions() >= 3);
    }

    public float get(int row, int col, int z) {
        // LOGGER.d("%d %d %d = %d",rows, cols, depth, row * (cols * depth) + col * depth + z);
        return data[row * (cols * depth) + col * depth + z];
    }
}
