package org.tensorflow.demo.poseestimation;

import org.tensorflow.Shape;

public class Mat2I {
    private final int[] data;
    private final int rows;
    private final int cols;

    public Mat2I(int[] data, Shape shape) {
        this.data = data;
        this.rows = (int) shape.size(0);
        this.cols = (int) shape.size(1);
    }

    public int get(int row, int col) {
        return data[row * cols + col];
    }
}
