package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.os.Trace;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.poseestimation.EstimatedPose;
import org.tensorflow.demo.poseestimation.Mat2I;
import org.tensorflow.demo.poseestimation.Mat3F;

import java.util.ArrayList;
import java.util.List;

public class TensorFlowPoseEstimator {
    private static final Logger LOGGER = new Logger();

    private TensorFlowInferenceInterface inferenceInterface;
    private Shape offsetShape;
    private Shape heatmapShape;
    private static final String[] outputNames = {"offset_2", "heatmap", "heatmap_values"};
    private static final String inputName = "image";
    private Shape inputShape;
    private int[] inputBuf;
    private float[] inputFloatBuf;
    private int imageMean;
    private float imageStd;
    private float[] offsetBuf;
    private float[] heatmapBuf;
    private Shape heatvaluesShape;
    private int[] heatvaluesBuf;

    private static final String[] PART_NAMES = new String[]{
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    };
    private static final int NUM_KEYPOINTS = PART_NAMES.length;
    private static final int OUTPUT_STRIDE = 16;

    private void preprocessInputBuf() {
        for (int i = 0; i < inputBuf.length; i++) {
            int val = inputBuf[i];
            float r = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            float g = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            float b = (((val) & 0xFF) - imageMean) / imageStd;
            inputFloatBuf[i * 3 + 0] = r;
            inputFloatBuf[i * 3 + 1] = g;
            inputFloatBuf[i * 3 + 2] = b;
        }
    }

    private static int numElements(Shape s) {
        int result = 1;
        for (int i = 0; i < s.numDimensions(); i++) {
            result *= s.size(i);
        }
        return result;
    }


    class HeatmapValues extends Mat2I {
        HeatmapValues(int[] data, Shape shape) {
            super(data, shape);
        }

        public List<PointF> toOffsetPoints(List<PointF> offsetVectors, int outputStride) {
            List<PointF> result = new ArrayList<>();
            final int size = offsetVectors.size();
            for (int i = 0; i < size; i++) {
                final PointF v = offsetVectors.get(i);
                float y = get(i, 0) * outputStride + v.y;
                float x = get(i, 1) * outputStride + v.x;

                result.add(new PointF(x, y));
            }
            return result;
        }
    }

    public List<EstimatedPose> recognizeImage(Bitmap bitmap) {
        // LOGGER.d("%d %d", offsetBuf.length, heatmapBuf.length);

        bitmap.getPixels(inputBuf, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        preprocessInputBuf();

        inferenceInterface.feed(
                inputName, inputFloatBuf,
                inputShape.size(0), inputShape.size(1), inputShape.size(2), 3 // shape
        );

        Trace.beginSection("run");
        inferenceInterface.run(outputNames);
        Trace.endSection();

        inferenceInterface.fetch(outputNames[0], offsetBuf);
        inferenceInterface.fetch(outputNames[1], heatmapBuf);
        inferenceInterface.fetch(outputNames[2], heatvaluesBuf);

        // result of argmax2d (17, 2), 17 == NUM_KEYPOINTS
        HeatmapValues heatvalues = new HeatmapValues(heatvaluesBuf, heatvaluesShape);
        Mat3F offsetMat = new Mat3F(offsetBuf, offsetShape);
        Mat3F heatmapMat = new Mat3F(heatmapBuf, heatmapShape);
        List<PointF> offsetPoints = getOffsetPoints(heatvalues, offsetMat);
        updatePointsConfidence(heatmapMat, heatvalues);

        EstimatedPose pose = new EstimatedPose(offsetPoints, keypointsConfidence, PART_NAMES);
        LOGGER.d(pose.toString());

        ArrayList<EstimatedPose> result = new ArrayList<>();
        result.add(pose);
        return result;
    }

    private static float[] keypointsConfidence = new float[NUM_KEYPOINTS];
    private void updatePointsConfidence(Mat3F heatmapMat, HeatmapValues heatmapValues) {
        for (int i = 0; i < NUM_KEYPOINTS; i++) {
            int y = heatmapValues.get(i, 0);
            int x = heatmapValues.get(i, 1);
            LOGGER.d("updatePointsConfidence: %d %d %d", y, x, i);
            keypointsConfidence[i] = heatmapMat.get(y, x, i);
        }
    }

    private List<PointF> getOffsetPoints(HeatmapValues heatmapValues, Mat3F offsetMat) {
        // getOffsetVecotrs
        ArrayList<PointF> offsetVectors = new ArrayList<>();
        for (int i = 0; i < NUM_KEYPOINTS; i++) {
            int heatmapY = heatmapValues.get(i, 0);
            int heatmapX = heatmapValues.get(i, 1);

            float y = offsetMat.get(heatmapY, heatmapX, i);
            float x = offsetMat.get(heatmapY, heatmapX, i + NUM_KEYPOINTS);
            offsetVectors.add(new PointF(x, y));
        }


        return heatmapValues.toOffsetPoints(offsetVectors, OUTPUT_STRIDE);
    }

    private void logOutputShape(Operation op) {
        for (int i = 0; i < op.numOutputs(); i++) {
            Output<Object> o = op.output(i);
            LOGGER.d("%s %d shape: %s, %s", op.name(), i, o.dataType(), o.shape().toString());
        }
    }

    public static TensorFlowPoseEstimator create(
            final AssetManager assetManager,
            final String modelFilename,
            final int imageMean,
            final float imageStd) {

        final TensorFlowPoseEstimator estimator = new TensorFlowPoseEstimator();

        TensorFlowInferenceInterface ii = new TensorFlowInferenceInterface(assetManager, modelFilename);
        estimator.inferenceInterface = ii;

        final Graph g = ii.graph();

        final Operation inputOp = g.operation("image");
        final Operation offsetsOp = g.operation("offset_2");
        final Operation heatmapOp = g.operation("heatmap");
        final Operation heatvaluesOp = g.operation("heatmap_values");

        estimator.logOutputShape(offsetsOp);
        estimator.logOutputShape(heatmapOp);
        estimator.logOutputShape(heatvaluesOp);

        estimator.inputShape = inputOp.output(0).shape();
        estimator.offsetShape = offsetsOp.output(0).shape();
        estimator.heatmapShape = heatmapOp.output(0).shape();
        estimator.heatvaluesShape = heatvaluesOp.output(0).shape();
        estimator.inputBuf = new int[(int) (estimator.inputShape.size(1) * estimator.inputShape.size(1))];
        estimator.inputFloatBuf = new float[estimator.inputBuf.length * 3]; // RGB
        estimator.offsetBuf = new float[numElements(estimator.offsetShape)];
        estimator.heatmapBuf = new float[numElements(estimator.heatmapShape)];
        estimator.heatvaluesBuf = new int[numElements(estimator.heatvaluesShape)];
        estimator.imageMean = imageMean;
        estimator.imageStd = imageStd;
        return estimator;
    }

}
