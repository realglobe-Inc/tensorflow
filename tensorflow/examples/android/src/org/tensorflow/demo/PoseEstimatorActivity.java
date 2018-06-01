package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.ImageReader;
import android.os.SystemClock;
import android.util.Size;

import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.poseestimation.EstimatedPose;

import java.util.List;

public class PoseEstimatorActivity extends CameraActivity implements ImageReader.OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();
    private static final Size DESIRED_PREVIEW_SIZE = new Size(513, 513);
    // private static final int INPUT_SIZE = 337;
    private static final int INPUT_SIZE = 241;
    private static final boolean MAINTAIN_ASPECT = true;

    protected static final boolean SAVE_PREVIEW_BITMAP = false;

    private static final String MODEL_FILE = "file:///android_asset/posenet.pb";


    private Integer sensorOrientation;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private long lastProcessingTimeMs;
    private boolean computingDetection = false;
    private TensorFlowPoseEstimator estimator;
    private OverlayView poseOverlay;

    private List<EstimatedPose> lastPose;

    @Override
    protected void processImage() {
        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }

        if (poseOverlay != null) {
            poseOverlay.invalidate();
        }
        computingDetection = true;

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();
                        // final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
//                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//                        LOGGER.i("Detect: %s", results);
                        List<EstimatedPose> results = estimator.recognizeImage(croppedBitmap);
                        lastPose = results;
                        LOGGER.i("Estimate time: %d[ms]", SystemClock.uptimeMillis() - startTime);

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
//                        if (resultsView == null) {
//                            resultsView = (ResultsView) findViewById(R.id.results);
//                        }
//                        resultsView.setResults(results);
                        if (poseOverlay != null) {
                            poseOverlay.invalidate();
                        }
                        requestRender();
                        readyForNextImage();
                        computingDetection = false;
                    }
                });
    }

    @Override
    protected void onPreviewSizeChosen(Size size, int rotation) {
        estimator = TensorFlowPoseEstimator.create(getAssets(), MODEL_FILE, 128, 128);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                INPUT_SIZE, INPUT_SIZE,
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        poseOverlay = (OverlayView) findViewById(R.id.pose_overlay);
        poseOverlay.addCallback(new OverlayView.DrawCallback() {
            @Override
            public void drawCallback(Canvas canvas) {
                List<EstimatedPose> results = lastPose;
                if (lastPose == null || results.isEmpty()) {
                    return;
                }

                int width = canvas.getWidth();
                int height = canvas.getHeight();

                float wscale = (float) previewHeight / (float) INPUT_SIZE;
                float hscale = (float) previewWidth / (float) INPUT_SIZE;

                LOGGER.d("%d %d -> %f, %f", width, height, wscale, hscale);

                EstimatedPose pose = results.get(0).scale(wscale, hscale);
                List<EstimatedPose.Keypoint> keypoints = pose.getKeypoints();

                float[] floatPoints = new float[pose.getKeypoints().size() * 2];
                for (int i = 0; i < keypoints.size(); i += 2) {
                    EstimatedPose.Keypoint kp = pose.getKeypoints().get(i);
                    floatPoints[i * 2] = kp.x;
                    floatPoints[i * 2 + 1] = kp.y;
                }

                // cropToFrameTransform.mapPoints(floatPoints);
                Paint paint = new Paint();
                paint.setARGB(255, 0, 255, 255);
                paint.setStrokeWidth(10);
                //canvas.drawPoints(floatPoints, paint);
                for (int i = 0; i < keypoints.size(); i++) {
                    LOGGER.d(keypoints.get(i).part);
//                    float x = floatPoints[i * 2 + 0];// * ((float) width / previewWidth);
//                    float y = floatPoints[i * 2 + 1];// * ((float) height / previewHeight);
                    if (i < 1)
                        paint.setARGB(255, 0, 0, 0);
                    else if (i < 5)
                        paint.setARGB(255, 0, 255, 255);
                    else if (i < 7)
                        paint.setARGB(255, 255, 0, 0);
                    else
                        paint.setARGB(255, 255, 255, 255);

                    float x = keypoints.get(i).x;
                    float y = keypoints.get(i).y;
                    canvas.drawPoint(x, y, paint);
                }

                if (isDebug()) {
                    // draw debug
                }
            }
        });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_pose;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }
}
