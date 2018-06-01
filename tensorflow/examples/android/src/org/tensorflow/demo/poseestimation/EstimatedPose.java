package org.tensorflow.demo.poseestimation;

import android.graphics.PointF;

import java.util.ArrayList;
import java.util.List;

public class EstimatedPose {
    private List<Keypoint> keypoints;
    private float totalScore;
    public float getTotalScore() { return totalScore; }
    public float getAverageScore() { return totalScore / keypoints.size(); }

    public EstimatedPose(List<Keypoint> keypoints, float totalScore) {
        this.keypoints = keypoints;
        this.totalScore = totalScore;
    }
    
    public EstimatedPose(List<PointF> offsetPoints, float[] keypointsConfidence, String[] partNames) {
        assert keypointsConfidence.length == offsetPoints.size() && keypointsConfidence.length == partNames.length;

        totalScore = (float) 0.0;
        keypoints = new ArrayList<>();
        for (int i = 0; i < keypointsConfidence.length; i++) {
            float score = keypointsConfidence[i];
            keypoints.add(new Keypoint(offsetPoints.get(i), partNames[i], score));
        }
    }

    public List<Keypoint> getKeypoints() { return keypoints; }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        for (Keypoint keypoint : keypoints) {
            builder.append("\"").append(keypoint.toString()).append("\", ");
        }

        if (builder.length() > 2) {
            builder.setLength(builder.length() - 2);
        }

        builder.append("]");
        return builder.toString();
    }

    public EstimatedPose scale(float width, float height) {
        List<Keypoint> scaled = new ArrayList<>();
        for (int i = 0; i < keypoints.size(); i++) {
            scaled.add(keypoints.get(i).scale(width, height));
        }
        return new EstimatedPose(scaled, totalScore);
    }

    public class Keypoint extends PointF {
        public final String part;
        public final float score;

        Keypoint(PointF point, String part, float score) {
            super(point.x, point.y);
            this.part = part;
            this.score = score;
        }

        public String toString() {
            return part + " : " + super.toString() + " : " + String.valueOf(score);
        }

        public Keypoint scale(float width, float height) {
            return new Keypoint(new PointF(this.x * width, this.y * height), part, score);
        }
    }
}
