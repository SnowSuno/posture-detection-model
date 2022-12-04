import { Keypoint } from "@tensorflow-models/pose-detection";

export const encodeKeypoints = (keypoints: Keypoint[]): Float32Array => {
    return new Float32Array(keypoints
        .map(keypoint => [keypoint.x, keypoint.y])
        .flat(),
    );
};
