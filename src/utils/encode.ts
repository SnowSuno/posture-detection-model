import { Keypoint } from "@tensorflow-models/pose-detection";

export const flatKeypointsToArray = (keypoints: Keypoint[]): Float32Array => {
    return new Float32Array(keypoints
        .map(keypoint => [keypoint.x, keypoint.y])
        .flat(),
    );
};

export const flatOneHot = (label: number, size: number): number[] => {
    const labelOneHot = new Array(size).fill(0);
    labelOneHot[label] = 1;
    return labelOneHot;
};
