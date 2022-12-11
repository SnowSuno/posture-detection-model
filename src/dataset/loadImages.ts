import * as poseDetection from "@tensorflow-models/pose-detection";
import fs from "fs";
import * as tf from "@tensorflow/tfjs-node";
import globby from "globby";
import { flatKeypointsToArray, flatOneHot } from "../utils";
import { PixelInput, Sample } from "../utils/types";
import { PoseDetector } from "@tensorflow-models/pose-detection";

const loadImageData = async (path: string) => {
    const jpeg = fs.readFileSync(path);
    
    return tf.node.decodeJpeg(jpeg);
};
const getPose = async (detector: PoseDetector, image: PixelInput) => {
    const res = await detector.estimatePoses(image);
    
    if (!res[0]) console.log("no pose detected");
    
    return res[0];
};

export const loadDatasets = async (datasetPath: string) => {
    const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
            modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
            enableTracking: true,
            trackerType: poseDetection.TrackerType.BoundingBox,
        },
    );
    
    // const datasetPath = "./dataset/test";
    
    const classes = [0, 1];
    const dataSetFiles = await Promise.all(
        classes.map(async (label) => {
            const fileNames = await globby(`${datasetPath}/${label}/*.jpg`);
            tf.util.shuffle(fileNames);
            return fileNames;
        }),
    );
    
    const sampleDataset = async (
        dataset: string[][],
        validation: boolean,
        validationFraction: number = 0.15,
    ) => {
        return dataset.map(dataClass => {
            const validationCount = Math.ceil(dataClass.length * validationFraction);
            
            return validation
                ? dataClass.slice(0, validationCount)
                : dataClass.slice(validationCount);
        });
    };
    
    const trainDatasetFiles = await sampleDataset(dataSetFiles, false);
    const validationDatasetFlies = await sampleDataset(dataSetFiles, true);
    
    const getDataset = async (filesNames: string[][]): Promise<Sample[]> => {
        const samples = await Promise.all(filesNames.map((classFiles, classIndex) =>
            Promise.all(classFiles.map(async (fileName) => {
                console.log("processing", fileName);
                const imageData = await loadImageData(fileName);
                const pose = await getPose(detector, imageData);
                
                if (!pose) return null;
                
                const size = 224;
                if (imageData.size !== 3 * size * size) throw new Error(`Unexpected image size: ${imageData.size}`);
                const normalizedKeypoints = poseDetection.calculators.keypointsToNormalizedKeypoints(
                    pose.keypoints,
                    { height: size, width: size },
                );
                
                return {
                    data: flatKeypointsToArray(normalizedKeypoints),
                    label: flatOneHot(classIndex, classes.length),
                };
            })),
        ));
        
        return samples.flat().filter(Boolean) as Sample[];
    };
    
    const trainDataset = await getDataset(trainDatasetFiles);
    const validationDataset = await getDataset(validationDatasetFlies);
    
    const trainX = tf.data.array(trainDataset.map(sample => sample.data));
    const validationX = tf.data.array(validationDataset.map(sample => sample.data));
    const trainY = tf.data.array(trainDataset.map(sample => sample.label));
    const validationY = tf.data.array(validationDataset.map(sample => sample.label));
    
    return {
        trainDataset: tf.data.zip({ xs: trainX, ys: trainY }),
        validationDataset: tf.data.zip({ xs: validationX, ys: validationY }),
    };
};
