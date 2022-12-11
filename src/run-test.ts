import * as fs from "fs";

import * as tf from "@tensorflow/tfjs-node";
import * as poseDetection from "@tensorflow-models/pose-detection";
import { Tensor3D } from "@tensorflow/tfjs-core";
import { flatKeypointsToArray } from "./utils";
// import { globby } from "globby";
import globby from "globby";

const size = 400;

type PixelInput =
    | Tensor3D
    | ImageData
    | HTMLVideoElement
    | HTMLImageElement
    | HTMLCanvasElement
    | ImageBitmap;

interface Sample {
    data: Float32Array,
    label: number[],
}

const VALIDATION_FRACTION = 0.15;

const flatOneHot = (label: number, size: number): number[] => {
    const labelOneHot = new Array(size).fill(0);
    labelOneHot[label] = 1;
    return labelOneHot;
};

export const loadDatasets = async () => {
    const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
            modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
            enableTracking: true,
            trackerType: poseDetection.TrackerType.BoundingBox,
            
        },
    );
    // const detector = await poseDetection.createDetector(
    //     poseDetection.SupportedModels.BlazePose,
    //     {
    //         runtime: "tfjs",
    //         enableSmoothing: true,
    //         modelType: "full",
    //         enableSegmentation: true,
    //     },
    // );
    
    const loadImageData = async (path: string) => {
        // const canvas = new Canvas(size, size);
        // const ctx = canvas.getContext("2d");
        //
        // const image = await loadImage(path);
        // ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        //
        // return ctx.getImageData(0, 0, canvas.width, canvas.height);
        const jpeg = fs.readFileSync(path);
        
        return tf.node.decodeJpeg(jpeg);
    };
    
    const getPose = async (image: PixelInput) => {
        const res = await detector.estimatePoses(image);
        
        if (!res[0]) console.log("no pose detected");
        
        return res[0];
    };
    
    const datasetPath = "./dataset/test";
    
    const classes = [0, 1];
    const dataSetFiles = await Promise.all(
        classes.map(async (label) => {
            const fileNames = await globby(`${datasetPath}/${label}/*.jpg`);
            tf.util.shuffle(fileNames);
            return fileNames;
        }),
    );
    
    const sampleDataset = async (dataset: string[][], validation: boolean) => {
        return dataset.map(dataClass => {
            const validationCount = Math.ceil(dataClass.length * VALIDATION_FRACTION);
            
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
                const pose = await getPose(imageData);
                
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

const main = async () => {
    const { trainDataset, validationDataset } = await loadDatasets();
    // const examples: Float32Array[][] = [
    //     [],
    //     [],
    // ];
    
    const params = {
        denseUnits: 30,
        learningRate: 0.0001,
        batchSize: 10,
        epochs: 50,
    };
    
    const numClasses = 2;
    const inputSize = trainDataset.size;
    
    const varianceScaling = tf.initializers.varianceScaling({});
    
    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [10, 34],
                units: params.denseUnits,
                activation: "relu",
                kernelInitializer: varianceScaling,
                useBias: true,
            }),
            tf.layers.dropout({ rate: 0.5 }),
            tf.layers.dense({
                units: numClasses,
                kernelInitializer: varianceScaling,
                useBias: false,
                activation: "softmax",
            }),
        ],
    });
    
    const optimizer = tf.train.rmsprop(params.learningRate);
    
    model.compile({
        optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });
    
    
    const trainData = trainDataset.batch(params.batchSize);
    const validationData = validationDataset.batch(params.batchSize);
    
    await model.fitDataset(trainData, {
        epochs: params.epochs,
        validationData,
    });
    
    optimizer.dispose();
    
    return model;
};

console.log("Start")
main();

