import { loadDatasets } from "./dataset";
import { datasetPath } from "./path";
import * as tf from "@tensorflow/tfjs-node";

const main = async () => {
    const { trainDataset, validationDataset } = await loadDatasets(
        datasetPath("dataset1"),
    );
    
    
    const params = {
        denseUnits: 64,
        learningRate: 0.01,
        batchSize: 16,
        epochs: 60,
    };
    
    const numClasses = 2;
    const inputSize = trainDataset.size;
    
    const varianceScaling = tf.initializers.varianceScaling({});
    
    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                // inputShape: [10, 34],
                inputDim: 34,
                units: params.denseUnits,
                activation: "relu",
                kernelInitializer: varianceScaling,
                useBias: true,
            }),
            tf.layers.dropout({ rate: 0.05 }),
            tf.layers.dense({
                units: numClasses,
                kernelInitializer: varianceScaling,
                useBias: true,
                activation: "softmax",
            }),
        ],
    });
    
    const optimizer = tf.train.rmsprop(params.learningRate);
    
    model.compile({
        optimizer: "Adam",
        loss: "binaryCrossentropy",
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

main().catch(console.error);
