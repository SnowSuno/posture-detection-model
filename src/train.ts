import { loadDatasets } from "./dataset";
import { datasetPath } from "./path";
import * as tf from "@tensorflow/tfjs-node";

const main = async () => {
    const { trainDataset, validationDataset } = await loadDatasets(
        datasetPath("dataset1"),
    );
    
    
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

main().catch(console.error);
