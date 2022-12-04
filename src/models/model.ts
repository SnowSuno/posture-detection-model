import * as tf from "@tensorflow/tfjs-node";
import { loadDatasets } from "../dataset";

const main = async () => {
    const examples: Float32Array[][] = [
        [],
        [],
    ];
    
    const params = {
        denseUnits: 30,
        learningRate: 0.0001,
        batchSize: 10,
        epochs: 50,
    };
    
    const numClasses = examples.length;
    const inputSize = examples[0][0].length;
    
    const varianceScaling = tf.initializers.varianceScaling({});
    
    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [inputSize],
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
    
    const { trainDataset, validationDataset } = await loadDatasets();
    
    const trainData = trainDataset.batch(params.batchSize);
    const validationData = validationDataset.batch(params.batchSize);
    

    await model.fitDataset(trainData, {
        epochs: params.epochs,
        validationData,
    });
    
    optimizer.dispose();
    
    return model;
    
};

main()
