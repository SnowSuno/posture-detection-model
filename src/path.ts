import * as path from "path";


export const DATASET_PATH = path.join(__dirname, "..", "dataset");
export const MODEL_PATH = path.join(__dirname, "..", "model");

export const datasetPath = (name: string) => path.join(DATASET_PATH, name);
export const modelPath = (name: string) => path.join(MODEL_PATH, name);
