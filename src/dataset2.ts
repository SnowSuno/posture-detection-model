import posenet from "@tensorflow-models/posenet";

import { Canvas, loadImage } from "canvas";
import { Tensor3D } from "@tensorflow/tfjs-core";
import { PosenetInput } from "@tensorflow-models/posenet/dist/types";

const size = 400;

type PixelInput =
    | Tensor3D
    | ImageData
    | HTMLVideoElement
    | HTMLImageElement
    | HTMLCanvasElement
    | ImageBitmap;

const main = async () => {
    const posenetModel = await posenet.load();
    
    
    const loadData = async (path: string) => {
        const canvas = new Canvas(size, size);
        const ctx = canvas.getContext("2d");
        
        const image = await loadImage(path);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        const res = await detector.estimatePoses(imageData);
        
        
        res[0];
        
    };
    
    const getPose = async (image: PosenetInput) => {
        const res = posenetModel.baseModel.predict(image)
        // const res = await detector.estimatePoses(image);
        
        return res[0];
    };
    
};


