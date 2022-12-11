import { Tensor3D } from "@tensorflow/tfjs-core";

export interface Sample {
    data: Float32Array,
    label: number[],
}

export type PixelInput =
    | Tensor3D
    | ImageData
    | HTMLVideoElement
    | HTMLImageElement
    | HTMLCanvasElement
    | ImageBitmap;
