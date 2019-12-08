import * as dotenv from 'dotenv'
import {join} from 'path'

dotenv.config()

export const SOURCE_MEDIA_FOLDER = process.env.SOURCE_MEDIA_FOLDER;
export const DESTINATION_FOLDER = process.env.DESTINATION_FOLDER;
export const GPU = process.env.GPU;

export const videoFramesFolder = (videoName: string) =>
    join(SOURCE_MEDIA_FOLDER, videoName, 'frames');

export const destinationFrame = (videoName: string, frameName: string) =>
    join(DESTINATION_FOLDER, videoName, 'segmentations', frameName);