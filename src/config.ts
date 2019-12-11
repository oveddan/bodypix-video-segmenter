import * as dotenv from 'dotenv'
import {join} from 'path'

dotenv.config()

export const SOURCE_MEDIA_FOLDER = process.env.SOURCE_MEDIA_FOLDER;
export const DESTINATION_FOLDER = process.env.DESTINATION_FOLDER;
export const LOCAL_PROCESSING_FOLDER = process.env.LOCAL_PROCESSING_FOLDER;
export const GPU = process.env.GPU;

export const sourceVideoPath = (video: string) =>
    join(SOURCE_MEDIA_FOLDER, video + '.mp4');

export const temporaryVideoPath = () =>
    join(LOCAL_PROCESSING_FOLDER, 'source.mp4');

export const temporaryResultsVideoPath = () =>
    join(LOCAL_PROCESSING_FOLDER, 'results.mp4');

export const videoFramesFolder = () => join(LOCAL_PROCESSING_FOLDER, 'frames');

export const resultFramesFolder = () =>
    join(LOCAL_PROCESSING_FOLDER, 'segmentations');

export const resultsVideoPath = (video: string) =>
    join(DESTINATION_FOLDER, video + 'results.mp4');

export const desitnationFrameFolder = (videoName: string) =>
    join(DESTINATION_FOLDER, videoName, 'segmentations');

export const destinationFrame = (videoName: string, frameName: string) =>
    join(desitnationFrameFolder(videoName), frameName);
