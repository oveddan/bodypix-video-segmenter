import * as program from 'commander';
import * as fs from 'fs';

import * as config from './config';
import {segmentFramesOfVideo} from './segmentFramesOfVideo';
import {convertFramesToVideo, convertVideoToFrames} from './videoUtils';

const DEFAULT_BATCH_SIZE = 10;

interface Program {
  video: string, internalResolution: number, batchSize: number
}

const parseArgs = (): Program => {
  program.requiredOption('-v', '--video <video>', 'the name of the video')
      .requiredOption(
          '-i', '--internalResolution <number>', 'the internal resolution')
      .option('-b', '--batchSize <batchSize>', undefined, DEFAULT_BATCH_SIZE);

  program.parse(process.argv);

  // console.log('result program', program);

  const video = program.args[0] as string;
  const internalResolution = +program.args[1] as number;
  const batchSize = +program.args[2] as number;

  return {video, internalResolution, batchSize};
};

const segmentVideo =
    async (video: string, internalResolution: number, batchSize: number) => {
  const sourceVideoPath = config.sourceVideoPath(video);
  const temporaryVideoPath = config.temporaryVideoPath();
  console.log(`copying video from ${sourceVideoPath} to ${temporaryVideoPath}`);
  await fs.promises.copyFile(sourceVideoPath, temporaryVideoPath);

  const framesFolder = config.videoFramesFolder();
  console.log(`converting video at ${temporaryVideoPath} to frames in folder ${
      framesFolder}`);
  await convertVideoToFrames(temporaryVideoPath, framesFolder);

  const resultFramesFolder = config.resultFramesFolder();
  console.log(`segmenting frames in ${framesFolder} to destination ${
      resultFramesFolder}`);
  await segmentFramesOfVideo(
      framesFolder, resultFramesFolder, internalResolution, batchSize);

  const temporaryResultsVideoPath = config.temporaryResultsVideoPath();
  console.log(`converting frames in folder ${resultFramesFolder} to video at ${
      temporaryResultsVideoPath}`);
  await convertFramesToVideo(
      sourceVideoPath, resultFramesFolder, temporaryResultsVideoPath);

  const resultsVideoPath = config.resultsVideoPath(video);
  console.log(
      `copying video from ${temporaryResultsVideoPath} to ${resultsVideoPath}`);
  await fs.promises.copyFile(temporaryResultsVideoPath, resultsVideoPath);
}

const main =
    async () => {
  const {video, internalResolution, batchSize} = parseArgs();

  segmentVideo(video, internalResolution, batchSize);
}

main();
