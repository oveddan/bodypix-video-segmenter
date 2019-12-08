import * as bodypix from '@tensorflow-models/body-pix';
import * as tf from '@tensorflow/tfjs-core'
import * as program from 'commander';

import * as config from './config';
import {Frame} from './types';
import {chunk, mkdirp} from './util';
import {getFramesOfVideo, loadImage, saveImageToFile} from './videoUtils';

interface Program {
  video: string, internalResolution: number, batchSize: number
}

const DEFAULT_BATCH_SIZE = 10;

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

const segmentFrameAndCreateResultsImage = async(
    net: bodypix.BodyPix, frame: Frame,
    internalResolution: number): Promise<tf.Tensor3D> => {
  const input = await loadImage(frame.fullPath);

  const resultImage = tf.tidy(() => {
    const {segmentation, partSegmentation} =
        net.segmentPersonPartsActivation(input, internalResolution);

    const segmentationToInt = segmentation.mul(255).round().toInt();

    const maskChannels = [
      segmentationToInt, partSegmentation, tf.zeros(segmentation.shape, 'int32')
    ].map(x => x.expandDims(2) as tf.Tensor3D);

    const maskRgb = tf.concat3d(maskChannels, 2);

    // console.log('ranks', input.shape, maskRgb.shape);

    const fullImage = tf.concat3d([input, maskRgb], 1);

    // console.log('shapes', fullImage.shape, maskRgb.shape, input.shape);

    return fullImage;
  });

  input.dispose();

  return resultImage;
};

const segmentFrameAndSaveResult =
    async (
        net: bodypix.BodyPix, frame: Frame, internalResolution: number,
        video: string) => {
  const segmentationResultImage =
      await segmentFrameAndCreateResultsImage(net, frame, internalResolution);

  const destinationFile = config.destinationFrame(video, frame.fileName);

  await saveImageToFile(segmentationResultImage, destinationFile);

  segmentationResultImage.dispose();
}

const main =
    async () => {
  const {video, internalResolution, batchSize} = parseArgs();

  // console.log('da video', video, internalResolution);

  console.log('loading bodypix...')
  const net = await bodypix.load(
      {architecture: 'ResNet50', quantBytes: 4, outputStride: 16});
  // const net = await bodypix.load(
  //     {architecture: 'MobileNetV1', quantBytes: 1, outputStride: 16});

  const frames = await getFramesOfVideo(video);

  await mkdirp(config.desitnationFrameFolder(video));

  const frameBatches = chunk(frames, batchSize);

  for (let i = 0; i < frameBatches.length; i++) {
    const frameBatch = frameBatches[i];

    const startTime = new Date().getTime();
    console.log(
        'estimating batch frames: ', frameBatch.map(({fileName}) => fileName));

    const segmentAndFramePromises = frameBatch.map(
        frame =>
            segmentFrameAndSaveResult(net, frame, internalResolution, video));

    await Promise.all(segmentAndFramePromises);

    console.log('batch completed in :', new Date().getTime() - startTime);
  }
}

main();
