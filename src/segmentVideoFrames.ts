import * as bodypix from '@tensorflow-models/body-pix';
import * as tf from '@tensorflow/tfjs-core'
import * as program from 'commander';
import * as config from './config';

import {Frame} from './types';
import {getFramesOfVideo, loadImage, saveImageToFile} from './videoUtils';

interface Program {
  video: string, internalResolution: number
}

const parseArgs = (): Program => {
  program.requiredOption('-v', '--video <video>', 'the name of the video')
      .requiredOption(
          '-i', '--internalResolution <number>', 'the internal resolution');

  program.parse(process.argv);

  // console.log('result program', program);

  const video = program.args[0] as string;
  const internalResolution = +program.args[1] as number;

  return {video, internalResolution};
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

    console.log('shapes', fullImage.shape, maskRgb.shape, input.shape);

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
  const {video, internalResolution} = parseArgs();

  // console.log('da video', video, internalResolution);

  console.log('loading bodypix...')
  const net = await bodypix.load(
      {architecture: 'ResNet50', quantBytes: 1, outputStride: 16});
  // const net = await bodypix.load(
  //     {architecture: 'MobileNetV1', quantBytes: 1, outputStride: 16});

  const frames = await getFramesOfVideo(video);

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    console.log('estimating frame ', frame.fileName);
    await segmentFrameAndSaveResult(net, frame, internalResolution, video);
  }
}

main();
