import * as bodypix from '@tensorflow-models/body-pix';
import * as tf from '@tensorflow/tfjs-core'
import * as program from 'commander';

import * as config from './config';
import {Frame} from './types';
import {chunk, mkdirp} from './util';
import {getFramesOfVideo, imageToPng, loadFileBlob, saveImageToFile} from './videoUtils';

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

const segmentFrameAndCreateResultsImage =
    (net: bodypix.BodyPix, input: tf.Tensor3D,
     internalResolution: number): tf.Tensor3D => {
      return tf.tidy(() => {
        const {segmentation, partSegmentation} =
            net.segmentPersonPartsActivation(input, internalResolution);

        const segmentationToInt = segmentation.mul(255).round().toInt();

        const maskChannels = [
          segmentationToInt, partSegmentation,
          tf.zeros(segmentation.shape, 'int32')
        ].map(x => x.expandDims(2) as tf.Tensor3D);

        const maskRgb = tf.concat3d(maskChannels, 2);

        // console.log('ranks', input.shape, maskRgb.shape);

        return tf.concat3d([input, maskRgb], 1) as tf.Tensor3D;
      });
    };

// const loadAndSegmentFrameAndCreateResultsImage =
//     async(net: bodypix.BodyPix, frame: Frame, internalResolution: number):
//         Promise<tf.Tensor3D> => {
//           const input = await loadImage(frame.fullPath);

//           const resultImage =
//               segmentFrameAndCreateResultsImage(net, input,
//               internalResolution);

//           input.dispose();

//           return resultImage;
//         };

// const segmentFrameAndSaveResult =
//     async (
//         net: bodypix.BodyPix, frame: Frame, internalResolution: number,
//         video: string) => {
//   const segmentationResultImage =
//       await loadAndSegmentFrameAndCreateResultsImage(
//           net, frame, internalResolution);

//   const destinationFile = config.destinationFrame(video, frame.fileName);

//   await saveImageToFile(segmentationResultImage, destinationFile);

//   segmentationResultImage.dispose();
// }

const loadBatchFileBlobs =
    async (frames: Frame[]) => {
  const files = await Promise.all(frames.map(
      async ({fullPath, fileName}) =>
          ({fileName, file: await loadFileBlob(fullPath)})));

  return files;
}

const segmentFrames =
    async (
        net: bodypix.BodyPix, frames: Frame[], internalResolution: number) => {
  const frameLoadStartTime = new Date().getTime();
  const filesBatch = await loadBatchFileBlobs(frames);

  console.log(
      'time to load frames: ', new Date().getTime() - frameLoadStartTime);

  const inferenceStartTime = new Date().getTime();
  const resultsImages = tf.tidy(() => {
    const images: tf.Tensor3D[] = filesBatch.map(({file}) => imageToPng(file));

    return images.map(
        frame =>
            segmentFrameAndCreateResultsImage(net, frame, internalResolution));
  });

  const resultImagesData =
      await Promise.all(resultsImages.map(async image => ({
                                            height: image.shape[0],
                                            width: image.shape[1],
                                            data: await image.data()
                                          })));

  resultsImages.forEach(x => x.dispose());

  console.log(
      'time to perform inference: ', new Date().getTime() - inferenceStartTime);


  return resultImagesData.map(
      (image, i) => ({...image, fileName: filesBatch[i].fileName}));
}

const segmentFramesAndSaveResult =
    async (
        net: bodypix.BodyPix, frames: Frame[], internalResolution: number,
        video: string) => {
  const segmentedFrames = await segmentFrames(net, frames, internalResolution);

  const saveStartTime = new Date().getTime();
  const saveFramesPromises =
      segmentedFrames.map(({fileName, height, width, data}) => {
        const destinationFile = config.destinationFrame(video, fileName);

        return saveImageToFile(height, width, data, destinationFile);
      });

  await Promise.all(saveFramesPromises);

  console.log('time to save frames: ', new Date().getTime() - saveStartTime);
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

    await segmentFramesAndSaveResult(
        net, frameBatch, internalResolution, video);


    console.log('batch completed in :', new Date().getTime() - startTime);
  }
}

main();
