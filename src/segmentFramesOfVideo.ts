import * as bodypix from '@tensorflow-models/body-pix';
import * as tf from '@tensorflow/tfjs-core'
import * as data from '@tensorflow/tfjs-data';
import {basename, join} from 'path';

// import * as config from './config';
import {Frame} from './types';
import {mkdirp} from './util';
import {getFramesInFolder, imageToPng, loadFileBlob, saveImageToFile} from './videoUtils';

// interface Program {
//   video: string, internalResolution: number, batchSize: number
// }

// const DEFAULT_BATCH_SIZE = 10;

// const parseArgs = (): Program => {
//   program.requiredOption('-v', '--video <video>', 'the name of the video')
//       .requiredOption(
//           '-i', '--internalResolution <number>', 'the internal resolution')
//       .option('-b', '--batchSize <batchSize>', undefined,
//       DEFAULT_BATCH_SIZE);

//   program.parse(process.argv);

//   // console.log('result program', program);

//   const video = program.args[0] as string;
//   const internalResolution = +program.args[1] as number;
//   const batchSize = +program.args[2] as number;

//   return {video, internalResolution, batchSize};
// };

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

// const loadBatchFileBlobs =
//     async (frames: Frame[]) => {
//   const files = await Promise.all(frames.map(
//       async ({fullPath, fileName}) =>
//           ({fileName, file: await loadFileBlob(fullPath)})));

//   return files;
// }

interface SegmentationResult {
  width: number, height: number, data: tf.backend_util.TypedArray,
      fileName: string
}

function zeroFill(number: string, width: number) {
  width -= number.toString().length;
  if (width > 0) {
    return new Array(width + (/\./.test(number) ? 2 : 1)).join('0') + number;
  }
  return number + '';  // always return a string
}

const getFileName = (frameNumber: number) => {
  const zeroFilled = zeroFill(String(frameNumber), 9);

  return `${zeroFilled}.png`;
};

const segmentFrames = async(
    net: bodypix.BodyPix, frames: FramesBatch,
    internalResolution: number): Promise<SegmentationResult[]> => {
  // const frameLoadStartTime = new Date().getTime();

  // console.log(
  //     'time to load frames: ', new Date().getTime() - frameLoadStartTime);

  const inferenceStartTime = new Date().getTime();

  const resultsImages = tf.tidy(() => {
    const images = frames.image.unstack(0) as tf.Tensor3D[];
    // const frameNumbers = frames.frame.unstack(0) as tf.Scalar[];

    // console.log(
    //     'segmenting frames: ', frames.map(({path}) =>
    //     getFileName(path)));

    // const images: tf.Tensor3D[] = frames.map(({image}) =>
    // imageToPng(image));

    return images.map(
        (frame) =>
            segmentFrameAndCreateResultsImage(net, frame, internalResolution));
  });

  const frameNumbers = await frames.frame.data();
  const resultImagesData = await Promise.all(
      resultsImages.map(async (image, i) => ({
                          height: image.shape[0],
                          width: image.shape[1],
                          data: await image.data(),
                          fileName: getFileName(frameNumbers[i])
                        })));

  resultsImages.forEach((image) => {
    image.dispose();
  });

  console.log(
      'time to perform inference: ', new Date().getTime() - inferenceStartTime);

  console.log(
      'processed images',
      resultImagesData.map(({fileName}) => fileName).join(','));

  return resultImagesData;
  // return resultImagesData.map(
  //     (image, i) => ({...image, fileName: getFileName(image.name}));
};

const saveResults = async (
    segmentedFrames: SegmentationResult[], destinationFolder: string) => {
  const saveStartTime = new Date().getTime();
  const saveFramesPromises =
      segmentedFrames.map(({fileName, height, width, data}) => {
        const destinationFile = join(destinationFolder, fileName);

        return saveImageToFile(height, width, data, destinationFile);
      });

  await Promise.all(saveFramesPromises);

  console.log('time to save frames: ', new Date().getTime() - saveStartTime);
};

const segmentFramesAndSaveResult = async (
    net: bodypix.BodyPix, frames: FramesBatch, internalResolution: number,
    destinationFoldere: string) => {
  const segmentedFrames = await segmentFrames(net, frames, internalResolution);

  saveResults(segmentedFrames, destinationFoldere);
};

interface FramesBatch {
  frame: tf.Tensor1D;
  image: tf.Tensor4D;
}

const pathToFrameNumber = (path: string): number => {
  const fileName = basename(path);

  const result = parseInt(fileName.split(',')[0]);

  return result;
};

const createFramesDataSet = (frames: Frame[], batchSize: number) => {
  const filePaths = frames.map(({fullPath}) => fullPath);

  const dataset = data.array(filePaths).mapAsync(
      async (path) => ({
        frame: pathToFrameNumber(path),
        image: imageToPng(await loadFileBlob(path))
      }));

  const batched = dataset.batch(batchSize) as
      data.Dataset<{frame: tf.Tensor1D, image: tf.Tensor4D}>;

  return batched.prefetch(2);
};


export const segmentFramesOfVideo =
    async (
        sourceFolder: string, resultsFolder: string, internalResolution: number,
        batchSize: number) => {
  // console.log('da video', video, internalResolution);
  console.log('batch size', batchSize);
  console.log('loading bodypix...')
  const net = await bodypix.load(
      {architecture: 'ResNet50', quantBytes: 4, outputStride: 16});
  // const net = await bodypix.load(
  //     {architecture: 'MobileNetV1', quantBytes: 1, outputStride: 16});

  const startTime = new Date().getTime();

  const frames = await getFramesInFolder(sourceFolder);

  await mkdirp(resultsFolder);

  // const frameBatches = chunk(frames, batchSize);

  // let prefetchFrames: (fileNames: string[]) => Promise<tf.Tensor3D[]>;

  const dataset = createFramesDataSet(frames, batchSize);

  await dataset.forEachAsync(async batch => {
    // console.log('batch', batch);
    await segmentFramesAndSaveResult(
        net, batch, internalResolution, resultsFolder);
    // console.log('got batch', batch);

    batch.image.dispose();
    batch.frame.dispose();
  });
  // for (let i = 0; i < frameBatches.length; i++) {
  //   const frameBatch = frameBatches[i];

  //   const startTime = new Date().getTime();
  //   console.log(
  //       'estimating batch frames: ',
  //       frameBatch.map(({fileName}) => fileName).join(','));

  //   await segmentFramesAndSaveResult(
  //       net, frameBatch, internalResolution, resultsFolder);

  //   console.log('batch completed in :', new Date().getTime() -
  //   startTime);
  // }

  console.log(
      `time to complete ${frames.length} frames: `,
      new Date().getTime() - startTime);
}

// const main = () => {
//   const {video, internalResolution, batchSize} = parseArgs();
//   segmentFramesOfVideo(video, internalResolution, batchSize);
// }
