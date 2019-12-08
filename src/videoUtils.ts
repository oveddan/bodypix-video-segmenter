import * as tf from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import {join} from 'path';
import * as pngjs from 'pngjs';

import * as config from './config';
import {Frame} from './types';

const tfjs_node = require('./tfjs_node');

export const loadFileBlob = (path: string) => fs.promises.readFile(path);

export const imageToPng = (file: Buffer) =>
    tfjs_node.node.decodeImage(file) as tf.Tensor3D

export async function loadImage(path: string):
    Promise<tf.Tensor3D> {
      const file = await loadFileBlob(path);

      return imageToPng(file);
    }

export async function getFilesInFolder(path: string) {
  const files = await fs.promises.readdir(path);

  return files.sort((a, b) => {
    const numberA = +a.split('.')[0];
    const numberB = +b.split('.')[0];

    return numberA - numberB;
  });
}

// async function savePosesByFrameToFile(poses, path) {
//   await fs.promises.writeFile(path, JSON.stringify({poses: poses}));
// }

export const getFramesOfVideo = async(video: string):
    Promise<Frame[]> => {
      const videoFramesFolder = config.videoFramesFolder(video);

      const frameFiles = await getFilesInFolder(videoFramesFolder);

      return frameFiles.map(
          fileName =>
              ({fileName, fullPath: join(videoFramesFolder, fileName)}));
    }


export async function saveImageToFile(
    height: number, width: number, data: tf.backend_util.TypedArray,
    path: string) {
  const pngImage =
      new pngjs.PNG({colorType: 2, width, height, inputColorType: 2});

  // console.log('data', segmentation.data);
  pngImage.data = Buffer.from(data);

  return new Promise((resolve) => {
    pngImage.pack()
        .pipe(fs.createWriteStream(path))
        .on('close', () => resolve());
  });
}
