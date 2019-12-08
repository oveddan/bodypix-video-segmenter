import * as tf from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import {join} from 'path';
import * as pngjs from 'pngjs';

import * as config from './config';
import {Frame} from './types';

const tfjs_node = require('./tfjs_node');

export async function loadImage(path: string): Promise<tf.Tensor3D> {
  const file = await fs.promises.readFile(path);

  const image = tfjs_node.node.decodeImage(file) as tf.Tensor3D;

  return image;
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


export async function saveImageToFile(image: tf.Tensor3D, path: string) {
  const data = await image.data();
  const pngImage = new pngjs.PNG({
    colorType: 2,
    width: image.shape[1],
    height: image.shape[0],
    inputColorType: 2
  });

  // console.log('data', segmentation.data);
  pngImage.data = Buffer.from(data);

  return new Promise((resolve) => {
    pngImage.pack()
        .pipe(fs.createWriteStream(path))
        .on('close', () => resolve());
  });
}
