import {exec} from 'child_process';
import * as mkdirpSource from 'mkdirp';

export const mkdirp = (dir: string) => {
  return new Promise((resolve, reject) => {
    mkdirpSource(dir, (err: Error) => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
};

export function chunk<T>(inputArray: T[], chunkSize: number): T[][] {
  return inputArray.reduce((resultArray, item, index) => {
    const chunkIndex = Math.floor(index / chunkSize);

    if (!resultArray[chunkIndex]) {
      resultArray[chunkIndex] = [];
    }

    resultArray[chunkIndex].push(item);

    return resultArray;
  }, []);
};

export const executeCommand = (command: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    const child = exec(command);
    child.stdout.pipe(process.stdout);
    child.stderr.pipe(process.stderr);
    child.on('exit', resolve);
    child.on('error', reject);
  });
};

export const getVideoFps = (videoPath: string): Promise<string> => {
  const command =
      `ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate ${
          videoPath}`;

  return new Promise((resolve, reject) => {
    const child = exec(command);
    child.stdout.pipe(process.stdout);
    child.stderr.pipe(process.stderr);

    let output = '';

    child.stdout.on('data', (chunk) => {
      output += (chunk as string);
    });

    child.on('close', (code: number) => {
      if (code === 0) {
        resolve(output.trim());
      } else {
        reject(`Command exited with code ${code}.`);
      }
    });

    child.on('error', reject);
  });
};
