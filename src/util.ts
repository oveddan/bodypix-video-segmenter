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
