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
