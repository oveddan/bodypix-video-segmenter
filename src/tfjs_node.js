const config = require('./config');

let tfjsnode;

if (config.GPU === 'True') {
  tfjsnode = require('@tensorflow/tfjs-node-gpu');
} else {
  tfjsnode = require('@tensorflow/tfjs-node');
}

module.exports = tfjsnode;
