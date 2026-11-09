import * as tf from '@tensorflow/tfjs';
await tf.ready();
console.log(tf.getBackend(), tf.version.tfjs);
tf.tensor([1, 2, 3]).square().print();
