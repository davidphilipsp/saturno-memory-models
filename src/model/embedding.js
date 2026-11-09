// Embedding com gradiente esparso. No backend cpu puro, o gradiente do tf.layers.embedding
// fica inviável com ~19 mil lexemas (medido na Fase A). Aqui: gather na ida, scatterND na volta.
import * as tf from '@tensorflow/tfjs';

export const embeddingLookup = tf.customGrad((table, idx, save) => {
  save([idx]);
  return {
    value: tf.gather(table, idx),
    gradFunc: (dy, [i]) => [tf.scatterND(i.expandDims(1), dy, table.shape), tf.zerosLike(i)],
  };
});
