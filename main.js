import * as tf from '@tensorflow/tfjs';
import fs from 'fs';


import '@tensorflow/tfjs-backend-wasm';

import { performance } from 'perf_hooks';
const fitItirations = 50;
const predictIterations = 50;
let fit_Values = [];
let predict_Values = [];

// setWasmPaths('./tfjs-backend-wasm-simd.js')
tf.setBackend('wasm').then(() => {
    main();
  });



 const model = tf.sequential();



async function main() {
    benchmarkIterationFit();
    console.log("Fit iterations...");
    for (let i = 0; i < fitItirations; i++) {
        const itStart = performance.now();
        await benchmarkIterationPredict();
        const itEnd = performance.now();
        fit_Values.push(itEnd - itStart);
    }

    console.log("Predict iterations...");
    let sum = 0;
    for (let i = 0; i < predictIterations; i++) {
        const itStart = performance.now();
        sum += await benchmarkIterationPredict();
        const itEnd = performance.now();
        predict_Values.push(itEnd - itStart);
    }
    const resultObj = { times: predict_Values };
    const outputJs = `export const benchmarkResults = ${JSON.stringify(resultObj, null, 2)};\n`;
    const resultObj2 = { times: fit_Values };
    const outputJs2 = `export const benchmarkResults = ${JSON.stringify(resultObj2, null, 2)};\n`;
 fs.writeFileSync('SIMD_proposal_fit.js', outputJs, 'utf-8');
 fs.writeFileSync('SIMD_proposal_predict.js', outputJs2, 'utf-8');
 console.log("Benchmark results saved to benchmark-results.js");
}

async function benchmarkIterationFit(){
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

    await model.fit(xs, ys, {epochs: 250});
}
 // Should be close to 39

 async function benchmarkIterationPredict(){;
    var x = model.predict(tf.tensor2d([20], [1, 1])).dataSync();
    return x[0];
}

 