const tf = require('@tensorflow/tfjs');
const model = tf.sequential();
let hidden = tf.layers.dense({
    inputShape: [2],
    units: 2,
    activation: "sigmoid"
});

let output = tf.layers.dense({
    units: 1,
    activation: "sigmoid"
});

model.add(hidden);
model.add(output);

const train_xs = tf.tensor2d([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]);
const train_ys = tf.tensor2d([
    [0],
    [1],
    [1],
    [0]
]);

const optimizer = tf.train.adam(0.1);
model.compile({
    optimizer: optimizer,
    loss: "meanSquaredError"
});

tf.tidy(() => {
    console.log("Start training ...");
    trainModel(200).then(result => {
        console.log("Final loss function: " + result.history.loss[0]);
        console.log("Prediction for: ");
        train_xs.print();
        console.log("Prediction:");
        model.predict(train_xs).print();
    })
});


async function trainModel(iterations) {
    let result;
    for (i = 0; i< iterations; i++){
        result = await model.fit(train_xs, train_ys, {
            shuffle: true,
            epochs: 10
        });
    }
    return result;
}