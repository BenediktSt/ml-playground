const tf = require('@tensorflow/tfjs');

const NORM_MONTH = 12;
const NORM_WORKING_DAYS = 31;
const NORM_WORKED_HOURS = 250;

const importedData = require("./data.json").data;

const normalisedDataInput = [];
const normalisedDataOutput = [];

for (entry of importedData) {
    normalisedDataInput.push([entry.month / NORM_MONTH, entry.workingDays / NORM_WORKING_DAYS]);
    normalisedDataOutput.push([entry.workedHours / NORM_WORKED_HOURS]);
}

const train_xs = tf.tensor2d(normalisedDataInput);
const train_ys = tf.tensor2d(normalisedDataOutput);

const model = tf.sequential();
const hidden = tf.layers.dense({
    inputShape: [2],
    units: 4,
    activation: "sigmoid"
});

const output = tf.layers.dense({
    units: 1,
    activation: "sigmoid"
});

model.add(hidden);
model.add(output);

const optimizer = tf.train.adam(0.1);
model.compile({
    optimizer: optimizer,
    loss: "meanSquaredError"
});

tf.tidy(() => {
    console.log("Start training ...");
    console.log("Training Data:");
    train_xs.print();
    trainModel(500).then(result => {
        console.log("Final loss function: " + result.history.loss[0]);
        console.log("Prediction for working hours next month:");

        const newData = tf.tensor2d([[9 / NORM_MONTH, 16 / NORM_WORKING_DAYS]]);

        model.predict(newData).data().then(predictions => {
            console.log(predictions.map(entry => entry * NORM_WORKED_HOURS))
        });
    })
});


async function trainModel(iterations) {
    let result;
    for (i = 0; i < iterations; i++) {
        result = await model.fit(train_xs, train_ys, {
            shuffle: true,
            epochs: 10
        });
    }
    return result;
}