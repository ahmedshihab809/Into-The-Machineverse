let brain;

function setup() {
  createCanvas(640, 480);
  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true,
    layers:[
      {
        type: 'dense',
        units: 17,
        activation: 'relu',
      },
      {
        type: 'dense',
        units: 12,
        activation: 'relu',
      },
      {
        type: 'dense',
        units: 8,
        activation: 'relu',
      },
      {
        type: 'dense',
        units: 6,
        activation: 'relu',
      },
      {
        type: 'dense',
        activation: 'softmax',
      },
    ]
  }
  brain = ml5.neuralNetwork(options);
  brain.loadData('tf.json', dataReady);
}

function dataReady() {
  brain.normalizeData();
  brain.train({epochs: 50}, finished); 
}

function finished() {
  console.log('model trained');
  brain.save();
}