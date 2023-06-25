let video;
let poseNet;
let pose;
let skeleton;
let confidence;
let brain;
let poseLabel = "y";
let tempPose;
let jackCounter = 0;
let squatCounter = 0;
function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);

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
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  brain.load(modelInfo, brainLoaded);
}

function brainLoaded() {
  console.log('pose classification ready!');
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) {
  
  if (results[0].confidence > 0.95) {
    tempPose = poseLabel;
    poseLabel = results[0].label;
  }
  console.log(results[0].confidence);
  confidence = results[0].confidence;
  confidence = confidence.toFixed(5);
  countJack();
  countSquat();
  classifyPose();
}

function countJack(){
  if(tempPose=='a' && poseLabel=='b'){
    jackCounter+=1;
  }
}
function countSquat(){
  if(tempPose=='c' && poseLabel=='d'){
    squatCounter+=1;
  }
}


function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}


function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }
  }
  pop();

  fill(255, 0, 255);
  noStroke();
  textSize(30);
  textAlign(CENTER, CENTER);
  text('Squat Counter:' + squatCounter, width / 2-210, height / 2+200);
  fill(0, 255, 255);
  noStroke();
  textSize(30);
  textAlign(CENTER, CENTER);
  text('Jack Counter:' + jackCounter, width / 2-216, height / 2+150);
  fill(0, 0, 0);
  fill(0, 0, 0);
  noStroke();
  textSize(40);
  textAlign(CENTER, CENTER);
  text('Confidence:' + confidence, width / 2+140, height / 2+200);
  // fill(255, 0, 255);
  // noStroke();
  // textSize(30);
  // textAlign(CENTER, CENTER);
  // text(poseLabel, width / 2, height / 2);

}