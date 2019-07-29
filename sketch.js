const desiredDigits = [0, 1], trainsPerFrame = 100, canvasWidth = 200, canvasHeight = 200;

let mnist, brain, trainedBrain, isLooping = true, userDigit, canvasContainer, trainedTimes = 0, trainedCorrect = 0;
let trainImages = [], trainLabels = [], canTrain = false, hasUserData = false;
let trainSubsetImages = [], trainSubsetLabels = [];

function createNeuralNetwork() {
    const nn = new NeuralNetwork(784, desiredDigits.length);
    nn.addHiddenLayer(16);
    nn.addHiddenLayer(16);
    return nn;
}

function setup() {
    canvasContainer = document.getElementsByClassName('container')[0];
    createCanvas(canvasWidth, canvasHeight).parent('canvas');
    userDigit = createGraphics(200, 200);

    loadMNIST((data) => {
        mnist = data;

        for (let i = 0; i < mnist.train_images.length; i++) {
            if (desiredDigits.includes(mnist.train_labels[i])) {
                trainImages.push(mnist.train_images[i]);
                trainLabels.push(mnist.train_labels[i]);
            }
        }

        for (let i = 0; i < trainsPerFrame; i++) {
            const idx = Math.ceil(random(trainImages.length - 1));
            trainSubsetImages.push(trainImages[idx]);
            trainSubsetLabels.push(trainLabels[idx]);
        }

        brain = createNeuralNetwork();

        select('#application').style('visibility', 'visible');
        select('#loading').style('display', 'none');
    });
}

function draw() {
    background(0);

    if (canTrain && mnist) {
        // console.log('trained:', ++trainedTimes);
        brain.spidersColony.train();
    }

    guessUserDigit();

    image(userDigit, 0, 0);
    if (mouseIsPressed && 0 <= mouseX && mouseX <= canvasWidth && 0 <= mouseY && mouseY <= canvasHeight) {
        hasUserData = true;
        userDigit.stroke(255);
        userDigit.strokeWeight(select('#brushSize').value());
        userDigit.line(mouseX, mouseY, pmouseX, pmouseY);
        select('body').elt.classList.add('no-scroll');
    } else {
        select('body').elt.classList.remove('no-scroll');
    }
}

function getAnswer(outputs) {
    let index = 0, max = outputs[0];
    for (let i = 1; i < outputs.length; i++) {
        if (outputs[i] > max) {
            max = outputs[i];
            index = i;
        }
    }
    return desiredDigits[index];
}

function guessUserDigit() {
    if (hasUserData && brain && brain.spidersColony) {
        let inputs = [];
        let img = userDigit.get();
        img.resize(28, 28);
        img.loadPixels();
        for (let i = 0; i < 784; i++) {
            inputs[i] = img.pixels[i * 4];
        }

        let guess = getAnswer(brain.predict(inputs));

        canvasContainer.setAttribute('data-guess', 'I guess it is a...\n' + guess);
        canvasContainer.classList.add('drawing');
    }
}

function resetDrawing() {
    hasUserData = false;
    canvasContainer.classList.remove('drawing');
    userDigit.background(0);
}

function toggleTraining() {
    canTrain = !canTrain;

    const btnTraining = select('#btnTraining');
    // const checkboxTrained = select('#useTrainedNN');
    let action;
    if (canTrain) {
        canvasContainer.classList.add('training');
        action = 'Stop';
        btnTraining.addClass('stop');
        // checkboxTrained.attribute('disabled', '');
    } else {
        canvasContainer.classList.remove('training');
        action = 'Start';
        btnTraining.removeClass('stop');
        // checkboxTrained.removeAttribute('disabled');
    }

    btnTraining.html(action + ' training');
}
