const spidersQuantity = 25;

class NeuralNetwork {
    constructor(inputNodes, outputNodes) {
        this.layersNodes = [inputNodes, outputNodes];
        this.setLearningRate();
        this.setActivationFunction();

        this.spidersColony = new Colony(spidersQuantity, this, this.layersNodes);
    }

    bootstrapWeigths() {
        this.spidersColony.bootstrapColony(spidersQuantity, this.layersNodes);
    }

    setLearningRate(learningRate = 0.1) {
        this.learningRate = learningRate;
    }

    setActivationFunction(func = sigmoid) {
        this.activationFunction = func;
    }

    addHiddenLayer(hiddenNodes) {
        this.layersNodes = this.layersNodes.slice(0, this.layersNodes.length - 1)
            .concat(hiddenNodes)
            .concat(this.layersNodes.slice(-1));

        this.bootstrapWeigths();
    }

    predict(inputArray, spider = undefined) {
        if (spider === undefined) {
            spider = this.spidersColony.getBestSpider().value;
        }

        let outputs = Matrix.fromArray(inputArray);
        for (let i = 0; i < spider.weights.length; i++) {
            outputs = Matrix.multiply(spider.weights[i], outputs);
            outputs.add(spider.bias[i]);
            // activation function!
            outputs.map(this.activationFunction.func);
        }

        return outputs.toArray();
    }
}

class ActivationFunction {
    constructor(func, dfunc) {
        this.func = func;
        this.dfunc = dfunc;
    }
}

let sigmoid = new ActivationFunction(
    x => 1 / (1 + Math.exp(-x)),
    y => y * (1 - y)
);
