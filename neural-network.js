class NeuralNetwork {
    constructor(inputNodes, outputNodes) {
        if (typeof inputNodes === 'number') {
            this.layersNodes = [inputNodes, outputNodes];

            this.bootstrapWeigths();
            this.setLearningRate();
            this.setActivationFunction();
        } else {
            let data = inputNodes;
            let activationFunc = outputNodes === undefined ? sigmoid : outputNodes;

            this.layersNodes = data.layersNodes;

            this.weights = [];
            for (let m of data.weights) {
                this.weights.push(new Matrix(m.rows, m.cols).map((el, i, j) => m.data[i][j]));
            }

            this.bias = [];
            for (let m of data.bias) {
                this.bias.push(new Matrix(m.rows, m.cols).map((el, i, j) => m.data[i][j]));
            }

            this.learningRate = data.learningRate;

            this.setActivationFunction(activationFunc);
        }
    }

    bootstrapWeigths() {
        this.weights = [];
        this.bias = [];
        for (let i = 1; i < this.layersNodes.length; i++) {
            this.weights.push(new Matrix(this.layersNodes[i], this.layersNodes[i - 1]).randomize());
            this.bias.push(new Matrix(this.layersNodes[i], 1).randomize());
        }
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

    predict(inputArray) {
        let outputs = Matrix.fromArray(inputArray);
        for (let i = 0; i < this.weights.length; i++) {
            outputs = Matrix.multiply(this.weights[i], outputs);
            outputs.add(this.bias[i]);
            // activation function!
            outputs.map(this.activationFunction.func);
        }

        return outputs.toArray();
    }

    train(inputArray, targetArray) {
        let inputs = Matrix.fromArray(inputArray);
        let outputs = [];
        for (let i = 0; i < this.weights.length; i++) {
            outputs.push(Matrix.multiply(this.weights[i], i === 0 ? inputs : outputs[i - 1]));
            outputs[i].add(this.bias[i]);
            // activation function!
            outputs[i].map(this.activationFunction.func);
        }

        let targets = Matrix.fromArray(targetArray);
        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        let outputErrors = Matrix.subtract(targets, outputs[outputs.length - 1]);
        for (let i = outputs.length - 1; i >= 0; i--) {
            // Calculate the error
            outputErrors = i === outputs.length - 1 ? outputErrors
                : Matrix.multiply(Matrix.transpose(this.weights[i + 1]), outputErrors);

            // Calculate gradient
            let gradients = Matrix.map(outputs[i], this.activationFunction.dfunc);
            gradients.multiply(outputErrors);
            gradients.multiply(this.learningRate);

            // Calculate deltas
            let inputsT = Matrix.transpose(i === 0 ? inputs : outputs[i - 1]);
            let weightDeltas = Matrix.multiply(gradients, inputsT);

            // Adjust the weights by deltas
            this.weights[i].add(weightDeltas);
            // Adjust the bias by its deltas (which is just the gradients)
            this.bias[i].add(gradients);
        }
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {
        data = JSON.parse(data);
        return new NeuralNetwork(data);
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