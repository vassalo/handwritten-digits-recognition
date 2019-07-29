const PF = 0.7;

class Colony {
    constructor(N, neuralNet, layersNodes) {
        this.neuralNet = neuralNet;

        // Step 1
        this.Nf = Math.floor((0.9 - random() * 0.25) * N);
        this.Nm = N - this.Nf;

        // Step 2
        let sum = 0, n = 0;
        for (let i = 1; i < layersNodes.length; i++) {
            sum += layersNodes[i] * layersNodes[i - 1];
            n += layersNodes[i] * layersNodes[i - 1];
        }
        this.crossoverRadius = sum / (2 * layersNodes.length);
        // this.crossoverRadius = 2;
        console.log('Nf:', this.Nf, ', Nm:', this.Nm, ', medianIdx:', Math.floor((this.Nm - 1) / 2), ', radius:', this.crossoverRadius);

        this.bootstrapColony(N, layersNodes);

        // Step 3, 4, 5 and 6
        this.train();
    }

    train() {
        // Step 3
        let accuracy = this.calculateWeights();

        // Step 4
        const females = this.moveFemaleSpiders();

        // Step 5
        this.findMedianMale();
        const males = this.moveMaleSpiders();

        // Step 4 + 5
        for (let i = 0; i < this.Nf + this.Nm; i++) {
            this.spiders[i] = i < this.Nf ? females[i] : males[i - this.Nf];
        }

        // Step 6
        this.crossover();

        canvasContainer.setAttribute('data-train', 'Training\nAccuracy:\n' + accuracy.toFixed(2) + '%');
    }

    getBestSpider() {
        let bestSpider = this.spiders[0], idx = 0;
        for (const spider of this.spiders) {
            if (spider.weight > bestSpider.weight) {
                bestSpider = spider;
            }
            idx++;
        }
        // console.log('bestSpider:', bestIdx);
        return bestSpider;
    }

    getWorstSpiderIdx() {
        let worstSpider = this.spiders[0], worstSpiderIdx = 0, idx = 0;
        for (const spider of this.spiders) {
            if (spider.weight < worstSpider.weight) {
                worstSpider = spider;
                worstSpiderIdx = idx;
            }
            idx++;
        }
        return worstSpiderIdx;
    }

    findMedianMale() {
        const males = [];
        for (let i = this.Nf; i < this.Nf + this.Nm; i++) {
            males.push({...this.spiders[i]});
        }
        males.sort((a, b) => {
            if (a.weight > b.weight) {
                return 1;
            }
            if (a.weight < b.weight) {
                return -1;
            }
            return 0;
        });
        const medianIdx = Math.floor((males.length - 1) / 2);
        this.medianMale = males[medianIdx];
    }

    bootstrapColony(N, layersNodes) {
        this.spiders = [];
        for (let s = 0; s < N; s++) {
            this.spiders.push({weight: -1, value: {weights: [], bias: []}});

            const spider = this.spiders[s].value;
            for (let i = 1; i < layersNodes.length; i++) {
                spider.weights.push(new Matrix(layersNodes[i], layersNodes[i - 1]).randomize());
                spider.bias.push(new Matrix(layersNodes[i], 1).randomize());
            }
        }
    }

    calculateWeights() {
        for (let s = 0; s < this.spiders.length; s++) {
            const currAccuracy = this.predictBatch(this.spiders[s].value);
            this.spiders[s].accuracy = currAccuracy;

            if (this.worstAccuracy === undefined || currAccuracy < this.worstAccuracy) {
                this.worstAccuracy = currAccuracy;
            } else if (this.bestAccuracy === undefined || currAccuracy > this.bestAccuracy) {
                this.bestAccuracy = currAccuracy;
                this.bestIdx = s;
            }
        }

        for (let i = 0; i < this.spiders.length; i++) {
            this.spiders[i].weight = (this.spiders[i].accuracy - this.worstAccuracy) / (this.bestAccuracy - this.worstAccuracy);
        }

        // console.log('best', bestAccuracy);
        return this.bestAccuracy;
    }

    predictBatch(spider) {
        let correctPredictions = 0;
        for (let idx = 0; idx < trainSubsetImages.length; idx++) {
            let inputs = [];
            for (let i = 0; i < 784; i++) {
                let bright = trainSubsetImages[idx][i];
                inputs[i] = bright / 255;
            }

            const guess = getAnswer(this.neuralNet.predict(inputs, spider));
            if (guess === trainSubsetLabels[idx]) {
                correctPredictions++;
            }
        }

        return correctPredictions / trainSubsetImages.length;
    }

    moveFemaleSpiders() {
        const updatedSpiders = [];

        let spiderB = this.spiders[0];
        for (let spider of this.spiders) {
            if (spider.weight > spiderB.weight) {
                spiderB = spider;
            }
        }

        for(let i = 0; i < this.Nf; i++) {
            const I = this.spiders[i];

            // VibC
            let spiderC = undefined, minorDistance = undefined;
            for (let c = 0; c < this.Nf; c++) {
                const C = this.spiders[c];
                if (i === c) {
                    continue;
                }

                if (C.weight > I.weight) {
                    const dist = MatricesArray.distance(I.value.weights, C.value.weights);

                    if (minorDistance === undefined || dist < minorDistance) {
                        minorDistance = dist;
                        spiderC = C;
                    }
                }
            }
            if (spiderC === undefined) {
                minorDistance = 0;
                spiderC = I;
            }
            const VibC = spiderC.weight * Math.pow(Math.E, -minorDistance * minorDistance);

            // VibB
            const distB = MatricesArray.distance(I.value.weights, spiderB.value.weights);
            const VibB = spiderB.weight * Math.pow(Math.E, -distB * distB);

            // Moving
            const alpha = random(), beta = random(), gama = random();
            const randWeights = MatricesArray.newArrayAlike(I.value.weights, () => random());
            const randBias = MatricesArray.newArrayAlike(I.value.bias, () => random());

            // console.log('Vibs', VibB, VibC, alpha, beta, gama);

            const opWeightsVibC = MatricesArray.multiplyScalar(alpha * VibC, MatricesArray.subtract(spiderC.value.weights, I.value.weights));
            const opBiasVibC = MatricesArray.multiplyScalar(alpha * VibC, MatricesArray.subtract(spiderC.value.bias, I.value.bias));
            const opWeightsVibB = MatricesArray.multiplyScalar(beta * VibB, MatricesArray.subtract(spiderB.value.weights, I.value.weights));
            const opBiasVibB = MatricesArray.multiplyScalar(beta * VibB, MatricesArray.subtract(spiderB.value.bias, I.value.bias));
            const opWeightsRand = MatricesArray.multiplyScalar(gama, MatricesArray.addScalar(-0.5, randWeights));
            const opBiasRand = MatricesArray.multiplyScalar(gama, MatricesArray.addScalar(-0.5, randBias));

            updatedSpiders.push({weight: I.weight, value: {}});
            if (random() < PF) {
                updatedSpiders[i].value.weights = MatricesArray.add(MatricesArray.add(MatricesArray.add(I.value.weights, opWeightsVibC), opWeightsVibB), opWeightsRand);
                updatedSpiders[i].value.bias = MatricesArray.add(MatricesArray.add(MatricesArray.add(I.value.bias, opBiasVibC), opBiasVibB), opBiasRand);
            } else {
                updatedSpiders[i].value.weights = MatricesArray.add(MatricesArray.subtract(MatricesArray.subtract(I.value.weights, opWeightsVibC), opWeightsVibB), opWeightsRand);
                updatedSpiders[i].value.bias = MatricesArray.add(MatricesArray.subtract(MatricesArray.subtract(I.value.bias, opBiasVibC), opBiasVibB), opBiasRand);
            }
        }

        return updatedSpiders;
    }

    moveMaleSpiders() {
        const updatedSpiders = [];

        let dividendWeights = MatricesArray.newArrayAlike(this.spiders[0].value.weights);
        let dividendBias = MatricesArray.newArrayAlike(this.spiders[0].value.bias);
        let divisor = 0;
        for (let i = this.Nf; i < this.Nf + this.Nm; i++) {
            divisor += this.spiders[i].weight;

            dividendWeights = MatricesArray.add(dividendWeights, MatricesArray.multiplyScalar(this.spiders[i].weight, this.spiders[i].value.weights));
            dividendBias = MatricesArray.add(dividendBias, MatricesArray.multiplyScalar(this.spiders[i].weight, this.spiders[i].value.bias));
        }
        const weightedAverageWeights = MatricesArray.multiplyScalar(1 / divisor, dividendWeights);
        const weightedAverageBias = MatricesArray.multiplyScalar(1 / divisor, dividendBias);

        for (let i = this.Nf; i < this.Nf + this.Nm; i++) {
            const I = this.spiders[i];

            if (I.weight > this.medianMale.weight) {
                // VibF
                let spiderF = undefined, minorDistance = undefined;
                for (let f = 0; f < this.Nf; f++) {
                    const F = this.spiders[f];
                    const dist = MatricesArray.distance(I.value.weights, F.value.weights);

                    if (minorDistance === undefined || dist < minorDistance) {
                        minorDistance = dist;
                        spiderF = F;
                    }
                }
                const VibF = spiderF.weight * Math.pow(Math.E, -minorDistance * minorDistance);

                // Moving
                const alpha = random(), gama = random();
                const randWeights = MatricesArray.newArrayAlike(I.value.weights, () => random());
                const randBias = MatricesArray.newArrayAlike(I.value.bias, () => random());

                const opWeightsVibF = MatricesArray.multiplyScalar(alpha * VibF, MatricesArray.subtract(spiderF.value.weights, I.value.weights));
                const opBiasVibF = MatricesArray.multiplyScalar(alpha * VibF, MatricesArray.subtract(spiderF.value.bias, I.value.bias));
                const opWeightsRand = MatricesArray.multiplyScalar(gama, MatricesArray.addScalar(-0.5, randWeights));
                const opBiasRand = MatricesArray.multiplyScalar(gama, MatricesArray.addScalar(-0.5, randBias));

                updatedSpiders.push({weight: I.weight, value: {}});
                updatedSpiders[i - this.Nf].value.weights = MatricesArray.add(I.value.weights, MatricesArray.add(opWeightsVibF, opWeightsRand));
                updatedSpiders[i - this.Nf].value.bias = MatricesArray.add(I.value.bias, MatricesArray.add(opBiasVibF, opBiasRand));
            } else {
                const alpha = random();
                const opWeights = MatricesArray.multiplyScalar(alpha, MatricesArray.subtract(weightedAverageWeights, I.value.weights));
                const opBias = MatricesArray.multiplyScalar(alpha, MatricesArray.subtract(weightedAverageBias, I.value.bias));

                updatedSpiders.push({weight: I.weight, value: {}});
                updatedSpiders[i - this.Nf].value.weights = MatricesArray.add(I.value.weights, opWeights);
                updatedSpiders[i - this.Nf].value.bias = MatricesArray.add(I.value.bias, opBias);
            }
        }

        return updatedSpiders;
    }

    crossover() {
        // this.calculateWeights();
        // this.findMedianMale();

        console.log('medianMale:', this.medianMale.weight, ', bestSpider:', this.bestAccuracy, ',', this.bestIdx < this.Nf ? 'female' : 'male');
        // console.log(this.medianMale, this.spiders);
        for (let i = this.Nf; i < this.Nf + this.Nm; i++) {
            // console.log(this.spiders[i].weight > this.medianMale.weight, this.spiders[i].weight, '>', this.medianMale.weight);
            if (this.spiders[i].weight > this.medianMale.weight) {
                const females = [];
                for (let f = 0; f < this.Nf; f++) {
                    const dist = MatricesArray.distance(this.spiders[i].value.weights, this.spiders[f].value.weights);
                    console.log('dist', dist);
                    if (dist <= this.crossoverRadius) {
                        females.push(this.spiders[f]);
                    }
                }
                // console.log('females:', females.length);

                if (females.length > 0) {
                    const adults = females.concat(this.spiders[i]);

                    let sumOfWeights = 0;
                    for (let s of adults) {
                        sumOfWeights += s.weight;
                    }

                    const probabilities = [];
                    for (let s of adults) {
                        probabilities.push(s.weight / sumOfWeights);
                    }

                    // Generate new Spider
                    const newSpider = {weight: -1, value: {weights: [], bias: []}};
                    for (let j = 0; j < adults[0].value.weights.length; j++) {
                        const rand = random();
                        let accumulatedP = 0;
                        for (let p = 0; p < probabilities.length; p++) {
                            accumulatedP += probabilities[p];
                            if (accumulatedP >= rand) {
                                newSpider.value.weights.push(adults[p].value.weights[j]);
                                newSpider.value.bias.push(adults[p].value.bias[j]);
                                break;
                            }
                        }
                    }

                    newSpider.accuracy = this.predictBatch(newSpider.value);
                    newSpider.weight = (newSpider.accuracy - this.worstAccuracy) / (this.bestAccuracy - this.worstAccuracy);
                    console.log('GENERATED SPIDER!');

                    // Compare it with the worst spider
                    const worstIdx = this.getWorstSpiderIdx();
                    if (newSpider.weight > this.spiders[worstIdx].weight) {
                        this.spiders[worstIdx] = newSpider;
                        console.log('REPLACED SPIDER!');
                    }

                }
            }
        }
        console.log('crossover finished');
    }
}
