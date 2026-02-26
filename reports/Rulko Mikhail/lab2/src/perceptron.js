class Perceptron {
    constructor() {
        this.w1 = 0;
        this.w2 = 0;
        this.bias = 0;
        this.epsilon = 0.001; 
    }

    getSum(x1, x2) {
        return x1 * this.w1 + x2 * this.w2 + this.bias;
    }

    predict(x1, x2) {
        return this.getSum(x1, x2) >= 0 ? 1 : -1;
    }

    train(dataset, mode = 'fixed', alphaFixed = 0.01) {
        let epochs = 0;
        const history = [];
        const maxEpochs = 10;

        while (epochs < maxEpochs) {
            let mseSum = 0;
            const shuffled = [...dataset].sort(() => Math.random() - 0.5);

            shuffled.forEach(point => {
                const out = this.getSum(point.x1, point.x2);
                const error = point.label - out;

                let alpha = alphaFixed;
                if (mode === 'adaptive') {
                    const norm = 1 + Math.pow(point.x1, 2) + Math.pow(point.x2, 2);
                    alpha = 0.5 * (1 / norm);
                }

                this.w1 += alpha * error * point.x1;
                this.w2 += alpha * error * point.x2;
                this.bias += alpha * error;

                mseSum += Math.pow(error, 2);
            });

            const avgMse = mseSum / dataset.length;
            history.push(avgMse);
            epochs++;

            if (avgMse <= this.epsilon) break;
        }
        return { history, epochs };
    }
}