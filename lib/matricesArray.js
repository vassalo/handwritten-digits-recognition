class MatricesArray {

    static hasError(a, b) {
        if (a === undefined || b === undefined) {
            console.error('Objects are undefined');
            return true;
        }
        if (a.length === undefined || b.length === undefined) {
            console.error('Objects must be arrays.');
            return true;
        }
        if (a.length !== b.length) {
            console.error('Arrays must be of the same size.');
            return true;
        }
        return false;
    }

    static newArrayAlike(reference, fillFunc = () => 0) {
        const res = [];
        for (let i = 0; i < reference.length; i++) {
            if (!(reference[i] instanceof Matrix)) {
                console.error('Reference must be an Array of Matrix objects.');
                return;
            }

            res.push(new Matrix(reference[i].rows, reference[i].cols).map(fillFunc));
        }
        return res;
    }

    static add(a, b) {
        if (this.hasError(a, b)) {
            console.log('add', a, b);
            return;
        }

        const res = [];
        for (let i = 0; i < a.length; i++) {
            const aux = Matrix.add(a[i], b[i]);
            res.push(aux);
        }
        return res;
    }

    static addScalar(n, a) {
        if (isNaN(n) || !Array.isArray(a)) {
            console.error('Add Scalar operation needs a number and an Array of Matrix objects.')
        }

        const res = [];
        for (let i = 0; i < a.length; i++) {
            res.push(a[i].copy().add(n));
        }
        return res;
    }

    static subtract(a, b) {
        if (this.hasError(a, b)) {
            console.log('subtract');
            return;
        }

        const res = [];
        for (let i = 0; i < a.length; i++) {
            res.push(Matrix.subtract(a[i], b[i]));
        }
        return res;
    }

    static multiplyScalar(n, matrices) {
        if (isNaN(n)) {
            console.error('N must be a number.', n);
        }
        if (!Array.isArray(matrices)) {
            console.error('Matrices must be an array of Matrix objects.');
        }

        for (let i = 0; i < matrices.length; i++) {
            matrices[i] = matrices[i].copy().multiply(n);
        }

        return matrices;
    }

    static distance(a, b) {
        if (this.hasError(a, b)) {
            console.log('distance', a, b);
            return;
        }

        let res = 0;
        for (let i = 0; i < a.length; i++) {
            const aux = a[i].euclidianDistance(b[i]);
            res += aux * aux;
        }
        return Math.sqrt(res);
    }
}
