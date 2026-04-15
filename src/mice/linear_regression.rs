struct LinearRegression {
    coefficients: Vec<f64>,
    dim: usize,
}

impl LinearRegression {
    pub fn new(dim: usize) -> LinearRegression {
        LinearRegression {
            coefficients: vec![0.0; dim],
            dim,
        }
    }

    pub fn estimate(&self, points: &[f64]) -> Vec<f64> {
        let mut total = vec![0.0; points.len() / self.dim];
        let mut counter = 0;
        for i in 0..points.len() {
            total[counter] += self.coefficients[i % self.dim] * points[i];
            if (i + 1) % self.dim == 0 {
                counter += 1;
            }
        }
        total
    }

    pub fn fit(&self, data: &[f64]) -> &LinearRegression {
        self
    }
}
