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

    pub fn fit(&self, data: &[f64], target: &[f64]) -> &LinearRegression {
        let product = vec![0.0; data.len()];
        // let nrows =
        // for i in 0..self.dim {
        //
        // }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*; // has access to everything, including private

    #[test]
    fn test_estimate() {
        const DIM: usize = 5;
        const N_POINTS: usize = 4;
        const POINTS: &[f64] = &[1.0; DIM * N_POINTS];
        const TOTAL: f64 = ((DIM - 1).pow(2) + (DIM - 1)) as f64 / 2.0;
        const EXPECTED: &[f64] = &[TOTAL; N_POINTS];

        let mut model = LinearRegression::new(DIM);
        for i in 0..DIM {
            model.coefficients[i] = i as f64;
        }
        let estimates = model.estimate(POINTS);
        assert!(EXPECTED.len() == estimates.len());
        for (p, q) in EXPECTED.iter().zip(&estimates) {
            assert!(
                (p - q).abs() < 1e-12,
                "expected: {:?} actual: {:?}",
                EXPECTED,
                &estimates
            );
        }
    }
}
