pub struct Matrix {
    values: *const f64,
    _data: Option<Vec<f64>>,
    nrows: usize,
    ncols: usize,
    tr: bool,
    len: usize,
}

impl Matrix {
    pub fn from_slice(m: &[f64], nrows: usize, ncols: usize) -> Matrix {
        let len = nrows * ncols;
        Matrix {
            values: m.as_ptr(),
            _data: None,
            nrows,
            ncols,
            tr: false,
            len,
        }
    }

    pub fn new(m: Vec<f64>, nrows: usize, ncols: usize) -> Matrix {
        let len = nrows * ncols;
        Matrix {
            values: m.as_ptr(),
            _data: Some(m),
            nrows,
            ncols,
            tr: false,
            len,
        }
    }

    pub fn eye(nrows: usize, ncols: usize) -> Matrix {
        let len = nrows * ncols;
        let mut v = vec![0.0; len];
        let mut count = 0;
        for i in 0..v.len() {
            if i % ncols == count {
                v[i] = 1.0;
                count += 1;
            }
        }
        Matrix {
            values: v.as_ptr(),
            _data: Some(v),
            nrows,
            ncols,
            tr: false,
            len,
        }
    }

    pub fn square(&self) -> Matrix {
        let mut product = vec![0.0; self.nrows * self.nrows];
        let values = self.as_slice();
        for row in 0..self.nrows {
            for col in row..self.nrows {
                let a = &values[row * self.ncols..(row + 1) * self.ncols];
                let b = &values[col * self.ncols..(col + 1) * self.ncols];
                let v: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                product[row * self.nrows + col] = v;
                product[col * self.nrows + row] = v;
            }
        }
        Matrix::new(product, self.nrows, self.nrows)
    }

    pub fn svd(&self) -> (Matrix, Matrix, Matrix) {
        (Matrix::eye(2, 2), Matrix::eye(2, 2), Matrix::eye(2, 2))
    }

    pub fn tr(&self) -> Self {
        self.transpose()
    }

    pub fn transpose(&self) -> Self {
        let mut m = Self::from_slice(self.as_slice(), self.ncols, self.nrows);
        m.tr = true;
        m
    }

    #[inline(always)]
    fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.values, self.len) }
    }

    #[inline(always)]
    fn _idx(&self, row: usize, col: usize) -> usize {
        if self.tr {
            println!("TEST {} {}", col * self.nrows + row, row * self.ncols + col);
            return col * self.nrows + row;
        }
        row * self.ncols + col
    }
}

impl std::ops::Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Self::Output {
        if self.ncols != other.nrows {
            panic!("Shape Mismatch!");
        }
        let mut product = vec![0.0; self.nrows * other.ncols];
        let (lvalues, rvalues) = (self.as_slice(), other.as_slice());
        for row in 0..self.nrows {
            for col in row..other.ncols {
                let a = &lvalues[row * self.ncols..(row + 1) * self.ncols];
                let b = (0..a.len()).map(|i| rvalues[i * other.ncols + col]);
                let v: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                product[row * self.nrows + col] = v;
                product[col * self.nrows + row] = v;
            }
        }
        Matrix::new(product, self.nrows, other.ncols)
    }
}
impl std::ops::Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Self::Output {
        &self * &other
    }
}

impl std::ops::Add for &Matrix {
    type Output = Vec<f64>;

    fn add(self, other: &Matrix) -> Self::Output {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!(
                "Shape Mismatch! Self:{},{} Other:{}{}",
                self.nrows, self.ncols, other.nrows, other.ncols
            );
        }
        let (lvalues, rvalues) = (self.as_slice(), other.as_slice());
        let sum = lvalues.iter().zip(rvalues).map(|(x, y)| x + y).collect();
        sum
    }
}

impl std::ops::Deref for Matrix {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_SQUARE: &[f64] = &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    const TEST_SQUARE_T: &[f64] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    const TEST_SQUARE_RESULT: &[f64] = &[2.0, 4.0, 6.0, 4.0, 8.0, 12.0, 6.0, 12.0, 18.0];
    const TEST_INV: &[f64] = &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
    const INV_FACTOR: f64 = 1.0 / 392.0;

    #[test]
    fn test_square() {
        let m = Matrix::from_slice(TEST_SQUARE, 3, 2).square();
        let square = m.as_slice();
        assert!(square.len() == 9, "Not the right shape! {}", square.len());
        for (p, q) in square.iter().zip(TEST_SQUARE_RESULT) {
            assert!(
                (p - q).abs() < 1e-10,
                "Expected: {:?}, Actual {:?}",
                TEST_SQUARE_RESULT,
                square
            );
        }
    }

    #[test]
    fn test_mult() {
        let m = &Matrix::from_slice(TEST_SQUARE, 3, 2) * &Matrix::from_slice(TEST_SQUARE_T, 2, 3);
        let prod = m.as_slice();
        assert!(prod.len() == 9, "Not the right shape! {}", prod.len());
        for (p, q) in prod.iter().zip(TEST_SQUARE_RESULT) {
            assert!(
                (p - q).abs() < 1e-10,
                "Expected: {:?}, Actual {:?}",
                TEST_SQUARE_RESULT,
                prod
            );
        }
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_slice(TEST_SQUARE, 3, 2).tr();
        assert!(m.tr);
        let tr = m.as_slice();
        assert!(
            tr.len() == TEST_SQUARE_T.len(),
            "Not the right shape! {}",
            tr.len()
        );
        for r in 0..2 {
            for c in 0..3 {
                let (p, q) = (TEST_SQUARE_T[r * 3 + c], tr[m._idx(r, c)]);
                assert!(
                    (p - q).abs() < 1e-10,
                    "Expected: {:?}, Actual {:?} \n({},{})",
                    p,
                    q,
                    r,
                    c
                );
            }
        }
    }

    #[test]
    fn test_svd() {
        let (u, e, v) = Matrix::from_slice(TEST_SQUARE, 3, 2).square().svd();
        let inv = u.as_slice();
        let true_inverse: Vec<f64> = TEST_INV.iter().map(|x| x * INV_FACTOR).collect();
        assert!(inv.len() == 9, "Not the right shape! {}", inv.len());
        for (p, q) in inv.iter().zip(&true_inverse) {
            assert!(
                (p - q).abs() < 1e-10,
                "Expected: {:?}, Actual {:?}",
                true_inverse,
                inv
            );
        }
    }
}
