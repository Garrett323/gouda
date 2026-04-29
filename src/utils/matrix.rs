use std::rc::Rc;

pub struct Matrix {
    values: Rc<[f64]>,
    nrows: usize,
    ncols: usize,
    stride: [usize; 2],
    len: usize,
}

impl Matrix {
    pub fn new(m: &[f64], nrows: usize, ncols: usize) -> Matrix {
        let len = nrows * ncols;
        Matrix {
            values: Rc::from(m),
            nrows,
            ncols,
            stride: [ncols, 1],
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
            values: Rc::from(v),
            nrows,
            ncols,
            stride: [ncols, 1],
            len,
        }
    }

    pub fn square(&self) -> Matrix {
        let mut product = vec![0.0; self.ncols * self.ncols];
        let values = self.as_slice();
        for row in 0..self.ncols {
            for col in row..self.ncols {
                let a = (0..self.nrows).map(|i| values[self._idx(i, row)]);
                let b = (0..self.nrows).map(|i| values[self._idx(i, col)]);
                let v: f64 = a.zip(b).map(|(x, y)| x * y).sum();
                product[row * self.ncols + col] = v;
                product[col * self.ncols + row] = v;
            }
        }
        Matrix::new(&product, self.ncols, self.ncols)
    }

    pub fn svd(&self) -> (Matrix, Matrix, Matrix) {
        (Matrix::eye(2, 2), Matrix::eye(2, 2), Matrix::eye(2, 2))
    }

    pub fn tr(&self) -> Self {
        self.transpose()
    }

    pub fn transpose(&self) -> Self {
        Matrix {
            values: Rc::clone(&self.values),
            nrows: self.ncols,
            ncols: self.nrows,
            stride: [1, self.ncols],
            len: self.len,
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[f64] {
        &self.values
    }

    #[inline(always)]
    fn _idx(&self, row: usize, col: usize) -> usize {
        row * self.stride[0] + col * self.stride[1]
    }
}

impl std::ops::Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Self::Output {
        if self.ncols != other.nrows {
            panic!("Shape Mismatch!");
        }
        let mut product = vec![0.0; self.nrows * other.ncols];
        for row in 0..self.nrows {
            for col in 0..other.ncols {
                // let a = &lvalues[row * self.ncols..(row + 1) * self.ncols];
                let a = (0..self.ncols).map(|i| self.values[self._idx(row, i)]);
                let b = (0..self.ncols).map(|i| other.values[other._idx(i, col)]);
                let v: f64 = a.zip(b).map(|(x, y)| x * y).sum();
                product[row * self.nrows + col] = v;
            }
        }
        Matrix::new(&product, self.nrows, other.ncols)
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for row in 0..self.nrows {
            write!(f, "[")?;
            for col in 0..self.ncols {
                if col > 0 {
                    write!(f, ", ")?;
                }
                let idx = row * self.stride[0] + col * self.stride[1];
                write!(f, "{:8.3}", self.as_slice()[idx])?;
            }
            write!(f, "]")?;
            if row < self.nrows - 1 {
                write!(f, ",\n ")?;
            }
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

impl std::ops::Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Self::Output {
        &self * &other
    }
}

impl std::ops::Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Self::Output {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!(
                "Shape Mismatch! Self:{},{} Other:{}{}",
                self.nrows, self.ncols, other.nrows, other.ncols
            );
        }
        let (lvalues, rvalues) = (self.as_slice(), other.as_slice());
        let sum: Vec<f64> = lvalues.iter().zip(rvalues).map(|(x, y)| x + y).collect();
        Matrix::new(&sum, self.nrows, other.ncols)
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
    const TEST_T_SQUARE_RESULT: &[f64] = &[2.0, 4.0, 6.0, 4.0, 8.0, 12.0, 6.0, 12.0, 18.0];
    const TEST_SQUARE_RESULT: &[f64] = &[14.0, 14.0, 14.0, 14.0];

    // const TEST_INV: &[f64] = &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
    // const INV_FACTOR: f64 = 1.0 / 392.0;

    #[test]
    fn test_square() {
        let m = Matrix::new(TEST_SQUARE, 3, 2).square();
        let square = m.as_slice();
        assert!(
            square.len() == TEST_SQUARE_RESULT.len(),
            "Not the right shape! {}",
            square.len()
        );
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
    fn test_square_transpose() {
        let m = Matrix::new(TEST_SQUARE, 3, 2);
        let t = m.tr();
        println!("{m}\n{t}");
        let square = t.square();
        assert!(
            square.len == TEST_T_SQUARE_RESULT.len(),
            "Not the right shape! {}",
            square.len
        );
        for (p, q) in square.as_slice().iter().zip(TEST_T_SQUARE_RESULT) {
            assert!(
                (p - q).abs() < 1e-10,
                "Expected: {:?}, Actual \n{}",
                TEST_T_SQUARE_RESULT,
                square,
            );
        }
    }

    #[test]
    fn test_mult() {
        let m = &Matrix::new(TEST_SQUARE, 3, 2) * &Matrix::new(TEST_SQUARE_T, 2, 3);
        let prod = m.as_slice();
        assert!(prod.len() == 9, "Not the right shape! {}", prod.len());
        for (p, q) in prod.iter().zip(TEST_T_SQUARE_RESULT) {
            assert!(
                (p - q).abs() < 1e-10,
                "Expected: {:?}, Actual {:?}",
                TEST_T_SQUARE_RESULT,
                prod
            );
        }
    }

    #[test]
    fn test_transpose_mult() {
        let m = &Matrix::new(TEST_SQUARE, 3, 2) * &Matrix::new(TEST_SQUARE, 3, 2).tr();
        let prod = m.as_slice();
        assert!(prod.len() == 9, "Not the right shape! {}", prod.len());
        for (p, q) in prod.iter().zip(TEST_T_SQUARE_RESULT) {
            assert!(
                (p - q).abs() < 1e-10,
                "Expected: {:?}, Actual {:?}",
                TEST_T_SQUARE_RESULT,
                prod
            );
        }
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::new(TEST_SQUARE, 3, 2).tr();
        assert!((m.stride[0] == 1) && (m.stride[1] == 2), "{:?}", m.stride);
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
    fn test_nonsymmetric_mult() {
        // A (3x2):        B (2x3):
        // [1, 0]          [1, 2, 3]
        // [0, 1]          [4, 5, 6]
        // [0, 0]
        let a = Matrix::new(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3, 2);
        let b = Matrix::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let m = &a * &b;

        // Expected: [[1,2,3],[4,5,6],[0,0,0]]
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0];
        for (i, (p, q)) in m.as_slice().iter().zip(&expected).enumerate() {
            assert!((p - q).abs() < 1e-10, "index {i}: got {p}, expected {q}");
        }
    }

    // fn _test_svd() {
    //     let (u, e, v) = Matrix::new(TEST_SQUARE, 3, 2).square().svd();
    //     let inv = u.as_slice();
    //     let true_inverse: Vec<f64> = TEST_INV.iter().map(|x| x * INV_FACTOR).collect();
    //     assert!(inv.len() == 9, "Not the right shape! {}", inv.len());
    //     for (p, q) in inv.iter().zip(&true_inverse) {
    //         assert!(
    //             (p - q).abs() < 1e-10,
    //             "Expected: {:?}, Actual {:?}",
    //             true_inverse,
    //             inv
    //         );
    //     }
    // }
}
