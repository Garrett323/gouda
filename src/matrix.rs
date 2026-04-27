pub struct Matrix {
    values: *const f64,
    nrows: usize,
    ncols: usize,
    len: usize,
}

impl Matrix {
    pub fn from_slice(m: &[f64], nrows: usize, ncols: usize) -> Matrix {
        let len = nrows * ncols;
        Matrix {
            values: m.as_ptr(),
            nrows,
            ncols,
            len,
        }
    }
    pub fn square(&self) -> Vec<f64> {
        let mut product = vec![0.0; self.nrows * self.nrows];
        let values;
        unsafe { values = std::slice::from_raw_parts(self.values, self.len) }
        for row in 0..self.nrows {
            for col in row..self.nrows {
                let a = &values[row * self.ncols..(row + 1) * self.ncols];
                let b = &values[col * self.ncols..(col + 1) * self.ncols];
                let v: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                product[row * self.nrows + col] = v;
                product[col * self.nrows + row] = v;
            }
        }
        product
    }
}
// impl std::ops::Mul for &Matrix {
//     type Output = Vec<f64>;
//
//     fn mul(self, other: &Matrix) -> Self::Output {
//         if self.ncols != other.nrows {
//             panic!("Shape Mismatch!");
//         }
//         let mut product = vec![0.0; self.nrows * other.ncols];
//         let (lvalues, rvalues);
//         unsafe {
//             (lvalues, rvalues) = (
//                 std::slice::from_raw_parts(self.values, self.len),
//                 std::slice::from_raw_parts(other.values, other.len),
//             )
//         }
//         for row in 0..self.nrows {
//             for col in row..self.nrows {
//                 let a = &lvalues[row * self.ncols..(row + 1) * self.ncols];
//                 let b = &rvalues[col * self.ncols..(col + 1) * self.ncols];
//                 let v: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
//                 product[row * self.nrows + col] = v;
//                 product[col * self.nrows + row] = v;
//             }
//         }
//         product
//     }
// }
// impl std::ops::Mul for Matrix {
//     type Output = Vec<f64>;
//
//     fn mul(self, other: Matrix) -> Self::Output {
//         &self * &other
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_SQUARE: &[f64] = &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    const TEST_SQUARE_T: &[f64] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

    const TEST_SQUARE_RESULT: &[f64] = &[2.0, 4.0, 6.0, 4.0, 8.0, 12.0, 6.0, 12.0, 18.0];

    #[test]
    fn test_square() {
        let square = Matrix::from_slice(TEST_SQUARE, 3, 2).square();
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

    // #[test]
    // fn test_mult() {
    //     let prod =
    //         &Matrix::from_slice(TEST_SQUARE, 3, 2) * &Matrix::from_slice(TEST_SQUARE_T, 2, 3);
    //     assert!(prod.len() == 9, "Not the right shape! {}", prod.len());
    //     for (p, q) in prod.iter().zip(TEST_SQUARE_RESULT) {
    //         assert!(
    //             (p - q).abs() < 1e-10,
    //             "Expected: {:?}, Actual {:?}",
    //             TEST_SQUARE_RESULT,
    //             prod
    //         );
    //     }
    // }
}
