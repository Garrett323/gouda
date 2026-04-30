// TODO:
// Write mat mul Tests
// Support slicing
use crate::utils::Matrix;

// View ----------------------------------------------------
pub struct ColView<'a> {
    data: &'a [f64],
    start: usize,
    stride: usize,
    len: usize,
}

impl VecLike for ColView<'_> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a> ColView<'a> {
    pub fn new(data: &'a [f64], start: usize, stride: usize, len: usize) -> ColView<'a> {
        ColView {
            data,
            start,
            stride,
            len,
        }
    }

    fn get(&self, row: usize) -> &f64 {
        &self.data[self.start + row * self.stride]
    }
}

impl std::ops::Index<usize> for ColView<'_> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}

impl std::ops::Add for &ColView<'_> {
    type Output = ColVec;
    fn add(self, rhs: Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(x, y)| x + y).collect()
    }
}

impl std::ops::Sub for &ColView<'_> {
    type Output = ColVec;
    fn sub(self, rhs: Self) -> Self::Output {
        self.iter().zip(rhs.iter()).map(|(x, y)| x - y).collect()
    }
}

// !View ----------------------------------------------------
//
// Vec ----------------------------------------------------

pub struct ColVec {
    values: Vec<f64>,
}

impl ColVec {
    fn new(values: Vec<f64>) -> ColVec {
        ColVec { values }
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> + '_ {
        self.values.iter()
    }
}

impl VecLike for ColVec {
    fn len(&self) -> usize {
        self.values.len()
    }
}

impl FromIterator<f64> for ColVec {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut cv = ColVec::new(Vec::new());
        for e in iter {
            cv.values.push(e);
        }
        cv
    }
}

impl std::ops::Index<usize> for ColVec {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}
// !Vec ----------------------------------------------------
//
// Traits ----------------------------------------------------

pub trait VecLike: std::ops::Index<usize, Output = f64> {
    fn len(&self) -> usize;
    fn iter(&self) -> impl Iterator<Item = &f64> + '_ {
        (0..self.len()).map(|i| &self[i])
    }
    fn inner<R>(&self, rhs: &R) -> f64
    where
        R: VecLike,
    {
        self.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum()
    }
    fn outer<R>(&self, other: &R) -> Matrix
    where
        R: VecLike,
    {
        let mut values = vec![0.0; self.len() * self.len()];
        for r in 0..self.len() {
            for c in 0..self.len() {
                values[r * self.len() + c] = self[c] * other[r];
            }
        }
        Matrix::new(values, self.len(), self.len())
    }
    fn mat_mul(&self, other: &Matrix) -> Matrix {
        if other.nrows != self.len() {
            panic!("Shape mismatch");
        }
        let values: Vec<f64> = (0..other.ncols)
            .map(|row| (0..self.len()).map(|i| self[i] * other[(row, i)]).sum())
            .collect();
        Matrix::new(values, 1, other.ncols)
    }
}
macro_rules! impl_mul_inner {
    ($lhs:ty, $rhs:ty) => {
        impl std::ops::Mul<&$rhs> for &$lhs {
            type Output = f64;
            fn mul(self, rhs: &$rhs) -> Self::Output {
                self.inner(rhs)
            }
        }
    };
}

macro_rules! impl_mat_mul {
    ($lhs:ty) => {
        impl std::ops::Mul<&Matrix> for &$lhs {
            type Output = Matrix;
            fn mul(self, rhs: &Matrix) -> Self::Output {
                self.mat_mul(rhs)
            }
        }
    };
}

impl_mul_inner!(ColView<'_>, ColView<'_>);
impl_mul_inner!(ColVec, ColVec);
impl_mul_inner!(ColView<'_>, ColVec);
impl_mul_inner!(ColVec, ColView<'_>);

impl_mat_mul!(ColVec);
impl_mat_mul!(ColView<'_>);

// Tests ----------------------------------------------------
#[cfg(test)]
mod test {
    use super::*;
    const V: &[f64] = &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    const A: &[f64] = &[1.0, 2.0, 3.0];
    const B: &[f64] = &[1.0, 1.0, 1.0];
    const OUTER: &[f64] = &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];

    #[test]
    fn inner_product() {
        let v = ColVec::new(V.to_vec());
        let w = ColVec::new(V.to_vec());
        assert!((&v * &w) - (1.0 + 1.0 + 4.0 + 4.0 + 9.0 + 9.0) < 1e-10);
    }

    #[test]
    fn outer_product() {
        let v = ColVec::new(A.to_vec());
        let w = ColVec::new(B.to_vec());
        let outer = w.outer(&v);
        println!("{}\n {:?}", outer, OUTER);
        for (p, q) in outer.as_slice().iter().zip(OUTER.iter()) {
            assert!((p - q) < 1e-10);
        }
    }
}
