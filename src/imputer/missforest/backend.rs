use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Clone)]
enum Task {
    Regression,
    Classification,
}

#[derive(Clone)]
enum Node {
    Internal {
        feature_index: usize,
        threshold: f64,
        children: Box<[Node; 2]>,
    },
    Leaf {
        value: f64, // mean (regression) or majority class (classification)
    },
}

impl Node {
    fn predict(&self, sample: &[f64]) -> f64 {
        match self {
            Node::Leaf { value } => *value,
            Node::Internal {
                feature_index,
                threshold,
                children,
            } => {
                if sample[*feature_index] <= *threshold {
                    return children[0].predict(sample);
                }
                children[1].predict(sample)
            }
        }
    }
}

#[derive(Clone)]
pub struct DecisionTree {
    root: Option<Node>,
    task: Task,
    max_depth: usize,
    depth: Option<usize>,
    min_samples_leaf: usize,
}

#[derive(Clone)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    min_samples_leaf: usize,
    seed: u64,
    max_depth: usize,
}

impl RandomForest {
    pub fn new(
        n_trees: usize,
        seed: u64,
        max_depth: usize,
        min_samples_leaf: usize,
    ) -> RandomForest {
        RandomForest {
            trees: Vec::new(),
            n_trees,
            seed,
            min_samples_leaf,
            max_depth,
        }
    }
    pub fn fit(&mut self, data: &Array2<f64>, target: ArrayView1<f64>) -> &RandomForest {
        self.trees = Vec::with_capacity(self.n_trees);
        for i in 0..self.n_trees {
            self.trees.push(DecisionTree::new(
                self.max_depth,
                self.min_samples_leaf,
                Task::Regression,
            ));
            let (fit_data, fit_target) = self.bootstrap(data.view(), target);
            self.trees[i].fit(fit_data.view(), fit_target.view());
        }
        self
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(data.nrows());
        for t in &self.trees {
            result += &t.predict(data);
        }
        result / self.n_trees as f64
    }

    fn bootstrap(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n = x.nrows();
        let indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
        (x.select(Axis(0), &indices), y.select(Axis(0), &indices))
    }
}

impl DecisionTree {
    fn new(max_depth: usize, min_samples_leaf: usize, task: Task) -> DecisionTree {
        DecisionTree {
            root: None,
            max_depth,
            min_samples_leaf,
            task,
            depth: None,
        }
    }

    fn fit(&mut self, data: ArrayView2<f64>, target: ArrayView1<f64>) -> &Self {
        // setup book keeping
        let max_nodes: usize = usize::pow(2, self.max_depth as u32);
        let mut active_nodes: Vec<&Node> = Vec::new();
        // how to properly track the feauters used for splitting?
        let mut features: Vec<usize> = (0..data.ncols()).collect();
        // build tree
        for i in 0..max_nodes {
            // compute best split
            // split
            // check stop condition
            if active_nodes.len() == 0 {
                // all leaves done; tree is constructed
                break;
            }
        }
        //  consider prunning
        self
    }

    fn predict_row(&self, row: &[f64]) -> f64 {
        self.root.as_ref().unwrap().predict(row)
    }

    fn predict(&self, data: &Array2<f64>) -> Array1<f64> {
        let nrows = data.nrows();
        let predictions: Vec<f64> = (0..nrows)
            .into_par_iter()
            .map(|i| self.predict_row(data.row(i).as_slice().expect("Slice not in mem")))
            .collect();
        Array1::from_vec(predictions)
    }

    fn impurity(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
        match self.task {
            Task::Regression => 0.0,
            Task::Classification => {
                panic!("Not Implemneted Yet!")
            }
        }
    }

    fn depth(&self) -> usize {
        panic!("TODO: Implement")
    }

    fn n_leaves(&self) -> usize {
        panic!("TODO: Implement")
    }
}

#[cfg(test)]
mod decision_tree {
    use super::*;
    /// Linearly separable 2-class dataset.
    // fn clf_data() -> (Vec<Vec<f64>>, Array1<f64>) {
    //     let x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64, (i % 3) as f64]).collect();
    //     let y: Vec<usize> = (0..40).map(|i| if i < 20 { 0 } else { 1 }).collect();
    //     (x, y)
    // }

    /// Simple linear regression dataset: y = 2x.
    fn reg_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((40, 1), (0..40).map(|x| x as f64).collect())
            .expect("Failed to create test data!");
        let y = (0..40).map(|i| 2.0 * i as f64).collect();
        (x, y)
    }

    fn accuracy(pred: &[usize], truth: &[usize]) -> f64 {
        pred.iter().zip(truth).filter(|(p, t)| p == t).count() as f64 / truth.len() as f64
    }

    fn mse(pred: &[f64], truth: &[f64]) -> f64 {
        panic!("TODO: Implement")
    }

    #[test]
    fn predict_output_length_matches_input() {
        let (x, y) = reg_data();
        let mut clf = DecisionTree::new(4, 2, Task::Regression);
        clf.fit(x.view(), y.view());
        assert_eq!(clf.predict(&x).len(), x.len());
    }

    // #[test]
    // fn predict_proba_shape_and_sums_to_one() {
    //     let (x, y) = clf_data();
    //     let n_classes = 2;
    //     let clf = DecisionTree::new().fit(&x, &y);
    //     let proba = clf.predict_proba(&x);
    //     assert_eq!(proba.len(), x.len());
    //     for row in &proba {
    //         assert_eq!(row.len(), n_classes);
    //         let sum: f64 = row.iter().sum();
    //         assert!((sum - 1.0).abs() < 1e-9, "proba row sums to {sum}");
    //     }
    // }

    // #[test]
    // fn feature_importances_sum_to_one() {
    //     let (x, y) = clf_data();
    //     let clf = DecisionTree::new().fit(&x, &y);
    //     let sum: f64 = clf.feature_importances().iter().sum();
    //     assert!((sum - 1.0).abs() < 1e-9, "importances sum to {sum}");
    // }

    #[test]
    fn max_depth_is_respected() {
        let (x, y) = reg_data();
        let mut clf = DecisionTree::new(2, 0, Task::Regression);
        clf.fit(x.view(), y.view());
        assert!(clf.depth() <= 2);
    }

    #[test]
    fn deeper_tree_has_more_leaves() {
        let (x, y) = reg_data();
        let mut shallow = DecisionTree::new(2, 0, Task::Regression);
        let mut deep = DecisionTree::new(10, 0, Task::Regression);
        shallow.fit(x.view(), y.view());
        deep.fit(x.view(), y.view());
        assert!(deep.n_leaves() >= shallow.n_leaves());
    }

    // ── Correctness ───────────────────────────────────────────────────────────

    #[test]
    fn classifier_memorises_training_data() {
        let (x, y) = reg_data();
        let mut dt = DecisionTree::new(2, 0, Task::Regression);
        let pred = dt.fit(x.view(), y.view()).predict(&x);
        assert_eq!(mse(pred.as_slice().unwrap(), y.as_slice().unwrap()), 1.0);
    }

    #[test]
    fn regressor_fits_linear_target() {
        let (x, y) = reg_data();
        let pred = DecisionTree::new(2, 0, Task::Regression)
            .fit(x.view(), y.view())
            .predict(&x);
        let ss_res: f64 = pred.iter().zip(&y).map(|(p, t)| (p - t).powi(2)).sum();
        assert!(ss_res < 1e-6, "residual SS = {ss_res}");
    }

    #[test]
    fn single_sample_predict() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![0.0]);
        let pred = DecisionTree::new(2, 0, Task::Regression)
            .fit(x.view(), y.view())
            .predict(&x);
        assert_eq!(pred, y);
    }

    #[test]
    #[should_panic]
    fn predict_before_fit_panics() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        DecisionTree::new(2, 0, Task::Regression).predict(&x);
    }
}

#[cfg(test)]
mod random_forest {
    use super::*;
}
