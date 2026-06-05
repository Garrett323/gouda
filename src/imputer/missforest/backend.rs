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

    /// Maximum depth from this node (leaf = 0).
    fn depth(&self) -> usize {
        match self {
            Node::Leaf { .. } => 0,
            Node::Internal { children, .. } => 1 + children[0].depth().max(children[1].depth()),
        }
    }

    /// Number of leaf nodes in this subtree.
    fn n_leaves(&self) -> usize {
        match self {
            Node::Leaf { .. } => 1,
            Node::Internal { children, .. } => children[0].n_leaves() + children[1].n_leaves(),
        }
    }
}

#[derive(Clone)]
pub struct DecisionTree {
    root: Option<Node>,
    task: Task,
    max_depth: usize,
    // depth: Option<usize>,
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
        let mut rng = StdRng::seed_from_u64(self.seed);
        let seeds: Vec<u64> = (0..self.n_trees).map(|_| rng.random::<u64>()).collect();
        self.trees = (0..self.n_trees)
            .into_par_iter()
            .map(|i| {
                let mut tree =
                    DecisionTree::new(self.max_depth, self.min_samples_leaf, Task::Regression);
                let (fit_data, fit_target) = self.bootstrap(data.view(), target, seeds[i]);
                tree.fit(fit_data.view(), fit_target.view());
                tree
            })
            .collect();
        self
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(data.nrows());
        for t in &self.trees {
            result += &t.predict(data);
        }
        result / self.n_trees as f64
    }

    fn bootstrap(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
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
            // depth: None,
        }
    }

    fn fit(&mut self, data: ArrayView2<f64>, target: ArrayView1<f64>) -> &Self {
        // // setup book keeping
        // let max_nodes: usize = usize::pow(2, self.max_depth as u32);
        // let mut active_nodes: Vec<&Node> = Vec::new();
        // // how to properly track the feauters used for splitting?
        // let mut features: Vec<usize> = (0..data.ncols()).collect();
        // // build tree
        // for i in 0..max_nodes {
        //     // compute best split
        //     // split
        //     // check stop condition
        //     if active_nodes.len() == 0 {
        //         // all leaves done; tree is constructed
        //         break;
        //     }
        // }
        // //  consider prunning
        let all_features: Vec<usize> = (0..data.ncols()).collect();
        self.root = Some(self.build_node(data, target, &all_features, 0));
        self
    }

    /// Recursively build a node.
    fn build_node(
        &self,
        data: ArrayView2<f64>,
        target: ArrayView1<f64>,
        feature_indices: &[usize],
        depth: usize,
    ) -> Node {
        // Stop conditions: max depth reached, too few samples, or pure leaf
        let stop = depth >= self.max_depth || target.len() <= self.min_samples_leaf.max(1) || {
            // all targets identical → variance 0 → nothing to split
            let first = target[0];
            target.iter().all(|&v| (v - first).abs() < 1e-12)
        };

        if stop {
            return Node::Leaf {
                value: leaf_value(target, &self.task),
            };
        }

        match best_split(
            data,
            target,
            &self.task,
            self.min_samples_leaf,
            feature_indices,
        ) {
            None => Node::Leaf {
                value: leaf_value(target, &self.task),
            },
            Some((fi, threshold)) => {
                let col: Vec<f64> = data.column(fi).to_vec();
                let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                    (0..data.nrows()).partition(|&i| col[i] <= threshold);

                let left_data = data.select(Axis(0), &left_idx);
                let left_target = target.select(Axis(0), &left_idx);
                let right_data = data.select(Axis(0), &right_idx);
                let right_target = target.select(Axis(0), &right_idx);

                let left_node = self.build_node(
                    left_data.view(),
                    left_target.view(),
                    feature_indices,
                    depth + 1,
                );
                let right_node = self.build_node(
                    right_data.view(),
                    right_target.view(),
                    feature_indices,
                    depth + 1,
                );

                Node::Internal {
                    feature_index: fi,
                    threshold,
                    children: Box::new([left_node, right_node]),
                }
            }
        }
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

    fn depth(&self) -> usize {
        self.root.as_ref().map_or(0, |x| x.depth())
    }

    fn n_leaves(&self) -> usize {
        self.root.as_ref().map_or(0, |x| x.n_leaves())
    }
}

// Gini impurity for a split (classification).
fn gini_impurity(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    fn gini(v: ArrayView1<f64>) -> f64 {
        let n = v.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        // count occurrences of each integer class label
        let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        for &x in v.iter() {
            *counts.entry(x.round() as u64).or_insert(0) += 1;
        }
        let impurity: f64 = counts
            .values()
            .map(|&c| {
                let p = c as f64 / n;
                p * (1.0 - p)
            })
            .sum();
        impurity
    }
    let n_total = (left.len() + right.len()) as f64;
    if n_total == 0.0 {
        return 0.0;
    }
    (gini(left) * left.len() as f64 + gini(right) * right.len() as f64) / n_total
}

/// Leaf value: mean for regression, majority class for classification.
fn leaf_value(target: ArrayView1<f64>, task: &Task) -> f64 {
    match task {
        Task::Regression => target.mean().unwrap_or(0.0),
        Task::Classification => {
            let mut counts: std::collections::HashMap<u64, usize> =
                std::collections::HashMap::new();
            for &x in target.iter() {
                *counts.entry(x.round() as u64).or_insert(0) += 1;
            }
            counts
                .into_iter()
                .max_by_key(|&(_, c)| c)
                .map(|(cls, _)| cls as f64)
                .unwrap_or(0.0)
        }
    }
}

/// Find the best (feature, threshold) split for the given data/target subset.
/// Returns `None` when no split improves impurity.
fn best_split(
    data: ArrayView2<f64>,
    target: ArrayView1<f64>,
    task: &Task,
    min_samples_leaf: usize,
    feature_indices: &[usize],
) -> Option<(usize, f64)> {
    let n = data.nrows();
    if n < 2 * min_samples_leaf.max(1) {
        return None;
    }

    // current impurity (unsplit)
    let baseline = match task {
        Task::Regression => {
            let mean = target.mean().unwrap_or(0.0);
            target.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64
        }
        Task::Classification => {
            mse_impurity(target, target.slice(ndarray::s![0..0]))
            // we reuse gini below; baseline is just used for comparison
        }
    };

    let mut best_impurity = f64::MAX;
    let mut best_feature = 0usize;
    let mut best_threshold = 0.0f64;
    let mut found = false;

    for &fi in feature_indices {
        let col: Vec<f64> = data.column(fi).to_vec();

        // collect unique candidate thresholds (midpoints between sorted unique values)
        let mut sorted_vals: Vec<f64> = col.clone();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        for w in sorted_vals.windows(2) {
            let threshold = (w[0] + w[1]) / 2.0;

            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                (0..n).partition(|&i| col[i] <= threshold);

            if left_idx.len() < min_samples_leaf.max(1) || right_idx.len() < min_samples_leaf.max(1)
            {
                continue;
            }

            let left_target = target.select(Axis(0), &left_idx);
            let right_target = target.select(Axis(0), &right_idx);

            let impurity = match task {
                Task::Regression => mse_impurity(left_target.view(), right_target.view()),
                Task::Classification => gini_impurity(left_target.view(), right_target.view()),
            };

            if impurity < best_impurity {
                best_impurity = impurity;
                best_feature = fi;
                best_threshold = threshold;
                found = true;
            }
        }
    }

    // only split if it actually reduces impurity
    if found && best_impurity < baseline {
        Some((best_feature, best_threshold))
    } else {
        None
    }
}
/// Weighted mean-squared-error impurity for a split.
/// Returns the combined impurity (lower is better).
fn mse_impurity(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    fn variance_times_n(v: ArrayView1<f64>) -> f64 {
        let n = v.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let mean = v.mean().unwrap_or(0.0);
        v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
    }
    let n_total = (left.len() + right.len()) as f64;
    if n_total == 0.0 {
        return 0.0;
    }
    (variance_times_n(left) + variance_times_n(right)) / n_total
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

    fn _accuracy(pred: &[usize], truth: &[usize]) -> f64 {
        pred.iter().zip(truth).filter(|(p, t)| p == t).count() as f64 / truth.len() as f64
    }

    fn _mse(pred: ArrayView1<f64>, truth: ArrayView1<f64>) -> f64 {
        (&pred - &truth).pow2().sum()
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
        // DT needs enough depth to actually memorize the training data. => fewer leaves than
        // datapoints == exact match is not possble
        let mut dt = DecisionTree::new(6, 0, Task::Regression);
        let pred = dt.fit(x.view(), y.view()).predict(&x);
        println!("Ground Truth: {:?}", y);
        println!("Predictions: {:?}", pred);
        assert_eq!(_mse(pred.view(), y.view()), 0.0);
    }

    #[test]
    fn regressor_fits_linear_target() {
        let (x, y) = reg_data();
        let pred = DecisionTree::new(6, 0, Task::Regression)
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
    // use super::*;
}
