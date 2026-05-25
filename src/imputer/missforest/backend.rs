use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::prelude::*;

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
    min_samples_leaf: usize,
}

#[derive(Clone)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    seed: u64,
}

impl RandomForest {
    pub fn new(n_trees: usize, seed: u64) -> RandomForest {
        RandomForest {
            trees: Vec::new(),
            n_trees,
            seed,
        }
    }
    pub fn fit(&mut self, data: &Array2<f64>, target: ArrayView1<f64>) -> &RandomForest {
        self.trees = Vec::with_capacity(self.n_trees);

        self
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array1<f64> {
        data.column(0).to_owned()
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
        }
    }
    fn find_best_split(labels: &[f64]) {
        let mut best_gain = f64::MIN;
        // let total_gain = self.gain(labels);
    }

    fn predict_row(&self, row: &[f64]) -> f64 {
        self.root.as_ref().unwrap().predict(row)
    }

    fn predict(&self, data: &Array2<f64>) -> Vec<f64> {
        let nrows = data.nrows();
        (0..nrows)
            .map(|i| self.predict_row(data.row(i).as_slice().expect("Slice not in mem")))
            .collect()
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn decision_tree() {
        // assert!(false, "TODO implement DecisionTrees");
    }
    #[test]
    fn random_forest() {
        // assert!(false, "TODO implement RandomForest");
    }
}
