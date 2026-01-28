mod activations;
mod neuron;
mod models;
type VecF = ndarray::Array1<f64>;
use models::{Model, Perceptron, SingleLayerPerceptron, MultiLayerPerceptron};
use rand::SeedableRng;

// ============================================================================
// Example usage
// ============================================================================
fn main() {
    //perceptron example
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut p = Perceptron::new(2, &mut rng);
    let and_data = vec![
        (ndarray::array![0.0, 0.0], -1),
        (ndarray::array![0.0, 1.0], -1),
        (ndarray::array![1.0, 0.0], -1),
        (ndarray::array![1.0, 1.0], 1),
    ];
    for _ in 0..20 {
        for (x, y) in &and_data {
            p.train(x, *y, 0.1);
        }
    }
    for (x, y) in &and_data {
        let pred = p.predict(x);
        println!("Perceptron AND: input={:?}, target={}, got={}", x, y, pred);
    }
    //single layer perceptron example
    let mut slp = SingleLayerPerceptron::new((3, 3), &mut rng);
    let slp_data = vec![
        (ndarray::array![1.0, 0.0, 0.0], 0),
        (ndarray::array![0.0, 1.0, 0.0], 1),
        (ndarray::array![0.0, 0.0, 1.0], 2),
    ];
    use rand::seq::SliceRandom;
    let mut data = slp_data.clone();
    for _ in 0..200 {
        data.shuffle(&mut rng);
        for (x, y) in &data {
            slp.train(x, *y, 0.1);
        }
    }
    for (x, y) in &slp_data {
        let pred = slp.predict(x);
        println!("Single Layer Perceptron Identity: input={:?}, target={}, got={}", x, y, pred);
    }
    //multi-layer perceptron example
    let mut mlp = MultiLayerPerceptron::new(vec![2, 4, 1], &mut rng);
    let xor_data = vec![
        (ndarray::array![0.0, 0.0], ndarray::array![0.0]),
        (ndarray::array![0.0, 1.0], ndarray::array![1.0]),
        (ndarray::array![1.0, 0.0], ndarray::array![1.0]),
        (ndarray::array![1.0, 1.0], ndarray::array![0.0]),
    ];
    for _ in 0..5000 {
        for (x, y) in &xor_data {
            mlp.train(x, y.clone(), 0.1);
        }
    }
    for (x, y) in &xor_data {
        let pred = mlp.predict(x);
        println!("Multi-Layer Perceptron XOR: input={:?}, target={}, got={}", x, y[0], pred[0]);
    }
}

// ============================================================================
// Tests for learning examples
// ============================================================================

#[cfg(test)]
mod tests {
    use super::models::{Perceptron, SingleLayerPerceptron, MultiLayerPerceptron, Model};
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn perceptron_learns_and_gate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut p = Perceptron::new(2, &mut rng);
        let and_data = vec![
            (array![0.0, 0.0], -1),
            (array![0.0, 1.0], -1),
            (array![1.0, 0.0], -1),
            (array![1.0, 1.0], 1),
        ];
        for _ in 0..20 {
            for (x, y) in &and_data {
                p.train(x, *y, 0.1);
            }
        }
        for (x, y) in &and_data {
            let pred = p.predict(x);
            assert_eq!(pred, *y, "Perceptron failed AND: input={:?}", x);
        }
    }

    #[test]
    fn single_layer_perceptron_learns_identity() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut slp = SingleLayerPerceptron::new((3, 3), &mut rng);
        let slp_data = vec![
            (array![1.0, 0.0, 0.0], 0),
            (array![0.0, 1.0, 0.0], 1),
            (array![0.0, 0.0, 1.0], 2),
        ];
        use rand::seq::SliceRandom;
        let mut data = slp_data.clone();
        for _ in 0..200 {
            data.shuffle(&mut rng);
            for (x, y) in &data {
                slp.train(x, *y, 0.1);
            }
        }
        for (x, y) in &slp_data {
            let pred = slp.predict(x);
            assert_eq!(pred, *y, "SLP failed identity: input={:?}", x);
        }
    }

    #[test]
    fn mlp_learns_xor_gate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut mlp = MultiLayerPerceptron::new(vec![2, 4, 1], &mut rng);
        let xor_data = vec![
            (array![0.0, 0.0], array![0.0]),
            (array![0.0, 1.0], array![1.0]),
            (array![1.0, 0.0], array![1.0]),
            (array![1.0, 1.0], array![0.0]),
        ];
        for _ in 0..5000 {
            for (x, y) in &xor_data {
                mlp.train(x, y.clone(), 0.1);
            }
        }
        for (x, y) in &xor_data {
            let pred = mlp.predict(x);
            let close = (pred[0] - y[0]).abs() < 0.2;
            assert!(close, "MLP failed XOR: input={:?}, target={}, got={}", x, y[0], pred[0]);
        }
    }
}