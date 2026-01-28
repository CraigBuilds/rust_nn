
mod activations;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use crate::activations::ActivationFunction;

pub type VecF = Array1<f64>;
pub type MatF = Array2<f64>;

// ============================================================================
// Model Trait - Common interface for all neural network models
// ============================================================================
pub trait Model: Sized {
    type Target;
    type Config;
    
    fn new(config: Self::Config, rng: &mut StdRng) -> Self;
    fn forward(&self, x: &VecF) -> VecF;
    fn predict(&self, x: &VecF) -> Self::Target;
    fn train(&mut self, x: &VecF, y: Self::Target, lr: f64);
    /// Compute the loss between prediction and target
    fn loss_function(&self, prediction: &Self::Target, actual_value: &Self::Target) -> f64;
}

// ============================================================================
// Neuron Struct - Basic building block of neural networks
// ============================================================================
#[derive(Debug, Clone)]
pub struct Neuron {
    w: VecF,
    b: f64,
    act: ActivationFunction,
}

impl Neuron {

    /// Forward pass: returns (z, a) where z = w·x + b, a = act.f(z)
    /// i.e pre-activation and post-activation outputs
    /// pre-activation output is useful for computing derivatives during backpropagation
    pub fn forward(&self, x: &VecF) -> (f64, f64) {
        let z = self.w.dot(x) + self.b;
        let a = self.act.f(z);
        (z, a)
    }

    /// Update all weights and bias in place based, given delta_z, using gradient descent
    /// x is the input vector to this neuron at this forward pass
    /// delta_z (dL/dz) is the gradient of the loss (L) relative to the neuron's pre-activation output z for this input
    /// L is given by the overall model's loss function
    /// lr is the learning rate
    pub fn apply_gradient_descent(&mut self, x: &VecF, delta_z: f64, lr: f64) {
        for i in 0..self.w.len() {
            // Update each weight using gradient descent: w_i -= lr * dL/dw_i, where dL/dw_i = delta_z * x[i]
            let grad = delta_z * x[i]; // dL/dw_i, i.e partial derivative of loss with respect to weight w_i
            self.w[i] -= lr * grad; // Update weight based on learning rate and gradient
        }
        self.b -= lr * delta_z; // Update bias: b -= lr * dL/db, where dL/db = delta_z
    }

    /// Compute input gradient: dL/dx = w * delta_z
    /// delta_z is dL/dz, the gradient of the loss with respect to this neuron's pre-activation output z
    /// This is used during backpropagation to propagate gradients to previous layers
    pub fn input_grad(&self, delta_z: f64) -> VecF {
        self.w.clone() * delta_z
    }
}

// ============================================================================
// 1. Single Perceptron (one neuron, binary classification)
// ============================================================================
pub struct Perceptron(Neuron);

impl Model for Perceptron {

    type Target = i8;
    type Config = usize; // num_inputs
    
    /// Initialize a new Perceptron with random weights and bias
    fn new(num_inputs: Self::Config, rng: &mut StdRng) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let w = Array1::from((0..num_inputs).map(|_| normal.sample(rng)).collect::<Vec<f64>>());
        let b = normal.sample(rng);
        Perceptron(Neuron { w, b, act: ActivationFunction::Step })
    }
    
    /// Forward pass: returns output vector (single value for perceptron)
    fn forward(&self, x: &VecF) -> VecF {
        let (_, a) = self.0.forward(x);
        Array1::from(vec![a]) //convert a (a scalar) to VecF containing single element
    }
    
    /// Predict class label:
    /// x is input vector
    /// output is 1 if activated, -1 otherwise
    fn predict(&self, x: &VecF) -> Self::Target {
        let (_, a) = self.0.forward(x);
        if a >= 0.0 { 1 } else { -1 }
    }
    
    fn loss_function(&self, prediction: &Self::Target, actual_value: &Self::Target) -> f64 {
        if prediction == actual_value { 0.0 } else { 1.0 }
    }

    /// Train the perceptron using the Perceptron Learning Rule
    /// x is input vector
    /// actual_value is target label (1 or -1)
    /// lr is learning rate
    /// If prediction is incorrect, update weights and bias using gradient descent
    /// The gradient of the loss with respect to z is simply -actual_value for misclassified samples
    fn train(&mut self, x: &VecF, actual_value: Self::Target, lr: f64) {
        let prediction = self.predict(x);
        let loss = self.loss_function(&prediction, &actual_value);
        if loss > 0.0 {
            let delta_z = -(actual_value as f64); // Gradient of loss with respect to z for misclassified sample
            self.0.apply_gradient_descent(x, delta_z, lr);
        }
    }


}

// ============================================================================
// 2. Single-Layer Perceptron (multiple neurons, multi-class classification)
// ============================================================================
pub struct SingleLayerPerceptron {
    neurons: Vec<Neuron>,
}

impl Model for SingleLayerPerceptron {

    type Target = usize;
    type Config = (usize, usize); // (num_inputs, num_outputs)
    
    fn new(config: Self::Config, rng: &mut StdRng) -> Self {
        let (num_inputs, num_outputs) = config;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let neurons = (0..num_outputs)
            .map(|_| {
                let w = Array1::from((0..num_inputs).map(|_| normal.sample(rng)).collect::<Vec<f64>>());
                let b = normal.sample(rng);
                Neuron { w, b, act: ActivationFunction::Sigmoid }
            })
            .collect();
        SingleLayerPerceptron { neurons }
    }
    
    fn forward(&self, x: &VecF) -> VecF {
        Array1::from(
            self.neurons
                .iter()
                .map(|neuron| neuron.forward(x).1)
                .collect::<Vec<f64>>(),
        )
    }
    
    fn predict(&self, x: &VecF) -> Self::Target {
        self.forward(x)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }
    
    fn loss_function(&self, prediction: &Self::Target, actual_value: &Self::Target) -> f64 {
        if prediction == actual_value { 0.0 } else { 1.0 }
    }

    /// Train the single-layer perceptron using the Perceptron Learning Rule for multi-class classification
    /// x is input vector
    /// actual_value is target class index
    /// lr is learning rate
    fn train(&mut self, x: &VecF, actual_value: Self::Target, lr: f64) {
        // One-hot target vector
        let mut target = vec![0.0; self.neurons.len()];
        target[actual_value] = 1.0;
        // Forward pass to get activations
        let activations: Vec<f64> = self.neurons.iter().map(|n| n.forward(x).1).collect();
        // Gradient for each neuron: (a - y) * sigmoid'(z)
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let a = activations[i];
            let da_dz = a * (1.0 - a); // sigmoid derivative
            let delta_z = (a - target[i]) * da_dz;
            neuron.apply_gradient_descent(x, delta_z, lr);
        }
    }
}

// ============================================================================
// 3. Multi-Layer Perceptron (MLP) - Neural Network with Backpropagation
// ============================================================================
pub struct MultiLayerPerceptron {
    layers: Vec<Vec<Neuron>>,
}

impl Model for MultiLayerPerceptron {

    type Target = VecF;
    type Config = Vec<usize>; // layer_sizes
    
    fn new(layer_sizes: Self::Config, rng: &mut StdRng) -> Self {
        let normal = Normal::new(0.0, 0.5).unwrap();
        let mut layers = Vec::new();

        for i in 1..layer_sizes.len() {
            let num_inputs = layer_sizes[i - 1];
            let num_neurons = layer_sizes[i];
            
            let layer = (0..num_neurons)
                .map(|_| {
                    let w = Array1::from(
                        (0..num_inputs).map(|_| normal.sample(rng)).collect::<Vec<f64>>()
                    );
                    let b = normal.sample(rng);
                    Neuron { w, b, act: ActivationFunction::Sigmoid }
                })
                .collect();
            layers.push(layer);
        }
        
        MultiLayerPerceptron { layers }
    }
    
    fn forward(&self, x: &VecF) -> VecF {
        let mut current = x.clone();
        for layer in &self.layers {
            current = Array1::from(
                layer.iter()
                    .map(|neuron| neuron.forward(&current).1)
                    .collect::<Vec<f64>>()
            );
        }
        current
    }
    
    fn predict(&self, x: &VecF) -> Self::Target {
        self.forward(x)
    }
    

    fn loss_function(&self, prediction: &Self::Target, actual_value: &Self::Target) -> f64 {
        prediction.iter().zip(actual_value.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / prediction.len() as f64
    }

    fn train(&mut self, x: &VecF, actual_value: Self::Target, lr: f64) {
        // Forward pass with caching
        let mut z_cache = Vec::new();
        let mut a_cache = vec![x.clone()];
        let mut current = x.clone();
        for layer in &self.layers {
            let (zs, as_): (Vec<_>, Vec<_>) = layer.iter()
                .map(|n| n.forward(&current))
                .unzip();
            z_cache.push(Array1::from(zs));
            current = Array1::from(as_);
            a_cache.push(current.clone());
        }

        // Compute loss using loss_function
        let prediction = a_cache.last().unwrap();
        let loss = self.loss_function(prediction, &actual_value);
        if loss == 0.0 {
            return;
        }

        // Backpropagation
        let output_a = a_cache.last().unwrap();
        let output_z = z_cache.last().unwrap();
        let mut delta_z: VecF = output_a.iter()
            .zip(actual_value.iter())
            .zip(output_z.iter())
            .zip(self.layers.last().unwrap().iter())
            .map(|(((a, y_true), z), neuron)| {
                let dl_da = 2.0 * (a - y_true);
                let da_dz = neuron.act.df_dz(*z, *a).unwrap_or(1.0);
                dl_da * da_dz
            })
            .collect();
        for l in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[l];
            let layer_input = &a_cache[l];
            for (j, neuron) in layer.iter_mut().enumerate() {
                neuron.apply_gradient_descent(layer_input, delta_z[j], lr);
            }
            if l > 0 {
                let mut next_delta_z: VecF = Array1::zeros(layer[0].w.len());
                for (j, neuron) in layer.iter().enumerate() {
                    next_delta_z = next_delta_z + neuron.input_grad(delta_z[j]);
                }
                delta_z = Array1::from(
                    next_delta_z.iter()
                        .zip(&z_cache[l-1])
                        .zip(&a_cache[l])
                        .zip(&self.layers[l-1])
                        .map(|(((&g, &z), &a), n)| g * n.act.df_dz(z, a).unwrap_or(1.0))
                        .collect::<Vec<f64>>()
                );
            }
        }
    }
}

// ============================================================================
// Example usage
// ============================================================================
fn main() {
    // main left empty; see #[cfg(test)] mod tests for learning examples
}

// ============================================================================
// Tests for learning examples
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
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