use crate::neuron::{Neuron};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use crate::VecF;
use crate::activations::ActivationFunction;
use ndarray::Array1;

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
// Layer Trait - Common interface for layers in neural networks
// ============================================================================

// pub trait Layer {
//     fn forward(&self, input: &VecF) -> VecF;
//     fn backward(&mut self, input: &VecF, grad_output: &VecF, lr: f64) -> VecF;
// }

/// A fully connected feedforward layer
pub struct DenseLayer(pub Vec<Neuron>);

impl DenseLayer {

    /// Forward pass for the entire layer, returns activations as VecF
    pub fn forward(&self, x: &VecF) -> VecF {
        ndarray::Array1::from(self.0.iter().map(|n| n.forward(x).1).collect::<Vec<f64>>())
    }

    /// Generalized gradient application for any layer (SLP/MLP)
    /// The delta_fn closure computes delta_z for each neuron given the neuron, its activation, and its index
    /// For an SLP, delta_fn would implement the perceptron learning rule
    /// For an MLP, delta_fn would compute gradients based on backpropagation
    pub fn apply_gradients_with<F>(&mut self, input: &VecF, lr: f64, mut delta_fn: F)
    where
        F: FnMut(&Neuron, f64, usize) -> f64,
    {
        let activations: Vec<f64> = self.0.iter().map(|n| n.forward(input).1).collect();
        for (i, neuron) in self.0.iter_mut().enumerate() {
            let dz = delta_fn(neuron, activations[i], i);
            neuron.apply_gradient_descent(input, dz, lr);
        }
    }

    //Wrapper around `apply_gradients_with`` for perceptron learning rule
    pub fn apply_gradients_plr(&mut self, input: &VecF, actual_value: i8, lr: f64) {
        
        fn compute_delta_z(a: f64, actual_value: i8) -> f64 {
            let prediction = if a >= 0.0 { 1 } else { -1 };
            if prediction != actual_value {
                -(actual_value as f64)
            } else {
                0.0
            }
        }
        
        self.apply_gradients_with(input, lr, |_, a, _| compute_delta_z(a, actual_value));
    }

    /// Wrapper around `apply_gradients_with`` for MLP backpropagation
    pub fn apply_gradients_backprop(&mut self, input: &VecF, lr: f64, delta_z: &VecF) {
        fn get_delta_z(delta_z: &VecF, j: usize) -> f64 {
            delta_z[j]
        }
        self.apply_gradients_with(input, lr, |_, _, j| get_delta_z(delta_z, j));
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
        let neuron = &self.0;
        let output = neuron.predict_continuous(x);
        Array1::from(vec![output])
    }
    
    /// Predict class label:
    /// x is input vector
    /// output is 1 if activated, -1 otherwise
    fn predict(&self, x: &VecF) -> Self::Target {
        let neuron = &self.0;
        neuron.predict_step(x)
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
            let delta_z = -(actual_value as f64);
            let neuron = &mut self.0;
            neuron.apply_gradient_descent(x, delta_z, lr);
        }
    }


}

// ============================================================================
// 2. Single-Layer Perceptron (multiple neurons, multi-class classification)
// ============================================================================
pub struct SingleLayerPerceptron {
    layer: DenseLayer,
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
        SingleLayerPerceptron { layer: DenseLayer(neurons) }
    }
    
    fn forward(&self, x: &VecF) -> VecF {
        self.layer.forward(x)
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
        let mut target = vec![0.0; self.layer.0.len()];
        target[actual_value] = 1.0;
        self.layer.apply_gradients_plr(x, actual_value as i8, lr);
    }
}

// ============================================================================
// 3. Multi-Layer Perceptron (MLP) - Neural Network with Backpropagation
// ============================================================================
pub struct MultiLayerPerceptron {
    layers: Vec<DenseLayer>,
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
            layers.push(DenseLayer(layer));
        }
        MultiLayerPerceptron { layers }
    }
    
    fn forward(&self, x: &VecF) -> VecF {
        let mut current = x.clone();
        for layer in &self.layers {
            current = Array1::from(
                layer.0.iter()
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
            let (zs, as_): (Vec<_>, Vec<_>) = layer.0.iter()
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
            .zip(self.layers.last().unwrap().0.iter())
            .map(|(((a, y_true), z), neuron)| {
                let dl_da = 2.0 * (a - y_true);
                let da_dz = neuron.act.df_dz(*z, *a).unwrap_or(1.0);
                dl_da * da_dz
            })
            .collect();
        for l in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[l];
            let layer_input = &a_cache[l];
            layer.apply_gradients_backprop(layer_input, lr, &delta_z);
            if l > 0 {
                let mut next_delta_z: VecF = Array1::zeros(layer.0[0].w.len());
                for (j, neuron) in layer.0.iter().enumerate() {
                    next_delta_z = next_delta_z + neuron.input_grad(delta_z[j]);
                }
                delta_z = Array1::from(
                    next_delta_z.iter()
                        .zip(&z_cache[l-1])
                        .zip(&a_cache[l])
                        .zip(self.layers[l-1].0.iter())
                        .map(|(((&g, &z), &a), n)| g * n.act.df_dz(z, a).unwrap_or(1.0))
                        .collect::<Vec<f64>>()
                );
            }
        }
    }
}

// ============================================================================
// 4. CNN (Convolutional Neural Network) - Neural Network with Convolutional Layers.
// ============================================================================