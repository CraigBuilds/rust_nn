use ndarray::Array1;
type VecF = Array1<f64>;

// ============================================================================
// Activation Functions Enum - Different activation functions for neurons
// ============================================================================

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum ActivationFunction {
    Identity,
    ReLU,
    LeakyReLU { slope: f64 },
    Sigmoid,
    Tanh,
    Step, // non-differentiable (for perceptron)
}

impl ActivationFunction {
    pub fn f(self, z: f64) -> f64 {
        match self {
            ActivationFunction::Identity => z,
            ActivationFunction::ReLU => if z > 0.0 { z } else { 0.0 },
            ActivationFunction::LeakyReLU { slope } => if z > 0.0 { z } else { slope * z },
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            ActivationFunction::Tanh => z.tanh(),
            ActivationFunction::Step => if z >= 0.0 { 1.0 } else { -1.0 },
        }
    }

    // derivative d a / d z; None for Step
    pub fn df_dz(self, z: f64, a: f64) -> Option<f64> {
        match self {
            ActivationFunction::Identity => Some(1.0),
            ActivationFunction::ReLU => Some(if z > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::LeakyReLU { slope } => Some(if z > 0.0 { 1.0 } else { slope }),
            ActivationFunction::Sigmoid => Some(a * (1.0 - a)),
            ActivationFunction::Tanh => Some(1.0 - a * a),
            ActivationFunction::Step => None,
        }
    }
}

// ============================================================================
// Neuron Struct - Basic building block of neural networks
// ============================================================================
#[derive(Debug, Clone)]
pub struct Neuron {
    pub w: VecF,
    pub b: f64,
    pub act: ActivationFunction,
}

impl Neuron {
    /// Predict output for binary classification (step activation)
    pub fn predict_step(&self, x: &VecF) -> i8 {
        let (_, a) = self.forward(x);
        if a >= 0.0 { 1 } else { -1 }
    }

    /// Predict output for regression or continuous output
    pub fn predict_continuous(&self, x: &VecF) -> f64 {
        let (_, a) = self.forward(x);
        a
    }

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

    pub fn compute_delta_z(&self, a: &VecF, target: &VecF, next_delta_z: Option<&VecF>, next_layer: Option<&DenseLayer>) -> VecF {
        let mut delta_z = VecF::zeros(self.0.len());
        for (j, neuron) in self.0.iter().enumerate() {
            let (z, a_j) = neuron.forward(&Array1::zeros(neuron.w.len())); // dummy input to get z
            let da_dz = neuron.act.df_dz(z, a_j).unwrap(); // assuming non-Step activations here
            let delta = if let Some(next_dz) = next_delta_z {
                // Hidden layer
                let mut sum = 0.0;
                if let Some(next_layer) = next_layer {
                    for (k, next_neuron) in next_layer.0.iter().enumerate() {
                        sum += next_neuron.w[j] * next_dz[k];
                    }
                }
                sum * da_dz
            } else {
                // Output layer
                (a[j] - target[j]) * da_dz
            };
            delta_z[j] = delta;
        }
        delta_z
    }

}

pub struct MultiLayerPerceptron {
    pub layers: Vec<DenseLayer>,
}

impl MultiLayerPerceptron {
    /// Train the MLP using backpropagation
    pub fn train(&mut self, x: &VecF, target: &VecF, lr: f64) {
        let mut a_cache: Vec<VecF> = Vec::with_capacity(self.layers.len() + 1);
        a_cache.push(x.clone());
        
        // Forward pass (calculate and cache activations (outputs) for each layer)
        for layer in &self.layers {
            let a = layer.forward(a_cache.last().unwrap());
            a_cache.push(a);
        }
        
        //Backwards pass (compute gradients and update weights)
        let mut next_delta_z: Option<VecF> = None;
        let num_layers = self.layers.len();
        for i in (0..num_layers).rev() {
            let (left, right) = self.layers.split_at_mut(i + 1);
            let layer = &mut left[i];
            let layer_input = &a_cache[i];
            let a = &a_cache[i + 1];
            let next_layer = if i + 1 < num_layers { Some(&right[0]) } else { None };
            let delta_z = layer.compute_delta_z(a, target, next_delta_z.as_ref(), next_layer);
            layer.apply_gradients_backprop(layer_input, lr, &delta_z);
            next_delta_z = Some(delta_z);
        }
    }
}



fn main() {
    println!("This is a placeholder main function.");
}
