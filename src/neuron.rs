use crate::activations::ActivationFunction;
use crate::VecF;

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
