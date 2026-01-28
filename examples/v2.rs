use ndarray::Array1;
use ndarray::array;
use ndarray::s;
type VecF = Array1<f64>;

// ============================================================================
// Activation Functions Enum - Different activation functions for neurons
// ============================================================================

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum ActivationFunction {
    Identity,
    ReLU,
    LeakyReLU { slope: f64 },
    Sigmoid,
    Tanh,
    Step, // non-differentiable (for perceptron)
}

impl ActivationFunction {
    fn f(self, z: f64) -> f64 {
        match self {
            ActivationFunction::Identity => z,
            ActivationFunction::ReLU => if z > 0.0 { z } else { 0.0 },
            ActivationFunction::LeakyReLU { slope } => if z > 0.0 { z } else { slope * z },
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            ActivationFunction::Tanh => z.tanh(),
            ActivationFunction::Step => if z >= 0.0 { 1.0 } else { -1.0 },
        }
    }

    // gradient of this function (None for Step)
    fn df_dz(self, z: f64, actual_output: f64) -> Option<f64> {
        match self {
            ActivationFunction::Identity => Some(1.0),
            ActivationFunction::ReLU => Some(if z > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::LeakyReLU { slope } => Some(if z > 0.0 { 1.0 } else { slope }),
            ActivationFunction::Sigmoid => Some(actual_output * (1.0 - actual_output)),
            ActivationFunction::Tanh => Some(1.0 - actual_output * actual_output),
            ActivationFunction::Step => None,
        }
    }
}

// ============================================================================
// Neuron Struct - Basic building block of neural networks
// ============================================================================
#[derive(Debug, Clone)]
struct Neuron {
    w: VecF,
    b: f64,
    act: ActivationFunction,
}

impl Neuron {

    /// Forward pass: returns (z, actual_output) where z = w·x + b, actual_output = act.f(z)
    /// param x: input vector to the neuron
    /// return: (z, actual_output) i.e pre-activation and post-activation outputs
    /// pre-activation output is useful for computing derivatives during backpropagation
    fn forward(&self, x: &VecF) -> (f64, f64) {
        let z = self.w.dot(x) + self.b;
        let actual_output = self.act.f(z);
        (z, actual_output)
    }

    /// Update all weights and bias in place based, given delta_z, using gradient descent
    /// x is the input vector to this neuron at this forward pass
    /// delta_z (dL/dz) is the gradient of the loss (L) relative to the neuron's pre-activation output z for this input
    /// why z (pre-activation)? Because z is the input to the activation function (computed from the weights)
    /// so z is the thing that directly depends on the weights, and we need dL/dz to compute dL/dw and dL/db
    /// L is given by the overall model's loss function
    fn apply_gradient_descent(&mut self, x: &VecF, delta_z: f64, learning_rate: f64) {
        for i in 0..self.w.len() {
            // Update each weight using gradient descent: w_i -= lr * dL/dw_i, where dL/dw_i = delta_z * x[i]
            let delta_w_i = delta_z * x[i]; // dL/dw_i, i.e partial derivative of loss with respect to weight w_i
            self.w[i] -= learning_rate * delta_w_i; // Update weight based on learning rate and gradient
        }
        let delta_b = delta_z; // dL/db = delta_z
        self.b -= learning_rate * delta_b; // Update bias: b -= lr * dL/db, where dL/db = delta_z
    }
}

// ============================================================================
// Dense Layer Struct - A fully connected layer of neurons
// ============================================================================
struct DenseLayer(Vec<Neuron>);

impl DenseLayer {

    /// Forward pass for the entire layer, returns activations as VecF
    fn forward(&self, x: &VecF) -> VecF {
        self.0.iter().map(|n| n.forward(x).1).collect()
    }

    /// This is the core backpropagation step for the layer.
    /// Compute vector of delta_z for the layer given activations (actual outputs) and targets (expected outputs), or next layer's delta_z
    /// For output layer, expected_output must be provided; for hidden layers, next_delta_z and next_layer must be provided
    /// param actual_output: activations of this layer
    /// param expected_output: expected outputs for this layer (only for output layer)
    /// param next_delta_z: delta_z vector from the next layer (only for hidden layers)
    /// param next_layer: reference to the next layer (only for hidden layers)
    /// param input: input vector to this layer during forward pass
    /// return: vector of delta_z for this layer
    fn compute_gradients(&self, actual_output: &VecF, expected_output: &VecF, next_delta_z: Option<&VecF>, next_layer: Option<&DenseLayer>, input: &VecF) -> VecF {
        let mut delta_z_vec = VecF::zeros(self.0.len()); // Initialize delta_z vector
        for (j, neuron) in self.0.iter().enumerate() {
            let (z, actual_output_j) = neuron.forward(input); //pre and post activation outputs
            let da_dz = neuron.act.df_dz(z, actual_output_j).unwrap(); // derivative of activation function (assuming non-Step activations here)
            let delta_z = if next_delta_z.is_none() {
                // Output layer is simple - delta_z is just the (actual_output minus expected_output) times derivative of activation function
                (actual_output[j] - expected_output[j]) * da_dz
            } else {
                // Hidden layer is more complicated - delta_z depends on next layer's weights and delta_z
                let mut sum = 0.0;
                let next_layer = next_layer.unwrap();
                let next_dz = next_delta_z.unwrap();
                for (k, next_neuron) in next_layer.0.iter().enumerate() {
                    sum += next_neuron.w[j] * next_dz[k];
                }
                sum * da_dz
            };
            delta_z_vec[j] = delta_z;
        }
        delta_z_vec
    }

    /// Apply gradients to all neurons in the layer using the provided delta_z vector.
    /// Each neuron's weights and bias are updated using gradient descent.
    fn apply_gradient_descent(&mut self, input: &VecF, lr: f64, delta_z_vec: &VecF) {
        for (neuron, &delta_z) in self.0.iter_mut().zip(delta_z_vec.iter()) {
            neuron.apply_gradient_descent(input, delta_z, lr);
        }
    }
}

// ============================================================================
// Multi-Layer Perceptron Struct - A simple feedforward neural network
// ============================================================================

struct MultiLayerPerceptron {
    layers: Vec<DenseLayer>,
}

impl MultiLayerPerceptron {
    /// Train the MLP using backpropagation
    fn train(&mut self, x: &VecF, expected_output: &VecF, lr: f64) {
        
        // Initialize caches for activations
        let mut actual_output_cache: Vec<VecF> = Vec::with_capacity(self.layers.len() + 1);
        actual_output_cache.push(x.clone());
        
        // Forward pass (calculate and cache activations (outputs) for each layer)
        for layer in &self.layers {
            let actual_output = layer.forward(actual_output_cache.last().unwrap());
            actual_output_cache.push(actual_output);
        }
        
        //Backwards pass (compute gradients and update weights)
        let mut next_delta_z_vec: Option<VecF> = None;
        let num_layers = self.layers.len();
        for i in (0..num_layers).rev() {
            let (left, right) = self.layers.split_at_mut(i + 1); //enable mutable borrow of one layer while having immutable access to others
            let layer = &mut left[i];
            let layer_input = &actual_output_cache[i];
            let actual_output = &actual_output_cache[i + 1];
            let next_layer = if i + 1 < num_layers { Some(&right[0]) } else { None };

            let delta_z_vec = layer.compute_gradients(actual_output, expected_output, next_delta_z_vec.as_ref(), next_layer, layer_input);
            layer.apply_gradient_descent(layer_input, lr, &delta_z_vec);
            
            next_delta_z_vec = Some(delta_z_vec);
        }
    }
}

// ============================================================================
// Convolutional Layer Struct - not fully connected; uses local connections and shared filters
// ============================================================================
struct ConvolutionalLayer1D(Vec<Neuron>);
impl ConvolutionalLayer1D {
    /// Forward pass for convolutional layer
    /// Each neuron acts as a filter sliding over the input vector (1D convolution)
    /// Output length = input length - filter length + 1
    fn forward(&self, x: &VecF) -> VecF {
        let filter_len = self.0[0].w.len();
        let input_len = x.len();
        let output_len = input_len.saturating_sub(filter_len) + 1;
        let mut output = VecF::zeros(output_len);

        for i in 0..output_len {
            // For each position, create a slice of the input
            let x_slice = x.slice(s![i..i + filter_len]);
            // Each neuron is a filter; apply all filters and sum their outputs
            let mut sum = 0.0;
            for neuron in &self.0 {
                let (_z, actual_output) = neuron.forward(&x_slice.to_owned());
                sum += actual_output;
            }
            output[i] = sum;
        }
        output
    }

    /// Compute gradients (delta_z) for the convolutional layer.
    /// For output layer: expected_output must be provided, next_delta_z/next_layer are None.
    /// For hidden layer: next_delta_z and next_layer must be provided, expected_output is ignored.
    /// - actual_output: activations of this layer
    /// - expected_output: expected outputs for this layer (only for output layer)
    /// - next_delta_z: delta_z vector from the next layer (only for hidden layers)
    /// - next_layer: reference to the next layer (only for hidden layers)
    /// - input: input vector to this layer during forward pass
    /// Returns: vector of delta_z for this layer (length = number of neurons)
    fn compute_gradients(
        &self,
        actual_output: &VecF,
        expected_output: &VecF,
        next_delta_z: Option<&VecF>,
        next_layer: Option<&ConvolutionalLayer1D>,
        input: &VecF,
    ) -> VecF {
        let mut delta_z_vec = VecF::zeros(self.0.len());
        for (j, neuron) in self.0.iter().enumerate() {
            // For convolution, each neuron slides over the input, so we need to aggregate gradients over all positions
            // Here, we assume each neuron is a filter and output is sum of activations over all positions
            // We'll compute the gradient for the filter as the sum of gradients over all positions

            // For each position in the input where this filter applies:
            let filter_len = neuron.w.len();
            let input_len = input.len();
            let output_len = input_len.saturating_sub(filter_len) + 1;
            let mut grad_sum = 0.0;

            for i in 0..output_len {
                let x_slice = input.slice(s![i..i + filter_len]);
                let (z, actual_output_j) = neuron.forward(&x_slice.to_owned());
                let da_dz = neuron.act.df_dz(z, actual_output_j).unwrap();

                let delta_z = if next_delta_z.is_none() {
                    // Output layer: delta_z = (actual_output - expected_output) * da/dz
                    (actual_output[i] - expected_output[i]) * da_dz
                } else {
                    // Hidden layer: sum over next layer's weights * next_delta_z, times da/dz
                    let mut sum = 0.0;
                    let next_layer = next_layer.unwrap();
                    let next_dz = next_delta_z.unwrap();
                    for (k, next_neuron) in next_layer.0.iter().enumerate() {
                        // Each next_neuron's filter overlaps with this neuron's output at some positions
                        // For simplicity, assume full connection (like dense), or you can implement proper convolutional backprop
                        // Here, we just sum the product of the corresponding weight and next_delta_z
                        if j < next_neuron.w.len() {
                            sum += next_neuron.w[j] * next_dz[k];
                        }
                    }
                    sum * da_dz
                };
                grad_sum += delta_z;
            }
            delta_z_vec[j] = grad_sum;
        }
        delta_z_vec
    }

    fn apply_gradient_descent(&mut self, input: &VecF, lr: f64, delta_z_vec: &VecF) {
        for (neuron, &delta_z) in self.0.iter_mut().zip(delta_z_vec.iter()) {
            neuron.apply_gradient_descent(input, delta_z, lr);
        }
    }
}

// ============================================================================
// Convolutional Neural Network Struct - A simple CNN
// ============================================================================
struct ConvolutionalNeuralNetwork {
    layers: Vec<ConvolutionalLayer1D>,
}

impl ConvolutionalNeuralNetwork {

    fn forward(&self, x: &VecF) -> VecF {
        self.layers.iter().fold(x.clone(), |a, layer| layer.forward(&a))
    }

    /// Train the CNN using backpropagation
    fn train(&mut self, x: &VecF, expected_output: &VecF, lr: f64) {
        // Initialize caches for activations
        let mut actual_output_cache: Vec<VecF> = Vec::with_capacity(self.layers.len() + 1);
        actual_output_cache.push(x.clone());

        // Forward pass (calculate and cache activations (outputs) for each layer)
        for layer in &self.layers {
            let actual_output = layer.forward(actual_output_cache.last().unwrap());
            actual_output_cache.push(actual_output);
        }

        //Backwards pass (compute gradients and update weights)
        let mut next_delta_z_vec: Option<VecF> = None;
        let num_layers = self.layers.len();
        for i in (0..num_layers).rev() {
            let (left, right) = self.layers.split_at_mut(i + 1);
            let layer = &mut left[i];
            let layer_input = &actual_output_cache[i];
            let actual_output = &actual_output_cache[i + 1];
            let next_layer = if i + 1 < num_layers { Some(&right[0]) } else { None };

            let delta_z_vec = layer.compute_gradients(actual_output, expected_output, next_delta_z_vec.as_ref(), next_layer, layer_input);
            layer.apply_gradient_descent(layer_input, lr, &delta_z_vec);

            next_delta_z_vec = Some(delta_z_vec);
        }
    }
}

fn main() {
    // Example: XOR problem (2 inputs, 1 output)
    // XOR truth table:
    // x1 x2 | y
    // 0  0  | 0
    // 0  1  | 1
    // 1  0  | 1
    // 1  1  | 0


    // Training data
    let inputs = vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![1.0, 1.0],
    ];
    let expected_outputs = vec![
        array![0.0],
        array![1.0],
        array![1.0],
        array![0.0],
    ];

    // Build a simple MLP: 2 inputs -> 2 hidden (tanh) -> 1 output (sigmoid)
    let mut mlp = MultiLayerPerceptron {
        layers: vec![
            DenseLayer(vec![
                Neuron {
                    w: array![0.5, -0.5],
                    b: 0.0,
                    act: ActivationFunction::Tanh,
                },
                Neuron {
                    w: array![-0.5, 0.5],
                    b: 0.0,
                    act: ActivationFunction::Tanh,
                },
            ]),
            DenseLayer(vec![
                Neuron {
                    w: array![0.5, 0.5],
                    b: 0.0,
                    act: ActivationFunction::Sigmoid,
                },
            ]),
        ],
    };

    // Training loop
    let lr = 0.1;
    for epoch in 0..10000 {
        for (x, y) in inputs.iter().zip(expected_outputs.iter()) {
            mlp.train(x, y, lr);
        }
        // Optionally print loss every 1000 epochs
        if epoch % 1000 == 0 {
            let loss: f64 = inputs.iter().zip(expected_outputs.iter())
                .map(|(x, y)| {
                    let actual_output = mlp.layers.iter().fold(x.clone(), |a, layer| layer.forward(&a));
                    (actual_output[0] - y[0]).powi(2)
                })
                .sum::<f64>() / 4.0;
            println!("Epoch {epoch}, loss: {loss:.4}");
        }
    }

    // Test predictions
    println!("Trained MLP predictions for XOR:");
    for (x, y) in inputs.iter().zip(expected_outputs.iter()) {
        let actual_output = mlp.layers.iter().fold(x.clone(), |a, layer| layer.forward(&a));
        println!("Input: {:?}, Target: {}, Predicted: {:.3}", x, y[0], actual_output[0]);
    }
}
