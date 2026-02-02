

2026-01-29 10:42
Status: #baby
Tags: [[Deep Learning]], [[Neural Networks Collection]], [[Artificial Intelligence]]

# Long Short-Term Memory (LSTM)
## 0. Intuition
LSTM is invented to solve the [[Vanishing/Exploding Gradients - RNN]] problems of RNN architecture, happening when their are too much data the hidden state weight might explode or vanish.
LSTM solve this by adding 2 types of memory, short-term and long-term memory interacting with each other
- for each time step t, the long term memory will be erased a bit since as time go by, the past data hold less significance -> **this stage called the FORGET GATE**. 
- Then the short-term memory will be updated using the new input data and long-term memory (so it will also have the context of the general picture from long-term memory) to then added back into long-term memory -> **this stage is called the INPUT GATE**
- After the long-term memory is updated, it will use the new long-term memory combined with the input data in that time step again to calculate the next hidden state, so that the hidden state passed down to the next cell will have the context of the input data and updated long-term memory (this part is intuitive) -> **this stage is called the OUTPUT GATE**

Sum up:

| Component         | Purpos                                                                    | Analogy                                                |
| ----------------- | ------------------------------------------------------------------------- | ------------------------------------------------------ |
| Forget Gate       | Decides what info is no longer useful and should be detected              | Clearing your "short-term memory" after a taks is done |
| Input Gate        | Decides which new information is worth storing in the Cell State          | Taking notes only on the important parts of a lecture  |
| Cell State Update | The actual modification of the long-term memory                           | Updating your mental model with the new notes          |
| Output Gate       | Decides what part of the Cell State to reveal as the Hidden State $h_{t}$ | Deciding what information to share in a presentation   |


## 1. Architecture
![[Pasted image 20260129232844.png]]

### 1.1. Terminology
Similar to [[6 - Main Notes/Recurrent Neural Network (RNN)]], except LSTM has a Cell State $C_{t}$ running through entire sequence, act as a long-term memory, and there are new weights for each gates too
Parameters:
- 
Weights:
- Input weight (the input before)


### 1.2. Forward Pass - What happened in a LSTM cell
In a single LSTM cell, instead of reusing one weight $W_{hh}$ for each input, LSTM introduce 3 gates: Forget Gate, Input Gate, Output Gate accompanied with a Cell state ($C_{t}$)
#### 1.2.1. Forget Gate
Decides how much of the previous cell state $C_{t-1}$ to keep
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$[h_{t-1}, x_{t}]$ is vector concatenation, it basically means $\begin{bmatrix}h_{t-1} \\ x_{t}\end{bmatrix}$ and $W_{f}$ is actually also a concatenated/joint matrix of $\begin{bmatrix}W_{fh} \ | \ W_{fx}\end{bmatrix}$, so the formula in its expanded form, most intuitive form is:
$$W_f \cdot [h_{t-1}, x_t] = [W_{fh} \mid W_{fx}] \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} = \underbrace{ W_{fh} \cdot h_{t-1} + W_{fx} \cdot x_t }_{  }$$
- $W_{fh}$: Weights for the hidden state
- $W_{fx}$: Weights for the input

#### 1.2.2. Input Gate
Decides what new info to add
$$
\begin{align}
i_{t} = \sigma(W_{i} \cdot [h_{t-1}, x_{t}]+b_{i}) \\
\tilde{C_{t}} = \tanh(W_{C} \cdot[h_{t-1}, x_{t}] + b_{C})
\end{align}
$$
- $\tilde{C_{t}}$: Candidate Cell State, the draft input to Cell State, not immediately saved to Cell State but have to pass through input gate first
- $i_{t}$: the input gate, acts as a filter to decides which part of the candidate state are actually important enough to keep
The Cell State new update will be the product of these 2 values

*Why use $\tanh$ instead of $\sigma$ for Candidate Cell State $\tilde{C_{t}}$:*
*Because while $\sigma$ squash the input between 0 and 1, $\tanh$ did in the range -1 and 1 allowing both positive and negateive update to Cell State*

#### 1.2.3. Cell State Update
It's the update part of the new cell state within the Input Gate after passing through the Forget Gate
$$
C_{t} =\underbrace{  f_{t}* C_{t-1} }_{ \text{after forget gate} }+\underbrace{ i_{t} *\tilde{C_{t}} }_{ \text{ the update part } }
$$
$*$ : element-wise product, means that each element in vector $f_{t}$ multiply with its according element in vector $C_{t-1}$. **This is different from dot product**

#### 1.2.4. Output Gate
Decides the next hidden state using the input, previous hidden state and updated cell state
$$
\begin{align}
o_{t} &= \sigma(W_{o} \cdot [h_{t-1}, x_{t}] + b_{o}) \\
h_{t}&=o_{t} * \tanh(C_{t})
\end{align}
$$
use $\tanh$ for the same reason above because it scale between -1 and 1

*Question may arise that why don't we caluclate the draft hidden state $\tanh(C_{t})$ by adding a Weight matrix to it but instead just input Cell state into tanh function. This is because $o_{t}$ act as a **element-wise** weight mechanism (much more expressive then just having a weight to it) already and it has its own Weight ($W_{o}$). Other sargue that the parameter in the $\tanh$ function in the Input Gate has its own weight $W_{C}$ but it is actuallyb because there are 2 vectors being input ($h_{t-1}$ and $x_{t}$) so it needs a joint weight matrix to map those 2 vectors into a same space*

### 1.3. Backpropagation
This is why LSTM solves the [[Vanishing/Exploding Gradients - RNN]] problem in RNN. Intuiviely, it is a backward pass through 4 parallel neural networks (the gates) that all merge into the cell state.

So the total error coming from the future (t+1) has 2 components unlike RNN only has $dh_{t}$:
1. $dh_{next}$: The gradient from the next hidden state
2. $dC_{next}$: The gradient from the next cell state
Intuitively, this means there are 2 "Highways" that carry information (and error) through time. At any step t, we have two types of error arriving from the future. 
> The special thing about LSTM BPTT is that instead of just having a fixed formula, it is recursive because the error of hidden state and Cell state in time step t ($C_{t}, h_{t}$) affect their value in the future ($C_{next},h_{next}$ with next starting from t+1 to T). So we must accumulate the local error of C and h along backpass way

$\delta_{element}$ represents the total error of the element or simply put the effect of loss (L) to it, including the local/immediate error + future error that it causes. For example:

#### 1.3.1. Goal Setting
So for each cell we must find the unified Weight update including 4 types of weights' gradients to perform gradient descent, including $\delta_{W} = [\delta_{W_{f}}, \delta_{W_{i}}, \delta_{W_{C}}, \delta_{W_{o}}]$
When we backpropagate, we realize that the Weight of these gates mentioned above is just the chain rule of partial derivative of the cell state, gate function, then the inside sum. For visualization, the effect of $\delta_{W}$ flows through either one of these pipeline:
$$
\begin{align}
&Loss \to \hat{y}_{t} \to h_{t} \to C_{t} \to \text{Gate Function} (f_{t}, i_{t}, \tilde{C_{t}}) \to \text{z (the inside sum)} \to [W_{f,i,C },h_{t-1}] \\

&Loss \to h_{t} \to \text{Gate Function}, (o_{t}) \to \text{z (the inside sum)} \to [W_{ o }, h_{t-1}] \\
 
&Loss \to \hat{y}_{t} \to h_{t} \to C_{t} \to C_{t-1}
\end{align}
$$
So for example, the chain rule for gradient of $W_{f}$  at time step t will folllow this error pipeline:
$$
\frac{ \partial L }{ \partial W_{f_{t}} } = \frac{ \partial L }{ \partial C_{t} } \ast \frac{ \partial C_{t}}{ \partial f_{t} } \ast \frac{ \partial f_{t} }{ \partial z_{f_{t}} } \ast\frac{ \partial z_{f_{t}} }{ \partial W_{f_{t} } } 
$$
same goes for other gradients, we just have to sum up all the gradients for all the time step than we can start optimize. We notice that in this partial derivative function, the only thing can not be calculated right away is $\frac{ \partial L }{ \partial C_{t} }$ since it accumulate the error in the future time step too. And if we expand the gradients of $W_{o}$ , the same goes for $\frac{ \partial L }{ \partial h_{t} }$. So we will represent them like this:
$$
\begin{align}
&\delta C_{t} = \frac{ \partial L }{ \partial C_{t} } + \delta C_{next} \ast f_{t} \\
&\delta h_{t} = \underbrace{ \frac{ \partial L }{ \partial h_{t} } }_{ \text{local error} } + \underbrace{ \delta h_{next}   }_{ \text{future error} }
\end{align}
$$
If we can find these 2 terms, then we can find the others.
The notation for the future error and $\delta$ of Cell state and hidden state can be a little bit confusing, but it is actually just for code convenience, the author change $\delta C_{next}$ to $\delta C_{next} \ast f_{t}$ since it's just simple multiplication but they all means the error carries from the future Cell State. What does this future error means:
> Future Error Intuition: If you look at the error/loss pipeline above, you will notice that the hidden state $h_{t-1}$ and Cell state $C_{t-1}$ error flow not only from $\hat{y}_{t-1}$, they are also from their own future version $h_{t}$ and $C_{t}$. That explains what we call the local error (the error happen in the right side of current cell), and the future error (the error happen in the left side of the next cell)


#### 1.3.2. Finding $\delta h_{t}$ and $\delta h_{t}$ (States)
We can immediately calculate the local error since the loss function has direct relation with $h_{t}$ and $h_{t}$ has direct relation with $C_{t}$
Let's say the loss function is [[Mean Squared Error (MSE]]): $L= \frac{1}{2} (y_{t}- \hat{y_{t} })^2$ (n=1 because we are calculating the local MSE on the current timestamp t). We have these functions:
$$
\begin{align}
L = \frac{1}{2} (y_{t}-\hat{y}_{t})^2 \\
\hat{y} = W_{hy}h_{t} + b_{hy} \\
h_{t} = o_{t} \ast \tanh(C_{t})
\end{align}
$$
With chain rules in partial derivatives, we can easily find that:
$$
\begin{align}
\frac{ \partial L }{ \partial h_{t} } &= W_{y}^T \cdot(\hat{y_{t}} - y_{t}) \\
\frac{ \partial L }{ \partial C_{t} } &= o_{t} \ast (1-\tanh^2(C_{t}))
\end{align}
$$
> Note that we are using $y_{t}$ and $\hat{y}_{t}$ because these are the prediction and real value locally produced in LSTM cell $t$, for the time stamp $t+1$,

So we have this formula for $\delta C_{t}$ and $\delta h_{t}$
$$
\begin{align}
\delta h_{t} &= W_{y}^T \cdot(\hat{y_{t}} - y_{t}) + \delta h_{next} \\
\delta C_{t} &= \delta h_{t} * o_{t} \ast (1-\tanh^2(C_{t})) + (\delta C_{next} * f_{t+1})
\end{align}
$$
Question may arise that why is the future error of cell state $\delta C_{next} \ast f_{t}$. It's full form is actually this chain of partial derivative, we will also expand the partial derivative form of $\delta h_{next}$ for convenient:
$$
\begin{align}
\text{Future error } C_{t} &= \frac{ \partial L }{ \partial C_{t+1} } * \frac{ \partial C_{t+1} }{ \partial C_{t} } = \frac{ \partial L }{ \partial C_{t+1} } \ast f_{t+1} \\
\text{Future error } h_{t} &= \sum_{G \in \{f,i,o,g\}} ( \underbrace{\frac{\partial L}{\partial z_{G, t+1}}}_{\delta G_{t+1}} \cdot \underbrace{\frac{\partial z_{G, t+1}}{\partial h_t}}_{W_{G, h}} ) 
\end{align}
$$
> We notice that $\delta C_{next}$ is actually represent $\frac{ \partial L }{ \partial C_{t+1} }$ but we have explained in 1.3.2 that it is just for convinience. Another thing we maybe confused about is the expansion of partial derivative of future error $h_{t}$, is is actually just a sum of all the gates gradients in time stamp t+1 where they all use $h_{t}$


#### 1.3.3. Finding $\delta f_{t}, \delta i_{t}, \delta o_{t}, \delta \tilde{C}$  (Gates)
The reason we don't directly find weight right away is because we still want to calculate the future error of the previous hidden state. These can be calculated easily by looking at the function for these gates and use chain rules for partial derivatives:
$$\begin{align} \delta o_t &= \delta h_t * \tanh(C_t) * o_t * (1 - o_t) \\ \delta f_t &= \delta C_t * C_{t-1} * f_t * (1 - f_t) \\ \delta i_t &= \delta C_t * \tilde{C}_t * i_t * (1 - i_t) \\ \delta \tilde{C}_t &= \delta C_t * i_t * (1 - \tilde{C}_t^2) \end{align}$$
And here are the fully expanded version so you can see the actual result of deriving:
$$\begin{align} \delta o_t &= \underbrace{\frac{\partial L}{\partial h_t}}_{\text{State Error}} * \underbrace{\tanh(C_t)}_{\text{Interaction}} * \underbrace{\sigma(z_{o,t})(1 - \sigma(z_{o,t}))}_{\text{Sigmoid Slope}} \\ \delta f_t &= \underbrace{\frac{\partial L}{\partial C_t}}_{\text{State Error}} * \underbrace{C_{t-1}}_{\text{Interaction}} * \underbrace{\sigma(z_{f,t})(1 - \sigma(z_{f,t}))}_{\text{Sigmoid Slope}} \\ \delta i_t &= \underbrace{\frac{\partial L}{\partial C_t}}_{\text{State Error}} * \underbrace{\tanh(z_{\tilde{C},t})}_{\text{Interaction}} * \underbrace{\sigma(z_{i,t})(1 - \sigma(z_{i,t}))}_{\text{Sigmoid Slope}} \\ \delta \tilde{C}_t &= \underbrace{\frac{\partial L}{\partial C_t}}_{\text{State Error}} * \underbrace{\sigma(z_{i,t})}_{\text{Interaction}} * \underbrace{(1 - \tanh^2(z_{\tilde{C},t}))}_{\text{Tanh Slope}} \end{align}$$
> Note that the gates deltas are actually the gradients at the pre-activation level, meaning that the chain rule goes until the sum inside the activation function: $\delta f_{t } = \frac{ \partial L }{ \partial f_{t} } * \frac{ \partial f_{t} }{ \partial z }$ with z as the sum inside the forget gate function


#### 1.3.4. Finding $dW_{f}, dW_{o}, dW_{i}, dW_{C}$ (Weights)
To find the gradient of the weight, just apply this formula:
$$dW_{G} = \frac{\partial L}{\partial W_G} = \sum_{t=1}^{T} \delta G_t \cdot q_t^T$$
with $G$ represent the gates' symbols, $G \in [f,o,i,\tilde{C}]$
This is  basically add another part in the chain rule, with $q_{t} = [h_{t-1},  x_{t}] = \frac{ \partial z }{ \partial W_{G} }$
Then just apply the usual [[Gradient Descent]] formula to update the new weights for each gates:
$$W_{new} = W_{old} - \eta \cdot \delta W$$


#### 1.3.5.  Pass down the Future error (to t-1)
As mentioned in 1.3.2, we know that we must pass down the newly recorded error. The intuition is that our current future. Here are the formulas, they are the same of what written in 1.3.2 but in more visualized format:
$$
\begin{align}
\delta z_t &= W_o^T \delta o_t + W_f^T \delta f_t + W_i^T \delta i_t + W_{\tilde{C}}^T \delta \tilde{C}_t \\
 
\delta C_{next} &\leftarrow \delta C_t
\end{align}
$$
$\delta C_{t}$ simply becomes $\delta C_{next}$ means $\frac{ \partial L }{ \partial C_{t} }$ now becomes $\frac{ \partial L }{ \partial C_{t+1} }$
And  $W_{g}^T \delta G_{t}$ for each gate is just:
$$
\underbrace{ \frac{ \partial L }{ \partial q_{t} } \cdot \frac{ \partial q_{t} }{ \partial z } }_{ \delta G_{t} } \cdot \underbrace{ \frac{ \partial z }{ \partial q_{t} } }_{ W_{G} } 
$$
We can not use $h_{t}$ directly because in the implementation, it will inside a concatenated vector $q_{t}$ as mentioned above, we just have to take the $h_{t}$ part inside it whenever we want to use it for calculation. For convinience, the pseudocode for this part is:
$$
\delta h_{next} \to \delta q_{t}[0: H]
$$
with H is hidden state dimension



## 2. Code
### 2.1. LSTM implementation
```python
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.H = hidden_size
        self.D = input_size
        self.Z_dim = self.H + self.D # Size of concatenated vector z
        
        # Initialize Weights (Using Xavier/Glorot initialization for stability)
        # We stack all 4 gates (f, i, C, o) into one big matrix for speed, 
        # or keep them separate. Here, separate for clarity matching your formulas.
        self.W_f = np.random.randn(self.H, self.Z_dim) / np.sqrt(self.Z_dim)
        self.W_i = np.random.randn(self.H, self.Z_dim) / np.sqrt(self.Z_dim)
        self.W_C = np.random.randn(self.H, self.Z_dim) / np.sqrt(self.Z_dim) # Candidate
        self.W_o = np.random.randn(self.H, self.Z_dim) / np.sqrt(self.Z_dim)
        
        # Biases
        self.b_f = np.zeros((self.H, 1))
        self.b_i = np.zeros((self.H, 1))
        self.b_C = np.zeros((self.H, 1))
        self.b_o = np.zeros((self.H, 1))
        
        # Gradients (Initialized to zero)
        self.dW_f, self.dW_i, self.dW_C, self.dW_o = 0, 0, 0, 0
        self.db_f, self.db_i, self.db_C, self.db_o = 0, 0, 0, 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_step(self, x, h_prev, C_prev):
        """
        Runs one time step forward.
        x: Input vector (D, 1)
        h_prev, C_prev: Previous states (H, 1)
        """
        # 1. Concatenate Input (z)
        z = np.row_stack((h_prev, x))
        
        # 2. Compute Gate Activations (Forward)
        f = self.sigmoid(np.dot(self.W_f, z) + self.b_f)
        i = self.sigmoid(np.dot(self.W_i, z) + self.b_i)
        C_bar = np.tanh(np.dot(self.W_C, z) + self.b_C) # Candidate
        o = self.sigmoid(np.dot(self.W_o, z) + self.b_o)
        
        # 3. Update Cell State (Memory Highway)
        C_curr = (f * C_prev) + (i * C_bar)
        
        # 4. Update Hidden State (Output Highway)
        h_curr = o * np.tanh(C_curr)
        
        # Cache results for BPTT
        cache = (z, f, i, C_bar, o, C_prev, C_curr, h_prev, x)
        
        return h_curr, C_curr, cache

    def backward_step(self, dh_next, dC_next, cache):
        """
        Runs one time step backward (BPTT).
        dh_next: Gradient of Loss w.r.t h_curr (from future + local loss)
        dC_next: Gradient of Loss w.r.t C_curr (from future)
        """
        z, f, i, C_bar, o, C_prev, C_curr, h_prev, x = cache
        
        # --- PHASE 1: State Error Accumulation ---
        # 1. Total Error at Hidden State (Already passed in as dh_next)
        
        # 2. Total Error at Cell State (Local + Future Handover)
        # Part A: Local influence from h_curr
        # Derivative of tanh(C_curr) is (1 - tanh^2(C_curr))
        dC_local = dh_next * o * (1 - np.tanh(C_curr)**2)
        
        # Part B: Total Cell Error (Sum of local and future)
        dC_curr = dC_local + dC_next

        # --- PHASE 2: Gate Deltas (Element-wise *) ---
        # 3. Output Gate Delta
        do = dh_next * np.tanh(C_curr) * o * (1 - o)
        
        # 4. Forget Gate Delta
        df = dC_curr * C_prev * f * (1 - f)
        
        # 5. Input Gate Delta
        di = dC_curr * C_bar * i * (1 - i)
        
        # 6. Candidate State Delta
        dC_bar = dC_curr * i * (1 - C_bar**2)

        # --- PHASE 3: Weight Gradients (Outer Product .) ---
        # Accumulate gradients (Step 4 in our theory)
        self.dW_f += np.dot(df, z.T)
        self.dW_i += np.dot(di, z.T)
        self.dW_C += np.dot(dC_bar, z.T)
        self.dW_o += np.dot(do, z.T)
        
        self.db_f += np.sum(df, axis=1, keepdims=True)
        self.db_i += np.sum(di, axis=1, keepdims=True)
        self.db_C += np.sum(dC_bar, axis=1, keepdims=True)
        self.db_o += np.sum(do, axis=1, keepdims=True)

        # --- PHASE 4: Error Handover to Past (Step 5 in our theory) ---
        # 1. Calculate Error w.r.t z (Combined input)
        # This sums the error from all 4 gates
        dz = (np.dot(self.W_f.T, df) + 
              np.dot(self.W_i.T, di) + 
              np.dot(self.W_C.T, dC_bar) + 
              np.dot(self.W_o.T, do))
        
        # 2. Slice dz to get error for h_prev and x
        dh_prev = dz[:self.H, :] # Top part is Hidden State
        dx = dz[self.H:, :]      # Bottom part is Input
        
        # 3. Calculate Error w.r.t C_prev (Forget Gate Handover)
        dC_prev = dC_curr * f
        
        return dh_prev, dC_prev, dx
```


### 2.2. How to run
```python
# Assume we have a list of caches from the forward pass
caches = [cache_t1, cache_t2, ..., cache_T]
loss_gradients = [dy_1, dy_2, ..., dy_T] # Gradients from the Loss function

# Initialize "Future" errors to zero for the last step
dh_next = np.zeros((hidden_size, 1))
dC_next = np.zeros((hidden_size, 1))

# Iterate Backwards
for t in reversed(range(len(caches))):
    # 1. Add the Local Loss Gradient to the error from the future
    dh_curr = loss_gradients[t] + dh_next
    
    # 2. Run one BPTT step
    dh_prev, dC_prev, dx = lstm.backward_step(dh_curr, dC_next, caches[t])
    
    # 3. Update the "Next" variables for the loop to use in the NEXT iteration (which is t-1)
    dh_next = dh_prev
    dC_next = dC_prev
```



## 3. Assignments





# References
[[Neural Network (NN)]]
[[6 - Main Notes/Recurrent Neural Network (RNN)]]
[[Transformers]]