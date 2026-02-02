 
2026-01-29 10:41
Status: #teen
Tags: [[Deep Learning]], [[Neural Networks Collection]], [[Artificial Intelligence]]


# Recurrent Neural Network (RNN)
## 0. Intuition
RNN là một loại neural network dùng cho loại data có liên tiếp theo thời gian, giải quyết bài toán dự đoán bước tiếp theo, RNN handle được dạng data có các số lượng khác nhau, điều mà Neural Network bình thường không làm được
Disadvantages của ANN và cách RNN giải quyết:
- không đơn giản giải được bài toán map mốt số lượng input fixed ra output nên các weights và bias cũ không dùng được nếu thêm data cho ngày tiếp theo (inflexible)
-> RNN có thể xử lí flexible số lượng data vì nó chỉ đơn giản unroll thêm hidden states
- không hiểu được flow of time và momentum của data, nó treat each node as the same type of data regardless of their order sequence
-> RNN có hidden state qua từng ngày để capture momentum của data
- với ANN, mỗi data node lại có một weights và bias khác nhau, không học được gì từ các data khác
-> RNN có một weight chung để học và adjust theo cách data đang di chuyển


## 1. Architecture
![[Pasted image 20260129122348.png]]
In an RNN, we don't just* have n different nodes, we have **one** input layer and feed data into it n times (the loop). The architecture above is RNN rolled and unrolled
### 1.1. Terminology
**Input ($x_t$):** The stock price at time $t$.
**Hidden State ($h_t$):** The "memory" of the network. It stores info from days $1, \dots, t$.
**Weights:**
- $W_{xh}$: Weights connecting input to hidden state.
- $W_{hh}$: Weights connecting the _previous_ hidden state to the _current_ hidden state (this is the "recurrence").        
- $W_{hy}$: Weights connecting hidden state to the output.

### 1.2. Forward Pass
For each day/token t, the network calculates
#### 1.2.1. The Hidden State: 
$$h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$


where $\sigma$ is an activation function (like Tanh or ReLU) and b are bias terms
#### 1.2.2. The Prediction (at the final step)
$$\hat{y} = W_{hy}h_5 + b_y$$

### 1.3. Backpropagation Through Time (BPTT)
In RNN, we use BPTT. Because the same weights ($W_{hh}, W_{xh}$) are used at every time step, the graident must be summed across all time steps
#### 1.3.1. The Loss Function (MSE)
$$
L = \frac{1}{2}(\hat{y} - y)^{2}
$$
#### 1.3.2. The Gradient for $W_{hy}$ 
This is straighforward chain rule:
$$
\frac{ \partial L }{ \partial W_{hy} }=\frac{ \partial L }{ \partial \hat{y} } \cdot \frac{ \partial \hat{y} }{ \partial W_{hy} } = (\hat{y}-y) \cdot h_{T}
$$
since $\frac{ \partial \hat{y} }{ \partial x } = h_{T}$

#### 1.3.3. The Gradient for $W_{hh}$ (The "Chain" part)
This is the complex part of gradient of RNN. Since $h_{t}$ depends on $h_{t-1}$ and $h_{t-1}$ depends on $h_{t_{2}}$, etc., we apply the chain rule backwards through the sequence, understand that the high-level function relationship is $L(\hat{y}(h_{T}(h_{T-1}(h_{T-2}(\dots(h_{t}(W_{hh})))))$ to illustrate how much the loss affect the gradient $W_{hh}$ at time step t:
$$
\frac{ \partial L }{ \partial W_{hh} } =\sum_{t=1}^{T}\frac{ \partial L }{ \partial \hat{y} } \cdot \frac{ \partial \hat{y} }{ \partial h_{T} } \cdot \left( \prod_{k=t+1}^{T} \frac{ \partial h_{k} }{ \partial h_{k-1} } \right) \cdot \frac{ \partial h_{t} }{ \partial W_{hh} } 
$$
***The Summation $\left( \sum_{t=1}^{T} \right)$:***
*In an RNN, the same weight matrix $(W_{hh})$ is reused at every single time step from 1 to T, so in order to calculate the total gradient for $W_{hh}$, we must calculate how much it affected the loss at each individual time step t and **sum** those contribution together*

***The Product $\prod_{k=t+1}^{T}$:***
*This is the "Chain Rule", it's just an expansion of partial derivative for a function within a function. We learn that if $y = f(g(x))$ then $\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$*
*And as mentioned above the relationship between $L$ and $W_{hh}$ in time step t: $L(\hat{y}(h_{T}(h_{T-1}(h_{T-2}(\dots(h_{t}(W_{hh})))))$*


## 2. Code

```python
import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Hyperparameters
        self.hidden_size = hidden_size
        
        # Weight Initialization (using Xavier/Glorot-like scaling)
        # U: Input to Hidden
        self.U = np.random.randn(hidden_size, input_size) * 0.01 
        # V: Hidden to Hidden (Recurrence)
        self.V = np.random.randn(hidden_size, hidden_size) * 0.01
        # W: Hidden to Output
        self.W = np.random.randn(output_size, hidden_size) * 0.01
        
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        inputs: List of input vectors [x_1, x_2, ..., x_T]
        Returns: outputs, hidden_states
        """
        h = {}
        h[-1] = np.zeros((self.hidden_size, 1)) # Initial hidden state
        y_preds = []

        for t in range(len(inputs)):
            # h_t = tanh(U*x_t + V*h_{t-1} + bh)
            h[t] = np.tanh(np.dot(self.U, inputs[t]) + np.dot(self.V, h[t-1]) + self.bh)
            
            # y_t = W*h_t + by
            y_t = np.dot(self.W, h[t]) + self.by
            y_preds.append(y_t)
            
        return y_preds, h

    def bptt(self, inputs, targets, y_preds, h):
        """
        Backward Pass: Implementation of the summation and product formula
        """
        dU, dV, dW = np.zeros_like(self.U), np.zeros_like(self.V), np.zeros_like(self.W)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(h[0])

        # Backpropagate starting from the last time step T
        for t in reversed(range(len(inputs))):
            # 1. Error at Output: (y_pred - target)
            dy = y_preds[t] - targets[t]
            
            # 2. Gradients for W and by
            dW += np.dot(dy, h[t].T)
            dby += dy
            
            # 3. Gradient for Hidden State h[t] 
            # (Error from output + error flowing back from future time step)
            dh = np.dot(self.W.T, dy) + dh_next
            
            # 4. Backprop through Tanh nonlinearity
            dh_raw = (1 - h[t] * h[t]) * dh 
            
            # 5. Local Gradients for U, V, and bh
            dbh += dh_raw
            dU += np.dot(dh_raw, inputs[t].T)
            dV += np.dot(dh_raw, h[t-1].T)
            
            # 6. Prepare error to flow to h[t-1] (The Product Term logic)
            dh_next = np.dot(self.V.T, dh_raw)
            
        return dU, dV, dW, dbh, dby
```

## 3. Assignments
# References
[[6 - Main Notes/Long Short-Term Memory (LSTM)]]
[[Neural Network (NN)]]

