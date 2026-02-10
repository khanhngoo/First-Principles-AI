2026-02-10 16:12
Status: #baby 
Tags: [[Deep Learning]], [[Transformers]]

# Word2Vec - Continuous bag of words (CBOW)

## 0. Intuition
Basically a nother version of Word2Vec besides skip gram, this time we use the embedding of contex vectors to find the embedding of a word

Imagine the sentence: "The quick brown fox jumps". We want to predict "fox" (Center) using "quick", "brown", "jumps" (Context).

- **Input:** We take the one-hot vectors for all context words ($x_{quick}, x_{brown}, x_{jumps}$).
- **Projection (Hidden Layer):** We look up their embeddings in $W$ and **AVERAGE** them.
    - This is why it's a "Bag" of words. The order doesn't matter.
    - "Quick brown jumps" produces the exact same average as "Jumps quick brown".
- **Output:** We use this **averaged vector** ($h$) to predict the center word via Softmax.

## 1. Architecture
![[Pasted image 20260210165444.png]]


### 1.1. Terminology
Revise the terminology in [[Word2Vec - Skip Gram]]

### 1.2. Forward Pass
This basically looks like a reverse of skip gram model

We must first understand how the output is computed. The first step is to evaluate the output of hidden layer $\mathbf{h}$. This is computed by averaging the one-hot encoded inputs
$$
h = \frac{1}{C} \cdot W_{V\times N} \cdot \sum_{i=1}^{C}x_{i} 
$$
This means we take the one hot encoded input, then sum them together so we get something like $[1,1,1,0, 0,\dots,1]$ with 1 in the index of the context word one-hot encoded value ($V$-dimension) then multiply it with $W_{V\times N}$ and divide it with C so we have a hidden layer of $1\times N$ dimension.

One way to visualize this is to consider each row of the Weight matrix an embedding of the context word. And the role of the summation is to find out what are the row/word in context that we should pull out to predict the center/target word. For example:
$$
\begin{bmatrix}
1 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
0.1 & 2.0 & 0.5 & 4.0 \\
2.0 & 4.0 & 0.8 & 0.0 \\
0.0 & -2.0 & 0.1 & 6.0
\end{bmatrix}
$$
According to matrix multiplication, the result of this operation pull out row 1 and 3 and sum them up, which 1 and 3 is also the value of the context words being one-hot encoded. This way of thinking is much more intuitive, so some have written it like this
$$h = \frac{1}{C} \sum_{i \in \text{context}} v_{w_i}$$
Some implementation use Sum instead of Mean, but Mean is standard

Then we can move to next part, where we calculate the score $u$ for the Center word
$$
u = h \cdot W_{N\times V}'
$$
This process produces scores for the entire library. Then we apply softmax to calculate the probability of the center word then the loss function using [[Categorical Cross-Entrophy (CCE)]] exactly like in [[Word2Vec - Skip Gram]]
$$
\begin{align}
y &= Softmax(u) \\
L &= -\log(\hat{y}_{center})
\end{align}
$$
Note that in the Skip Gram model, the loss function result will be in a vector form since it accounts for every other context words in context window being predicted, but in CBOW, the loss function will only be a scalar value as the loss being predicted by comparing the center word prediction to its ground truth, specifically a vector with most column filled with 0 and only the loss value in the center word column because that's hwo CCE works to prevent the other gradient be affected

Technically the Loss function will look like this
$$L = - \sum_{j=1}^V y_j \log(\hat{y}_{j})$$
but because the ground truth vector one-hot encoded (filled with 0 for every other word index except the center word), so it basically conver tto the above formula
the output error is


### 1.3. Backpropagation
The backpropagation of CBOW is slightly different because of the averaging
#### 1.3.1. Gradient w.r.t $W'\left( \frac{ \partial L }{ \partial W' } \right)$
This is quite similar to the gradient of predicting each context word in skip gram, we just don't have to index with $j$ since this is only value we are predicting
We have:
$$\frac{\partial L}{\partial u} = \hat{y} - y$$
Let's call this error vector $e$ ($V \times 1$)
The Gradient w.r.t Output Matrix ($W'$) will be
$$\frac{\partial E}{\partial W'} = h \cdot e^T$$
This is quite similar to skip gram with $\hat{y}$ here represent $\hat{y}_{j}$ or $y$ rep $y_{j}$ in Skip Gram, corresponding to the $j_{th}$ context words gradients


#### 1.3.2. Gradient w.r.t $W\left( \frac{ \partial L }{ \partial W } \right)$
Similar to Skip Gram, the error must pass h first, before it reach $W$ so we calculate gradient w.r.t h
$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial h}$$
Since $u=h\cdot W'$, the derivative is $W'$
$$
\mathbf{EH} = W' \cdot (\hat{y} - y)
$$
This is basically the same as Skip Gram, this may be quite confused we use $W'$ instead of summation but it is basically the same due to matrix multiplication

Now it's the different part, unlike in Skip Gram, the hidden value is just grabbed from the centered word index row in $W$ so it only affect that row. In the calculation of hidden state in CBOW, every row is invovled, or every context word is invovled, the formula for hidden state is
$$h = \frac{1}{C} (v_{x_1} + v_{x_2} + \dots + v_{x_C})$$
Since this is a simple linear sum, the derivative of $h$ w.r.t any single input vector $v_{x_{i}}$ is just the constant scaling factor $\frac{1}{C}$
$$
\frac{ \partial h }{ \partial v_{x_{i}} } = \frac{1}{C}
$$
So we have the gradient for each input vectors or each row in the $W$ like this
$$
\begin{align}
\frac{\partial E}{\partial v_{x_i}} &= \frac{\partial E}{\partial h} \cdot \frac{\partial h}{\partial v_{x_i}} \\
\frac{\partial E}{\partial v_{x_i}} &= \mathbf{EH} \cdot \frac{1}{C}
\end{align}
$$
*Intuitively, this means every word in the context window receives the **exact same error vector**, scaled down by the window size.*
*If "drink" and "coffee" were the context, and the model guessed "cold" instead of "hot," both "drink" and "coffee" get "blamed" equally for leading the model astray.*


#### 1.3.3. Update Equations (SGD)
**Update Rule for $W'$**
$$W'_{\text{new}} = W'_{\text{old}} - \eta \cdot (h \cdot (\hat{y} - y)^T)$$
**Update Rule for $W$**
$$v_{w_{new}} \leftarrow v_{w_{old}} - \eta \cdot \frac{1}{C} \cdot \mathbf{EH}$$
with $v_{w}$ represent every row in $W$

One note is that considering the result of Word2Vec, we often use $W$ as the words embedding for standard, but $W'$ is still valuable so in paper like GloVE,  researchers use the average of $W$ and $W'$


# References
