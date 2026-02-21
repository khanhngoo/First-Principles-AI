2026-02-09 19:18
Status: #baby 
Tags: [[Deep Learning]], [[Neural Networks Collection]]

# Word2Vec - Skip Gram

## 0. Intuition
Skip Gram use the input word (often in the middle) and to predict C/2 context words in left and right wing. The model one-hot encoded the input and ground truth, go through a layer of weight matrix to get the target word embedding and multiply by the output weight matrix (or the embedding of context vector) to produce the score $u$ for every word. Then use a softmax function to get the probability of the word happening, compare with the one-hot encoded ground truth to readjust the Weights again and again
> Input Weight Matrix $W_{I}$ and output Weight Matrix $W_{O}$ are shared as we go through word by word, then we can have optimal Weight matrices after it going through all the word in the text

There is this intuition that we are doing a fake prediction task using this neural network. We are not trying to predict the words but rather use gradient descent to produce error signal to pull the vector to its right place represent the right semantic meaning for each word

One common confusion is that there are 2 embeddings resides in 2 weight matrices
- $W_{V\times N}$ is the actual embeddings of words
- $W_{N\times V}'$ is more like a contextual neighbors signals pool to guide the prediction
Often times we will remove $W_{N\times V}'$ and only keep $W_{V\times N}$ to represent the embeddings of words

Why is this: this is because the vectors of the $W_{N\times V}'$ is more blurred or more close together since its update account for every vector in the pool so it can not produce a definite vector that can distinct itself from other words. On the other hand, in $W_{V\times N}$ we often zeroed out non-input vectors so we can solely update the embedding of 1 word while also adjust based on the context of other word by backpropagating through the contextual signal pools $W_{N\times V}'$


## 1. Architecture

![[Pasted image 20260207163604.png]]

### 1.1. Terminology
- $x_{k}$ is the $k^{th}$ word being one-hot encoded
- $V$ is the dimension of $x_{k}$ being one-hot encoded -> $V = \text{number of words}$
- $N$ is the dimension of the hidden layer, or the embedded size
- $W_{V\times N}$ is the Input weight matrix
- $h_{i}$ is the hidden state, or can be seen as the embedding of the target word, later you will understand why
- $W_{N\times V}'$ is the Output weight matrix, to then use to give scores to each word using the target word embedding and context word embedding (you will understand a moment later)

### 1.2. Feedforward Pass

#### 1.2.1. Embedding
So it is actually very intuitive. Let's make an example throughout the deep dive so you can easily visualize. Let's say the number of words in a dictionary is 10,000 ($V = 10,000$) and the dimensions of the embedding layer is 300 ($N = 300$)

Then we have $x_{k}$ like this:
$$
\underbrace{ [0, 0,1,0,0,\dots,0] }_{ 10,000 \text{ elements} }
$$
with 1 being the value represent the unique position of the word $w_{k}$ in the one-hot encoded array

Then it will multiply with Input Weight Matrix $W_{V\times N}$ to capture the embedding of this word. But there is a very convient thing happen when you do this, when we perform matrix multiplication:
$$
x_{k} \cdot W_{V \times N}
$$
Then the result will simply be the $k_{th}$ row in $W_{V \times N}$. For example
$$
\begin{bmatrix}
0 & 0 & 1 & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 2 & 3 \\
3 & 4 & 5 \\
7 & 5 & 2 \\
2 & 0 & 2
\end{bmatrix}
=
\underbrace{ \begin{bmatrix}
7 & 5 & 2
\end{bmatrix} }_{ \text{the third row} }
$$
As you see the result will be the $j^{th}$ row in the Weight Matrix $W_{V\times N}$ if $j^{th}$ is the index of 1 in the one-hot encoded input. So often people would for convenient understand that the vector $v_{I}$ as the extracted row vector in the mentioned position as the embedding vector of the input word instead of strictly considering the hidden state. This will be very convinient to think about when we look at the Output Weight Matrix $W_{N\times V}'$ where each column vector is considered as the word context embedding.
> Note that in real implementation, every other row in $W_{V\times N}$ will be zeroed out since they don't carry meaning in that step, and to prevent the gradient descent affect other embedding vector

#### 1.2.2. Scoring
After we have the embedded/hidden layer, we will use another Weight matrix to predict the score of each word likely to be near the input word given the position of the input word. In general, the equation will be like this:
$$
\underbrace{ u }_{ 1\times V } = \underbrace{ h.W' }_{ (1\times N)\times(N\times V) }
$$
$u$ is the list of scores of each word, you can see here that $u$ has the dimension of $1\times V$ meaning each element in the list $u_{j}$ represent the score of a word in a list of $V$ number of words.

You can understand that when we perform matrix multiplication, the vector row of hidden state will multiply by the $j^{th}$ vector column in the output weight matrix to calculate the score for that word to appear near the input word, let's call it $u_{j}$. Intuitively, this is the same as the embedding of the input target word $(v_{I})$ multiply by the emebedding of context word ($v_{C}$) by considering $v_{I}$ as the non-zero row vector in $W_{V\times N}$ and $v_{C}$ as any of the column vectors in $W_{N\times V}'$.

We got an 'aha' moment here to understand why we are using a prediction model like Neural Network to do something like embedding words to numbers. It is because we are utilizing one of the sub-task of Neural Network, not predicting but adjusting the Weights based on the ground truths, which is basically adjusting the embeddings of words given new evidence

*As you noticed here is that the neural network we use don't use activation function in the hidden layer. This is because:*
- *We are not detecting features or complex patterns in the hidden layer, it is just a "look-up  row" for the Input weight matrix. Adding an activation function like Sigmoid, would squeeze the embedding vector to too close 0 and 1, destroying the embedding distinctive representation, makes it semantic significance way less. In this case, the activation function would destroy the complex pattern instead of detecting one*
- *The model is linear by design, the goal of the Word2Vec is to create a embedding space where linear relationship holds true. For example, we want*
*$$
king - man + woman \approx  queen
$$
Adding the activation would negate this linear affect on the embeddings vectors
- *It require less computation, adding the backpropagation would have heavy burden on both forward and backpass computing process*


#### 1.2.3. Error catching
Now we have the prediction scores, we want to find the error so we can adjust the the weights. For this we will utilize a combination of [[Softmax]] and [[Categorical Cross-Entrophy (CCE)]]. They are used together because their mathematical relationship is incredibly elegant, we will see when we get to the gradient computation part. Matter of fact, this combination is very common in Machine Learning field as the loss function for multiclass classification problem. Indeed, our fake task we set for this Word2Vec Neural Network is a multiclass classification problem, predicting the right word (class) among multiple words

First the scores, or researchers call them logits, will be put into a softmax function to output a probability list summing up to 1 using [[Softmax]]
$$
\hat{y}_{j} = \frac{\exp(u_{j})}{\sum_{k=1}^{V} \exp(u_{k})}
$$
As $\hat{y}$ is the list of predicted probabilities and $u_{j}$ is the scores for the $j^{th}$ word. We can understand $\hat{y}_{j}$ as the ppredicted probability for class $j$

Then we apply a Categorical Cross-Entrophy, where it penalizes the model when it assigns low confidence to the correct class
$$
L(y, \hat{y}) = - \sum_{j=1}^{C} y_{j}\log(\hat{y}_{j}) 
$$
where
- $L(y,\hat{y})$: Categorical Cross-Entrophy Loss
- $y_{j}$: True label for class j
- $\hat{y}_{j}$: predicted probability for class j
- $C$: number of class in the context windowes (since we are only predicting the words within context windows)
Since most classes true label will be 0 except for the right one, the loss function basically can be visualized by this pseudo formula
$$
L = -\log(\text{P(right word)})
$$
We also know that $u_{j}$ is the scores calculated by dot product of 2 embedding vector, input  $v_{I}$ and context $v_{C}$ (which we have mentioned earlier where these 2 come from). The loss function can be fully expanded as
$$
L(y_{j}, \hat{y}_{j}) = -\log\left( \frac{\exp(u_{j})}{\sum_{k=1}^{V} \exp(u_{k})} \right)
$$
where j is the index of the 1 in the one-hot encoded ground truth, meaning $y_{j} = 1$
This is the loss of 1 word/slots in the context window, we will sum the slot for each of these slots to get the total Loss (scalar)


The reason we use Categorical Cross-Entrophy Loss is because Softmax involves quotients and exponentials while CCE involves logarithms, when we chain them together for the backpropagation, the Logarithm from CCE cancels out the Exponential from Softmax, so we have this beautyful formula
$$
Gradient = Prediction - Truth
$$
You will get the details later in the next part but looking at this, there are several advantages:
- **Linear signal:** gradient are computed as simple subtraction, so it won't explode or vanish easily
- **Intuitive Updates:** If the model predicts $0.8$ for a word that should be $1.0$, the gradient is $0.8 - 1.0 = -0.2$. The model knows exactly how much to "nudge" the weights to close that $0.2$ gap.
- **Steepest when Wrong:** If the model is very confident and wrong (predicting $0.001$ for the truth), the gradient is massive ($\approx -1$), forcing the model to change its mind quickly.

#### 1.3.4. Negative Sampling (Preview)
We notice a problem in this normal [[Categorical Cross-Entrophy (CCE)]] approach as loss function. It is that the computing is very heavy. Let's analyze the complexity of the loss function, to do that we will write down the fully expanded version of the loss function to the deepest parameter
$$
L(y, \hat{y})_{C} = -  y_{j}\log(\frac{\exp(v_{I}\cdot v_{C})}{\sum_{k=1}^{V} \exp(v_{I}\cdot v_{k})}) 
$$
1. Numerator (1 dot product): $v_{I}$ and $v_{c}$ are both size $N$ $\to$ this takes $N$ operations
2. Denominator (V dot products): to get the sum we must calculate every possible score, meaning V possible score and each score takes N operations $\to$ this takes $V\times N$ operations
3. The summation in CCE: takes $C$ operation
So technically it will be $O(C \times (N+1) \times V)$ but since C do not grow and +1 seems trivial, we can generalize the complexity of this Loss function as:
$$
O(V \times N)
$$
This is a very big complexity. Imagine you your hidden state has the size of 300 and your vocabulary contains 10,000 words. Then to calculate loss for each word, you would need approximately 3,000,000 operations

So to fight this complexity, researchers invent a new way to compute the loss function while still having great results and other minor beneifts, researchers invent [[Negative Sampling - Word2Vec Skip Gram]] to achieve a massive boost in computational efficiency

Brief:
1. **Intuition:** Instead of a massive Multi-Class Classification problem (is it word 1, 2, ... or 100,000?), we turn it into a tiny Binary Classification problem.
2. **The New Objective:** "Maximize the probability that the real pair is Real ($1$), and minimize the probability that the fake pairs are Real ($0$)."
Read the details in [[Negative Sampling - Word2Vec Skip Gram]]


### 1.3. Backpropagation
We know that our goal is to adjust the 2 weights $W_{V\times N}$ and $W_{N\times V}$
To do that we will visualize the error pipeline to see how should we set the chain rule for the gradients
$$
\begin{align}
Loss &\to \hat{y} \to u \to W' \\
Loss &\to \hat{y} \to u \to h \to W
\end{align}
$$

#### 1.3.1.  Finding Gradients of $W_{N\times V}'\left( \frac{ \partial L }{ \partial W' } \right)$
This gradient tell us how to update the context vectors
Based on the error pipeline above, we can expand this gradient into this chain:
$$
\frac{ \partial L }{ \partial W' }  =\frac{ \partial L }{ \partial \hat{y} } \cdot \frac{ \partial \hat{y} }{ \partial u } \cdot \frac{ \partial u }{ \partial W' } 
$$
But since we know that the loss function we use is the combination of Softmax and Cross-entrophy, or namely [[Categorical Cross-Entrophy (CCE)]], we know that $\frac{ \partial L }{ \partial u_{j} } = \hat{y}_{j} - y_{j}$ for each score $u_{j}$
So the gradient formula is
$$
\frac{ \partial L }{ \partial v_{j}' } = (\hat{y}_{j} - y_{j}) \cdot h
$$
The Matrix form as we combine all this vector gradient will be:
$$
\frac{ \partial L }{ \partial W' } = h \cdot (\hat{y} - y)^T
$$


#### 1.3.2. Finding Gradients of $W_{V\times N}\left( \frac{ \partial L }{ \partial W } \right)$
Looking at the rerror pipeline, this is similar to gradients of output weights, with extra step with the appearance of $h$
$$
\frac{ \partial L }{ \partial W } = \frac{ \partial L }{ \partial \hat{y} } \cdot \frac{ \partial \hat{y} }{ \partial u } \cdot \frac{ \partial u }{ \partial h } \cdot \frac{ \partial h }{ \partial W } 
$$
There is an important detail, the hidden state $h$ actually contributes to the calculation of score $u_{j}$ (every word in the vocabulary) via matrix multiplication. Therefore, the error at $h$ is the sum of the errors from all output nodes (every element in $u$)
$$
\frac{ \partial L }{ \partial h } = \sum_{j=1}^{V}\frac{ \partial L }{ \partial u_{j} }  \cdot \frac{ \partial u_{j} }{ \partial h }  
$$
Since $u_{j}=v_{j}'\cdot h$, the derivative w.r.t $h$ is $v_{j}'$
So we have this
$$
\frac{ \partial L }{ \partial h }  = \sum_{j=1}^{V} (\hat{y}_{j}-y_{j})\cdot v_{j}'
$$
Let's call this vector $\mathbf{EH}$ (Error at Hidden). This is an $N$-dimensional vector

Since $h = W \cdot x_{k}$ so $\frac{ \partial h }{ \partial W } = x_{k}$. And $x_{k}$ is  one-hot encoded input (a bunch of 0 and one value 1). So if we multiply with EH, we will have a gradient matrix $(V\times N)$ with almost every row filled with 0 except 1 with EH. The notation will be
$$
\frac{ \partial E }{ \partial w_{k} } = \begin{cases}
\mathbf{EH} &  \text{if k = I (input word)} \\
0  & \text{otherwise}
\end{cases}
$$
This is intuitive, since we input only 1 word so we will only want the gradient of 1 word embedding to adjust.

$$
EH \cdot x_{k} = \frac{ \partial L }{ \partial W' } 
$$
$$
\begin{bmatrix}
0 \\
0 \\
1 \\
0 \\
\end{bmatrix}
\cdot 
\begin{bmatrix}
1 & 2 & 3 & 4
\end{bmatrix}
=
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
1 & 2 & 3 & 4 \\
0 & 0 & 0 & 0
\end{bmatrix}
= \frac{ \partial L }{ \partial W } 
$$

#### 1.3.3. Update (SGD - Stochiastic Gradient Descent)
**Update Output Matrix**
$$W'_{\text{new}} = W'_{\text{old}} - \eta \cdot [h \cdot (\hat{y} - y)^T]$$
*(Every column in $W'$ is updated slightly*

**Update Input Matrix**
$$w_{I, \text{new}} = w_{I, \text{old}} - \eta \cdot \mathbf{EH}$$
Actually the raw form including the entire matrix is still right, except it won't update anything since the gradient of other rows are all 0
$$
W_{new} = W_{old} - \eta \cdot \mathbf{EH}
$$



## 2. Code




## 3. Assignments/Applications


# References
[[Word2Vec]]
[[Word2Vec - Continuous bag of words (CBOW)]]
[[Transformers]]
[[Sequence to Sequence Learning with Neural Networks - Original Paper]]
[[Negative Sampling - Word2Vec Skip Gram]]