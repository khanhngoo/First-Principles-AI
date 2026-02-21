2026-02-05 08:47
Status: #baby 
Tags: [[Deep Learning]], [[Neural Networks Collection]]

# Word Embedding and Word2Vec
## 0. Intuition
- Neural Network does not work well with words, so we need a way to turn words into number
- Continuous Bag of Words + Skip Gram
- Negative Sampling
Ý tưởng cơ bản của word2vec có thể được gói gọn trong các ý sau:
- Hai từ xuất hiện trong những văn cảnh giống nhau thường có ý nghĩa gần với nhau.
- Ta có thể đoán được một từ nếu biết các từ xung quanh nó trong câu. Ví dụ, với câu “Hà Nội là … của Việt Nam” thì từ trong dấu ba chấm khả năng cao là “thủ đô”. Với câu hoàn chỉnh “Hà Nội là thủ đô của Việt Nam”, mô hình word2vec sẽ xây dựng ra embeding của các từ sao cho xác suất để từ trong dấu ba chấm là “thủ đô” là cao nhất.



## 1. Architecture
Word2Vec utlizes 2 architectures **CBOW and Skip-gram**

### 1.1. Terminology
![[Pasted image 20260207162342.png]]
The blue colored one í called the target words, and the context words are defined as the words that are from not over C/2 if C is called the context window number
In the image C = 4

Word2vec ddefines embedding vector for each word in a dictionary $w$
When it is a target word, its embedding vector will be $u$ and $v$ if it is a context word
Similarly we have matrix embedding $U, V$


### 1.2. CBOW (Continuous Bag of Words)
Predict current word given context words within a specific window.t
Similar to a feed-forward neural network
![[Pasted image 20260207155833.png]]
READ MORE ABOUT CBOW ARCHITECTURE: [[Word2Vec - Continuous bag of words (CBOW)]]


### 1.3. Skip Gram
Skip Gram predicts surrounding context words given current word
![[Pasted image 20260207155904.png]]

![[Pasted image 20260207163604.png]]

READ MORE ABOUT SKIP GRAM ARCHITECTURE HERE: [[Word2Vec - Skip Gram]]


## 2. Comparison
| **Feature**        | **CBOW**                                                            | **Skip-gram**                                                      |
| ------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Speed**          | **Faster** ⚡                                                        | Slower 🐢                                                          |
| **Why?**           | 1 calculation per context window (Average $\rightarrow$ Predict 1). | $C$ calculations per context window (1 $\rightarrow$ Predict $C$). |
| **Accuracy**       | Better for **Frequent Words**.                                      | Better for **Rare Words**.                                         |
| **Representation** | Smooths over nuances (averaging blurs details).                     | Captures sharp, distinct meanings for every word.                  |
| **Data Needs**     | Needs less data to converge.                                        | Needs more data, but learns deeper relationships.                  |



# References
[[Sequence to Sequence Learning with Neural Networks - Original Paper]]
[[Transformers]]
[[Neural Network (NN)]]
