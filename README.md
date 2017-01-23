## User operation for Tensorflow: Log2round.

# Discription
This is code for a custom activation function Log2round for Tensorflow. It takes tensor as input and outputs a tensor of the same size. Log2round doesn't change gradient during the backpropagation step. Log2round makes it easy to implement an approach similar to the one discussed in [https://arxiv.org/pdf/1602.02830v3.pdf](https://arxiv.org/pdf/1602.02830v3.pdf).
```
log2round(x) = pow(2, n) * sign(x),
where integer n minimizes | |x| - pow(2,n) |; n can be of any sign.

d(log2round(x))/dx = dx/dx = 1
```

# Usage

To use Log2round one needs Tensorflow installed and import these modules:
```
import log2round_grad
from log2round_module import log2round
```
See log2round_test.py for an example.
