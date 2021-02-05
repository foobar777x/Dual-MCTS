# Multiple Policy Value MCTS

Here we have implemented a multiple policy value network which consists of a smaller network that explores various levels of the Monte Carlo tree and provides a lookahead for the larger network to converge faster to the optimum policy.

1. The following is how a vanilla mpv-mcts would perform to a (20,3) Nim game:

![alpha zero](https://user-images.githubusercontent.com/17771219/83661980-ec836b80-a594-11ea-8da5-f8a39820baea.png)

2. Next we run the same (20,3) Nim game on our implementation of mpv-mcts with metacontroller keeping the configuration of the main network same as the vanilla alpha-zero.

![mpv mcts](https://user-images.githubusercontent.com/17771219/83662011-f86f2d80-a594-11ea-8a9e-5233b70821a0.png)

We see that the mpv-mcts performs better than the alpha-zero algorithm in just 20 training steps
