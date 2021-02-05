## Alpharank 

The algorithmic implementation used from openspiel and integration code for algorithms like alphazero, MPV-MCTS, etc. is added in the alpharank_test.py file 

In the initial test, we compare the MPV_MCTS with Meta MPV-MCTS over the training for the Nim (10) game.

As an input to the alpharank algorithm, we give the snapshots to the network at certain instances from the beginning to the end of the training cycle. We have divided the training into 5 steps and we take the snapshot of the network at the end of each step.

The following graph shows the alpha-rank of each of the algorithm at the end of each step of training. Here we see that for the first step, the MPV-MCTS algorithm performs better that the Meta MPV-MCTS but from the second step onwards, the Meta MPV-MCTS algorithm constantly shows a better score than MPV-MCTS. Thus we can see that the rate of training of Meta MPV-MCTS is higher than MPV-MCTS. 

![mpv_vs_meta](https://user-images.githubusercontent.com/17771219/91606273-c7a65680-e93f-11ea-8d79-279dab931d6a.png)

The following is the network plot of the MPV-MCTS algorithm at the end of the fifth step of training:

![output_3](https://user-images.githubusercontent.com/17771219/91606360-ec9ac980-e93f-11ea-9cd7-489bfe7fa52f.png)

The following is the network plot of the Meta MPV-MCTS algorithm at the end of the fifth step of training:

![output_3](https://user-images.githubusercontent.com/17771219/91606280-ca08b080-e93f-11ea-90c2-ba966988f515.png)
