# Highest Safe Rung Problem

### One of the game runs -> MCTS vs MCTS

```
Initial state:
........
Player 0 sampled action: x(0,1)
Next state:
.x......
Player 1 sampled action: .(0,1)
Next state:
xx......
Player 0 sampled action: x(0,2)
Next state:
xxx.....
Player 1 sampled action: .(0,1)
Next state:
xxx.....
Player 0 sampled action: x(0,5)
Next state:
xxx..x..
Returns: -1.0 1.0 , Game actions: x(0,1) .(0,1) x(0,2) .(0,1) x(0,5)
Number of games played: 1
Number of distinct games played: 1
Players: mcts mcts
Overall wins [0, 1]
Overall returns [-1.0, 1.0]
```

### Network configuration is low to stop faster convergence:
HSR config: (k,q,n) = (2,3,7) \
Running HSR with Alpha-Zero:
![hsr_az_simplified(2,3,7)](https://user-images.githubusercontent.com/17771219/85291967-3b704280-b469-11ea-844d-fba6a7677a00.png)

Running HSR with Meta MPV-MCTS:
![hsr_mpv_simplified(2,3,7)](https://user-images.githubusercontent.com/17771219/85291982-3f03c980-b469-11ea-910b-25ed97710507.png)

### Network configuration is high so convergence if much faster for player 2:
HSR config: (k,q,n) = (2,3,7) \
Running HSR with Alpha-Zero:
![hsr(2,3,7)](https://user-images.githubusercontent.com/17771219/85293180-fcdb8780-b46a-11ea-9926-d5ee2b1d8bc3.png)

Running HSR with Meta MPV-MCTS:
![hsr_mpv(2,3,7)](https://user-images.githubusercontent.com/17771219/85293188-ffd67800-b46a-11ea-8fba-f9784a4f50a0.png)


### HSR game convergence evaluation 
HSR config: (k,q,n) = (2,3,7) \
Running HSR with Alpha-Zero:

![myplot_3](https://user-images.githubusercontent.com/17771219/87186453-87b8e080-c2b9-11ea-8b11-ea979ade36fd.png)

Blue - Proponent, Orange - Opponent

### HSR (3,3,8):
![alpharank](https://user-images.githubusercontent.com/17771219/92991928-eadf1300-f4b4-11ea-9f4e-9f5a073b6323.png)
![elorating](https://user-images.githubusercontent.com/17771219/92991929-ee729a00-f4b4-11ea-9892-e2dcdeaed1ae.png)

### HSR (4,4,16):
![alpharank](https://user-images.githubusercontent.com/17771219/92991941-f8949880-f4b4-11ea-9f7e-dcaf77ef8978.png)
![elorating](https://user-images.githubusercontent.com/17771219/92991949-04805a80-f4b5-11ea-899e-cf96b4f8437c.png)
