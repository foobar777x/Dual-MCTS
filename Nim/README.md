## Monte Carlo algorithm on Nim:

1. Perfect play (MC): Using the agent as the first player and human as the second. The agent uses optimal strategy to win the game.

~~~
Initial state:
..........
Player 0 sampled action: x(0,1)
Next state:
xx........
Choose an action (empty to print legal actions): 
Legal actions(s):
  2: o(0,2)    3: o(0,3)    4: o(0,4)  
Choose an action (empty to print legal actions): 4
Player 1 sampled action: o(0,4)
Next state:
xxooo.....
Player 0 sampled action: x(0,5)
Next state:
xxooox....
Choose an action (empty to print legal actions): 
Legal actions(s):
  6: o(0,6)    7: o(0,7)    8: o(0,8)  
Choose an action (empty to print legal actions): 8
Player 1 sampled action: o(0,8)
Next state:
xxoooxooo.
Player 0 sampled action: x(0,9)
Next state:
xxoooxooox
Returns: 1.0 -1.0 , Game actions: x(0,1) o(0,4) x(0,5) o(0,8) x(0,9)
Number of games played: 1
Number of distinct games played: 1
Overall wins [1, 0]
Overall returns [1.0, -1.0]
~~~

2. Perfect play (Human): Here we let the human play the first game with perfect play. Human wins the game by following the optimal policy.

~~~
Initial state:
..........
Choose an action (empty to print legal actions): 1
Player 0 sampled action: x(0,1)
Next state:
xx........
Player 1 sampled action: o(0,3)
Next state:
xxoo......
Choose an action (empty to print legal actions): 
Legal actions(s):
  4: x(0,4)    5: x(0,5)    6: x(0,6)  
Choose an action (empty to print legal actions): 5
Player 0 sampled action: x(0,5)
Next state:
xxooxx....
Player 1 sampled action: o(0,6)
Next state:
xxooxxo...
Choose an action (empty to print legal actions): 
Legal actions(s):
  7: x(0,7)    8: x(0,8)    9: x(0,9)  
Choose an action (empty to print legal actions): 9
Player 0 sampled action: x(0,9)
Next state:
xxooxxoxxx
Returns: 1.0 -1.0 , Game actions: x(0,1) o(0,3) x(0,5) o(0,6) x(0,9)
Number of games played: 1
Number of distinct games played: 1
Overall wins [1, 0]
Overall returns [1.0, -1.0]
~~~

3. Imperfect play (Human): Here we let the human play first and deliberately start with an imperfect move. We observe that MC takes advantage of this and starts using the optimal strategy after humans mistake:

~~~
Initial state:
..........
Choose an action (empty to print legal actions): 0
Player 0 sampled action: x(0,0)
Next state:
x.........
Player 1 sampled action: o(0,1)
Next state:
xo........
Choose an action (empty to print legal actions): 
Legal actions(s):
  2: x(0,2)    3: x(0,3)    4: x(0,4)  
Choose an action (empty to print legal actions): 4
Player 0 sampled action: x(0,4)
Next state:
xoxxx.....
Player 1 sampled action: o(0,5)
Next state:
xoxxxo....
Choose an action (empty to print legal actions): 
Legal actions(s):
  6: x(0,6)    7: x(0,7)    8: x(0,8)  
Choose an action (empty to print legal actions): 8
Player 0 sampled action: x(0,8)
Next state:
xoxxxoxxx.
Player 1 sampled action: o(0,9)
Next state:
xoxxxoxxxo
Returns: -1.0 1.0 , Game actions: x(0,0) o(0,1) x(0,4) o(0,5) x(0,8) o(0,9)
Number of games played: 1
Number of distinct games played: 1
Overall wins [0, 1]
Overall returns [-1.0, 1.0]
~~~



