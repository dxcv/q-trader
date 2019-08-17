#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:07:39 2019

@author: igor
"""

global queue

# Find minimum steps to reach specific cell by Knight
class Cell: 
    def __init__(self, x = 0, y = 0, dist = 0, path = []): 
        self.x = x 
        self.y = y 
        self.dist = dist
        self.path = path.copy()
        self.path.append((x,y))
          
    # checks whether cell is inside the board 
    def is_inside(self, N): 
        return (self.x >= 1 and self.x <= N and self.y >= 1 and self.y <= N)
      
# Method returns minimum steps to reach target position  
def get_best_path(start, finish, N):
    global queue
      
    #all possible movments for the knight 
    dx = [2, 2, -2, -2, 1, 1, -1, -1] 
    dy = [1, -1, 1, -1, 2, -2, 2, -2] 
      
    queue = []
    # make all cells unvisited  
    visited = [[False for i in range(N + 1)] for j in range(N + 1)] 
            
    # init starting position of Knight with 0 distance 
    c = Cell(start[0], start[1], 0)
    if start == finish: return c
    queue.append(c)
      
    # visit starting state 
    visited[c.x][c.y] = True
      
    # loop until queue is not empty  
    while(len(queue) > 0):           
        t = queue.pop(0) 
          
        # iterate for all reachable cells  
        for i in range(8): 
            c = Cell(t.x + dx[i], t.y + dy[i], t.dist + 1, t.path)

            # if current cell is equal to target cell - we have found the path!  
            if([c.x, c.y] == finish): 
                return c
              
            if(c.is_inside(N) and not visited[c.x][c.y]): 
                visited[c.x][c.y] = True
                queue.append(c) 
  
if __name__=='__main__':  
    N = 8
    start = [4, 4] 
    finish = [5, 4]
    t = get_best_path(start, finish, N)
    print(t.path, t.dist) 