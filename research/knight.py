#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:07:39 2019

@author: igor
"""

# ASSUMPTIONS
# Python 3 or higher is used
# Board size is 8 x 8

board_map = 'ABCDEFGH'
board_size = len(board_map)

from functools import reduce

# Find minimum steps to reach specific cell by Knight
class Cell: 
    def __init__(self, x = 0, y = 0, path = None): 
        self.x = x 
        self.y = y
        if path is None: path = []
        self.path = path.copy()
        self.path.append((x,y))

    # Checks whether cell is inside the board 
    def is_inside(self): 
        return (self.x >= 1 and self.x <= board_size and self.y >= 1 and self.y <= board_size)
    
    def __pos(self, x, y):
        return board_map[x-1]+str(y)
    
    def get_pos(self):
        if self.is_inside(): return self.__pos(self.x, self.y)
        return str(self.x)+','+str(self.y)
    
    def get_path_str(self):
        return reduce(lambda a, b: a+' '+b, map(lambda x: self.__pos(x[0], x[1]), self.path))

# Method returns minimum steps to reach target position
# TODO: Handle invalid parameters
def get_best_path(start, finish):

    # All possible movments for the knight 
    dx = [2, 2, -2, -2, 1, 1, -1, -1] 
    dy = [1, -1, 1, -1, 2, -2, 2, -2]
      
    queue = []
    # Make all cells unvisited  
    visited = [[False for i in range(board_size + 1)] for j in range(board_size + 1)] 
            
    # Init starting position of Knight with 0 distance 
    x = int(board_map.index(start[0])) + 1
    y = int(start[1])
    c = Cell(x, y)
    
    # We do not need to move Knight
    if start == finish: return c
    queue.append(c)
      
    # Visit starting state 
    visited[c.x][c.y] = True
      
    # TODO: Handle no path found case         
    # Loop until queue is not empty  
    while(len(queue) > 0):           
        t = queue.pop(0) 
          
        # Iterate for all reachable cells  
        for i in range(8): 
            c = Cell(t.x + dx[i], t.y + dy[i], t.path)

            # if current cell is equal to target cell - we have found the path!  
            if(c.get_pos() == finish): 
                return c
              
            if(c.is_inside() and not visited[c.x][c.y]): 
                visited[c.x][c.y] = True
                queue.append(c) 
  
if __name__=='__main__':
    instr = input('Please enter start and target positions:')
    params = instr.split()
        
    t = get_best_path(params[0], params[1])
    print(t.get_path_str())