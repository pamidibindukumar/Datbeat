
#program to count no of components for given adjacency type
class Graph:
    def __init__(self,m,n,g,adjacencytype):
        self.row=m
        self.col=n
        self.graph=g
        self.adjacencytype=adjacencytype
        
        
    def isSafe(self,i,j,visited):
        '''
        a method to check if given cell (i,j) can be  included in search
        '''
        # row number is in range, column number
        # is in range and value is 1
        # and not yet visited
        temp=(i>=0 and i<self.row and j>=0 and j<self.col and
             not visited[i][j] and self.graph[i][j])
        return temp
    
    def DepthFirstSearch(self,i,j,visited):
        '''method which considers all 8 neighbors of a given cell
        '''
        # These arrays are used to get row and
        # column numbers of 8 neighbours
        # of a given cell
        row_nbr = [-1, -1, -1,  0, 0,  1, 1, 1];
        col_nbr = [-1,  0,  1, -1, 1, -1, 0, 1];
        #mark the cell has visited
        visited[i][j]=True
        #here iterating over because we have 8 neighbors
        for k in range(8):
            if self.isSafe(i + row_nbr[k], j + col_nbr[k], visited):
                self.DepthFirstSearch(i + row_nbr[k], j + col_nbr[k], visited)
                
    def DiagonalSearch(self,i,j,visited):
        '''method which considers all 4 neighbors of a given cell diagonally
        '''
        # These arrays are used to get row and
        # column numbers of 4 neighbours
        # of a given cell
        row_nbr = [-1, -1,  1, 1];
        col_nbr = [-1,  1, -1, 1];
        visited[i][j]=True
        
        for k in range(4):
            if self.isSafe(i + row_nbr[k], j + col_nbr[k], visited):
                self.DiagonalSearch(i + row_nbr[k], j + col_nbr[k], visited)
                
    def HorizontalSearch(self,i,j,visited):
        '''method which considers all 2 neighbors of a given cell horizontally
        '''
        # These arrays are used to get row and
        # column numbers of 2 neighbours
        # of a given cell
        row_nbr = [0,0];
        col_nbr = [-1,1];
        visited[i][j]=True
        
        for k in range(2):
            if self.isSafe(i + row_nbr[k], j + col_nbr[k], visited):
                self.HorizontalSearch(i + row_nbr[k], j + col_nbr[k], visited)
                
    def VerticalSearch(self,i,j,visited):
        '''first which considers all 2 neighbors of a given cell vertically
        '''
        # These arrays are used to get row and
        # column numbers of 2 neighbours
        # of a given cell
        row_nbr = [-1,1];
        col_nbr = [0,0];
        visited[i][j]=True
        
        for k in range(2):
            if self.isSafe(i + row_nbr[k], j + col_nbr[k], visited):
                self.VerticalSearch(i + row_nbr[k], j + col_nbr[k], visited)
                
    def HorizontalVerticalSearch(self,i,j,visited):
        '''first which considers all 4 neighbors of a given cell horizontally and vertically
        '''
        # These arrays are used to get row and
        # column numbers of 4 neighbours
        # of a given cell
        row_nbr = [-1,1,0,0];
        col_nbr = [0,0,-1,1];
        visited[i][j]=True
        
        for k in range(4):
            if self.isSafe(i + row_nbr[k], j + col_nbr[k], visited):
                self.HorizontalVerticalSearch(i + row_nbr[k], j + col_nbr[k], visited)
                
    def countcomponents(self):
        # The main function that returns
        # count of components in a given boolean
        # 2D matrix

        # Make a bool array to mark visited cells.
        # Initially all cells are unvisited
        visited = [[False for j in range(self.col)]for i in range(self.row)]

        # Initialize count as 0 and traverse
        # through the all cells of
        # given matrix
        count=0
        
        for i in range(self.row):
            for j in range(self.col):
                # If a cell with value 1 is not visited yet,
                # then new component found
                if visited[i][j]==False and self.graph[i][j]==1:
                    # Visit all cells in this component
                    # and increment component count
                    self.DepthFirstSearch(i,j,visited)
                    count+=1
        return count
    
    def countcomponentshorizontal(self):
        visited = [[False for j in range(self.col)]for i in range(self.row)]
        count=0
        
        for i in range(self.row):
            
            for j in range(self.col):
                if visited[i][j]==False and self.graph[i][j]==1:
                    self.HorizontalSearch(i,j,visited)
                    count+=1
            
        return count
    
    def countcompoenetsvertical(self):
        visited = [[False for j in range(self.col)]for i in range(self.row)]
        count=0
        
        for i in range(self.row):
            
            for j in range(self.col):
                if visited[i][j]==False and self.graph[i][j]==1:
                    self.VerticalSearch(i,j,visited)
                    count+=1
            
        return count
    
    def countcompoenetshorizontalvertical(self):
        visited = [[False for j in range(self.col)]for i in range(self.row)]
        count=0
        
        for i in range(self.row):
            
            for j in range(self.col):
                if visited[i][j]==False and self.graph[i][j]==1:
                    self.HorizontalVerticalSearch(i,j,visited)
                    count+=1
            
        return count
    #DiagonalSearch
    def countcompoenetsdiagonal(self):
        visited = [[False for j in range(self.col)]for i in range(self.row)]
        count=0
        
        for i in range(self.row):
            
            for j in range(self.col):
                if visited[i][j]==False and self.graph[i][j]==1:
                    self.DiagonalSearch(i,j,visited)
                    count+=1
            
        return count
    
    def main(self):
        '''
        this method will compute number of components for a given adjacency type
        '''
        result=0
        if self.adjacencytype==1:
            result=self.countcomponentshorizontal()
        if self.adjacencytype==2:
            result=self.countcomponentsvertical()
        if self.adjacencytype==3:
            result=self.countcompoenetshorizontalvertical()
        if self.adjacencytype==4:
            result=self.countcompoenetsdiagonal()
        if self.adjacencytype==5:
            result=self.countcomponents()
        return result
    
if __name__=="__main__":

    adjacencytype=dict(
        'horizontal':1
        'vertical':2
        'horizontal_vertical':3
        'diagonal':4
        'horizontal_vertical_diagonal':5

    )
    arr = [[1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1]]


    row = len(arr)
    col = len(arr[0])

    g = Graph(row, col, arr,adjacencytype['horizontal'])
    
    print ("Number of islands is:")
    print (g.main())
    