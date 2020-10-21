from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

class Board:
    def __init__(self,width,height,cluesV,cluesH):
        self.width = width
        self.heigth = height
        self.cluesV = cluesV
        self.cluesH = cluesH
        self.board = [[-1 for _ in range(width)] for _ in range(height)]

    def col_to_row(self, n):
        return [self.board[x][n] for x in range(self.heigth)]

    @staticmethod
    def check_set(group:list,clues):
        if len(clues) == 0:
            return not 1 in group
        clueInd = 0
        groupInd = 1
        inBlock = group[0] == 1
        blockSize = int(inBlock)
        while 1:
            if groupInd == len(group):
                code = 1
                break
            if group[groupInd] == -1:
                code = 2
                break

            if group[groupInd] == 1:
                if inBlock:
                    blockSize+=1
                else:
                    blockSize += 1
                    inBlock = True

            else:
                if inBlock:
                    if clueInd == len(clues):
                        return False
                    if blockSize != clues[clueInd]:
                        return False
                    inBlock = False
                    blockSize = 0
                    clueInd+=1

            groupInd+=1


        if code == 1:
            if inBlock:
                if clueInd == len(clues):
                    return False
                if blockSize != clues[clueInd]:
                    return False
                inBlock = False
                blockSize = 0
                clueInd += 1
            return clueInd == len(clues)

        else:
            pos = groupInd-1
            if inBlock:
                if clueInd == len(clues):
                    return False
                if blockSize>clues[clueInd]:
                    return False
                pos += clues[clueInd]-blockSize
                clueInd+=1
            else:
                pos -= 1
            while clueInd != len(clues):

                pos+=1
                pos += clues[clueInd]
                clueInd+=1
            return pos<len(group)

    def pos_to_row(self,n):
        return n//self.width

    def pos_to_col(self,n):
        return n%self.width

    def pos_to_coord(self,n):
        return n%self.width,n//self.width

    def solve(self):
        pos = 0
        cnt = 0
        while True:
            cnt+=1
            if cnt % 1000000 == 0:
                for l in self.board:
                    print(l)
                print()
            c,r = self.pos_to_coord(pos)
            if self.board[r][c] == 1:
                self.board[r][c] = -1
                pos -= 1
            else:
                self.board[r][c] += 1
                if self.check_set(self.board[r],self.cluesH[r]) and self.check_set(self.col_to_row(c),self.cluesV[c]):
                    if pos == self.width * self.heigth - 1:
                        for l in self.board:
                            print(l)
                        print(cnt)
                        return
                    pos += 1

    def show(self):
        plt.imshow(self.board)
        plt.show()

# vc = [
#     [20],
#     [2,2,2],
#     [2,17,1],
#     [9,18,1],
#     [1,3,2,4,1],
#
#     [4,7,8,3,1],
#     [8,4,8,2,1],
#     [2,3,8,4],
#     [2,2,8,4],
#     [20],
#
#     [1,1],
#     [1,3,1,1],
#     [1,4,1,2],
#     [1,5,1,1,1],
#     [1,5,9,1],
#
#     [1,15,1],
#     [1,7,1,1],
#     [1,6,2],
#     [1,5,1],
#     [7]
# ]
#
# hc = [
#     [4],
#     [1,2],
#     [1,2],
#     [1,2],
#     [2,1],
#
#     [4],
#     [4],
#     [2,3],
#     [3,1,2],
#     [2,2,2],
#
#     [1,5,11],
#     [1,8,1],
#     [1,2,4,1],
#     [1,2,1,9],
#     [1,2,1,9],
#
#     [1,2,1,1,9],
#     [1,2,6,8],
#     [1,2,5,1,6],
#     [1,2,5,1,3],
#     [1,2,5,4],
#
#     [1,2,5,2],
#     [1,2,5,2],
#     [1,2,5,2],
#     [1,2,4,2],
#     [1,3,1,2],
#
#     [6,1,2],
#     [10,2],
#     [1,8,4],
#     [2,3,1,1],
#     [8,8]
# ]
#
# b = Board(20,30,vc,hc)
# b.solve()
# b.show()


# b = Board(5,5,[[2],[1],[4],[2],[4]],[[2],[1,1,1],[1,1],[3],[3]])
# b.solve()
# b.show()

# print(Board.check_set([0,0,1,1,0,1],[2,1]))     # True
# print(Board.check_set([1,0,1,1,0,1],[1,2,1]))   # True
# print(Board.check_set([0,0,0,0,0,0],[]))        # True
# print(Board.check_set([1,0,1,1,0,0],[1,2]))     # True
#
# print()
#
# print(Board.check_set([0,1,1,1,0,1],[2,1]))     # False
# print(Board.check_set([1,0,1,1,0,0],[1,2,1]))   # False
# print(Board.check_set([0,1,0,0,0,0],[]))        # False
# print(Board.check_set([0,0,1,1,0,0],[1,2]))     # False
#
# print()
#
# print(Board.check_set([0,0,1,1,-1,-1],[2,1]))   # True
# print(Board.check_set([0,0,1,1,-1],[2,1]))      # False
# print(Board.check_set([0,0,1,1,-1,-1],[3]))     # True
# print(Board.check_set([0,0,1,1,-1,-1],[4]))     # True
# print(Board.check_set([0,0,1,1,-1,-1],[5]))     # False
#
# print()
#
# print(Board.check_set([0,0,1,1,-1,-1,-1,-1],[2,1,1]))     # True
# print(Board.check_set([0,0,1,1,-1,-1,-1,-1],[2,1,2]))     # False





