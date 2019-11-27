# class T():
#     def __init__(self, no_list=list()):
#         self.no = 0
#         self.no_list = no_list
#     '''
#     output
#     the 1 class is 1 [1, 2]
#     the 2 class is 2 [1, 2]
#     '''

class T():
    def __init__(self):
        self.no = 0
        self.no_list = list()
    '''
    output
    the 1 class is 1 [1]
    the 2 class is 2 [2]
    '''

# class T():
#     no = 0
#     no_list = list()
#     '''
#     output
#     the 1 class is 1 [1, 2]
#     the 2 class is 2 [1, 2]
#     '''

t_list = list()

t = T()
t.no = 1
t.no_list.append(1)
t_list.append(t)

t2 = T()
t2.no = 2
t2.no_list.append(2)
t_list.append(t2)

for index, lis in enumerate(t_list):
    print('the', index+1, 'class is', lis.no, lis.no_list)

