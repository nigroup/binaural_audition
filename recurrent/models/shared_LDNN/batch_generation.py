import numpy as np
import math
import time
import random
import heapq
'''
Adapt minimal heap to lower the time complexity to bath*log(batch)*files
'''
def construct_rectangle(pathset,Total_epochs):
    # current_length = [0]*Total_epochs
    index_rectangle = [[]]*Total_epochs
    data = [(0,x) for x in range(Total_epochs)]
    for e in range(Total_epochs+1):
        random.shuffle(pathset)

        for j in pathset:
            heapq.heapify(data)
            minimal = heapq.heappop(data)
            k = minimal[1]
            val = minimal[0] + int(j[1])
            heapq.heappush(data,(val,k))
            # k = current_length.index(min(current_length))
            # current_length[k] += int(j[1])
            index_rectangle[k] = index_rectangle[k] + [int(j[0])]
    return np.array(index_rectangle)
def get_filepaths(Total_epochs, Batch_timelength,paths):
    output = []
    total_length = paths[:,1].astype(int).sum()
    num_clips = math.ceil(total_length/Batch_timelength) -1
    rectangle = construct_rectangle(paths.tolist(),Total_epochs)
    row_flag = np.array([0]*2*Total_epochs).reshape([Total_epochs,2])
    for _ in range(num_clips):
        for i in range(Total_epochs):
            len = 0
            string = ''
            while 1:
                id = rectangle[i][row_flag[i,0]]
                id_len = int(paths[id,1])
                real_len = id_len - row_flag[i,1]
                if len + real_len < Batch_timelength:
                    string = string +  paths[id,2] \
                             + '&' + str(row_flag[i,1]) \
                             + '&' + paths[id,1] + '@'
                    row_flag[i] = [row_flag[i,0]+1,0]
                    len += real_len
                elif len + real_len == Batch_timelength:
                    string = string + paths[id, 2] \
                             + '&' + str(row_flag[i, 1]) \
                             + '&' + paths[id, 1]
                    row_flag[i] = [row_flag[i, 0] + 1, 0]
                    break
                else:
                    extra = len + real_len - Batch_timelength
                    id_indice = id_len - extra
                    string = string + paths[id, 2] \
                             + '&' + str(row_flag[i, 1]) \
                             + '&' + str(id_indice)
                    row_flag[i] = [row_flag[i, 0], id_indice]
                    break;
            output.append(string)

    return output
# dir_train = 'trainpaths.npy'
# paths = np.load(dir_train)
# t = time.time()
# a = get_filepaths(1,6000,paths)
# print(time.time()-t)


# fx, fy = np.array([]).reshape(0,160), np.array([]).reshape(0,13)
# for instance in a[0].split('@'):
#     p, start, end = instance.split('&')
#     data = np.load(p)
#     x = np.reshape(data['x'], [-1, 160])
#     y = np.transpose(data['y'])
#     y[y == 0] = -1
#     fx = np.concatenate((fx, x[int(start):int(end)]), axis=0)
#     fy = np.concatenate((fy, y[int(start):int(end)]), axis=0)
# print(time.time()-t)