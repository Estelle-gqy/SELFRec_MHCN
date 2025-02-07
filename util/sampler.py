from random import shuffle, randint, choice, sample
import numpy as np

# by-user的推荐，根据每个user推荐item
# def next_batch_pairwise(data, batch_size, n_negs=1):
#     training_data = data.training_data
#     shuffle(training_data)
#     ptr = 0
#     data_size = len(training_data)
#     while ptr < data_size:
#         if ptr + batch_size < data_size:
#             batch_end = ptr + batch_size
#         else:
#             batch_end = data_size
#         users = [training_data[idx][0] for idx in range(ptr, batch_end)]  # 获取当前batch的用户
#         items = [training_data[idx][1] for idx in range(ptr, batch_end)]  # 获取当前batch的item
#         ptr = batch_end  # 当前batch的末尾
#         u_idx, i_idx, j_idx = [], [], []  # 分别表示user、item、负例样本集合
#         item_list = list(data.item.keys())
#         for i, user in enumerate(users):
#             i_idx.append(data.item[items[i]])  # data.item[items[i]] 找到user对应的item的index
#             u_idx.append(data.user[user])  # 找到user的index
#             for m in range(n_negs):  # ！从训练集所有的item中，随机抽取一个样本作为负例样本
#                 neg_item = choice(item_list)
#                 while neg_item in data.training_set_u[user]:  # 如果neg_item在当前user处理过的item集合中，重新抽一次，为什么错的原因！
#                     neg_item = choice(item_list)
#                 j_idx.append(data.item[neg_item])
#         yield u_idx, i_idx, j_idx


# by-item的推荐，根据每个item推荐user
def next_batch_pairwise(data, batch_size, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            # break
            batch_end = data_size
        examiners = [training_data[idx][0] for idx in range(ptr, batch_end)]
        users = [training_data[idx][3] for idx in range(ptr, batch_end)]  # 获取当前batch的用户
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]  # 获取当前batch的item
        labels = [training_data[idx][4] for idx in range(ptr, batch_end)]
        ptr = batch_end  # 当前batch的末尾
        u_idx, i_idx, e_idx = [], [], []
        # u_idx, i_idx, j_idx = [], [], []  # 分别表示user、item、负例user样本集合
        # user_list = list(data.user.keys())
        for i, examiner in enumerate(examiners):
            i_idx.append(data.item[items[i]])  # data.item[items[i]] 找到user对应的item的index
            u_idx.append(data.user[users[i]])  # 找到user的index
            e_idx.append(data.user[examiner])
        yield e_idx, u_idx, i_idx, labels

def next_batch_pointwise(data,batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    ptr = 0
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y

# def next_batch_sequence(data, batch_size,n_negs=1):
#     training_data = data.training_set
#     shuffle(training_data)
#     ptr = 0
#     data_size = len(training_data)
#     item_list = list(range(1,data.item_num+1))
#     while ptr < data_size:
#         if ptr+batch_size<data_size:
#             end = ptr+batch_size
#         else:
#             end = data_size
#         seq_len = []
#         batch_max_len = max([len(s[0]) for s in training_data[ptr: end]])
#         seq = np.zeros((end-ptr, batch_max_len),dtype=np.int)
#         pos = np.zeros((end-ptr, batch_max_len),dtype=np.int)
#         y = np.zeros((1, end-ptr),dtype=np.int)
#         neg = np.zeros((1,n_negs, end-ptr),dtype=np.int)
#         for n in range(0, end-ptr):
#             seq[n, :len(training_data[ptr + n][0])] = training_data[ptr + n][0]
#             pos[n, :len(training_data[ptr + n][0])] = list(reversed(range(1,len(training_data[ptr + n][0])+1)))
#             seq_len.append(len(training_data[ptr + n][0]) - 1)
#         y[0,:]=[s[1] for s in training_data[ptr:end]]
#         for k in range(n_negs):
#             neg[0,k,:]=sample(item_list,end-ptr)
#         ptr=end
#         yield seq, pos, seq_len, y, neg

def next_batch_sequence(data, batch_size,n_negs=1,max_len=50):
    training_data = list(data.original_seq.values())
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(range(1,data.item_num+1))
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        y =np.zeros((batch_end-ptr, max_len),dtype=np.int)
        neg = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end =  len(training_data[ptr + n]) > max_len and max_len-1 or len(training_data[ptr + n])-1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
            y[n, :end]=training_data[ptr + n][start+1:]
            negatives=sample(item_list,end)
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) >0:
                negatives = sample(item_list, end)
            neg[n,:end]=negatives
        ptr=batch_end
        yield seq, pos, y, neg, np.array(seq_len,np.int)
