# 数据集构建：针对每一条记录，找到当前审批人的

import pandas as pd
import os, math, tqdm, time
import re, jieba
from datetime import datetime
from collections import defaultdict

class Dataset(object):
    def __init__(self, data, staff_df):
        self.data = self.sort_by_time(data)
        self.staff_df = staff_df  # 作为构建社交关系的补充数据源
        self.user = {}
        self.examiner = defaultdict(dict)
        self.title = {}
        self.id2user = dict()
        self.id2title = dict()
        self.interaction_set = set()
        self.initialize(self.data)


    def initialize(self, data):
        print('Initializing data......')
        # 获取用户id
        self.user['null_token'] = 0
        self.id2user[0] = 'null_token'

        for idx in tqdm.tqdm(range(self.data.shape[0])):
            next_step_worker, one_title, examiner = self.data.loc[idx,][0], self.data.loc[idx,][1], \
                                                    self.data.loc[idx,][3]

            # 标题去标点去停用词
            stopwords = [' ', '', '\n', '《', '》', '[', ']', ',', '\'', '“', '”', '0', '1', '2', '3', '4', '5', '6',
                         '7', '8', '9']
            one_title = self.remove_punctuation(one_title)
            one_title = "".join([w for w in list(jieba.cut(one_title)) if w not in stopwords]).strip()
            self.data.loc[idx, '公文标题'] = one_title
            if one_title not in self.title:
                self.title[one_title] = len(self.title)
                self.id2title[self.title[one_title]] = one_title

            # 审批人id、名称提取
            if examiner not in self.user:
                self.user[examiner] = len(self.user)
                self.id2user[self.user[examiner]] = examiner

            # 下一环节处理人id、名称提取
            if isinstance(next_step_worker, str):
                temp = next_step_worker.split(',')
                self.data.at[idx, '下一环节处理人'] = temp
                worker_list = self.data.loc[idx, '下一环节处理人']
            elif math.isnan(next_step_worker) or next_step_worker == '':
                self.data.at[idx, '下一环节处理人'] = ["null_token"]
                worker_list = self.data.loc[idx, '下一环节处理人']
            for man in worker_list:
                if man.strip() not in self.user:
                    self.user[man] = len(self.user)
                    self.id2user[self.user[man]] = man

        for idx in tqdm.tqdm(range(self.data.shape[0])):
            examiner = self.data.loc[idx, ][3]
            # 提取审批者，曾经派发过的员工{examiner1:{worker1 : 1, worker2 : 1}, examiner2 : {worker3:1}}
            if examiner not in self.examiner:
                related_workers = self.data[self.data['审批人UID'] == examiner]['下一环节处理人'].to_list()
                worker_list = set()
                for rw in related_workers:
                    worker_list = worker_list.union(set(rw))

                for man in worker_list:
                    self.examiner[self.user[examiner]][self.user[man]] = 1

    def sort_by_time(self, records_data_df):
        # ！！按照审批时间排序，之前的数据做训练集、验证集，近的数据做测试集
        time_col = records_data_df["审批时间"].to_list()
        time_col = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in time_col]
        records_data_df["审批时间"] = pd.Series(time_col)
        records_data_df.sort_values(by="审批时间", inplace=True, ascending=True)
        records_data_df.reset_index(inplace=True)
        records_data_df = records_data_df.drop('index', axis=1)  # axis=1表示列
        return records_data_df

    def flatten_record(self, records_save_path):
        '''
        将处理记录转化成一个审批人对应一个下以环节处理人的形式，1 to N ---> 1 to 1
        生成retweet_records.txt文档,每一行包括（审批人id、公文id、公文标题、处理人id、rating）
        :param records_save_path
        :return 处理记录txt文档
        '''
        print(self.data.info())
        print('Preprocessing records.....')
        self.records_save_path = records_save_path

        with open(self.records_save_path, 'w', encoding='utf-8') as fp:
            for idx in tqdm.tqdm(range(self.data.shape[0])):
                next_step_worker, one_title, examiner = self.data.loc[idx, ][0], self.data.loc[idx, ][1], \
                                                        self.data.loc[idx, ][3]

                # 找出审批者曾经派发过文件的所有员工，不在worker list的标签为0，在的标签为1
                for worker in self.examiner[self.user[examiner]].keys():
                    if self.id2user[worker] in next_step_worker:  # 此时worker list已经是
                        fp.write(str(self.user[examiner]) + ' ' + str(self.title[one_title]) + ' ' + one_title + ' '+ str(worker) + ' 1\n')  # （审批人id、公文id、公文标题、处理人id、rating）
                    else:
                        fp.write(str(self.user[examiner]) + ' ' + str(self.title[one_title]) + ' ' + one_title + ' '+ str(worker) + ' 0\n')  # （审批人id、公文id、公文标题、处理人id、rating）


    def get_interaction_dataset(self, interaction_save_path, records_save_path, train_frac = 0.8):
        # return: 每一行包括(user1, user2) 代表两人是好友
        print('Processing interation dataset......')
        self.records_save_path = records_save_path
        self.interaction_save_path = interaction_save_path

        records = open(self.records_save_path, encoding='utf-8', errors='ignore').readlines()
        train_idx = int(train_frac * len(records))
        train_count = 0

        with open(self.interaction_save_path, 'w', encoding='utf-8') as fp:
            for idx, entry in self.data.iterrows():
                next_step_worker, examiner = entry[0], entry[3]
                if isinstance(next_step_worker, str):
                    worker_list = next_step_worker.split(',')
                elif math.isnan(next_step_worker) or next_step_worker == '':
                    worker_list = ['null_token']  # 如果下以环节处理人为空，设置为一个特殊的符号
                for man in worker_list:
                    examiner_index = self.user[examiner]
                    man_index = self.user[man]
                    if (examiner_index, man_index) not in self.interaction_set:
                        self.interaction_set.add((examiner_index, man_index))
                        if train_count <= train_idx:  # 只构建训练集的关系网
                            fp.write(str(examiner_index) + ' ' + str(man_index) + '\n')
                            train_count += 1

        # staff dataframe补充好友关系
        extra_relation_cnt = 0
        self.staff_df.sort_values(by="用户部门", ascending=True, inplace=True)
        self.staff_df.reset_index(inplace=True)
        self.staff_df = self.staff_df.drop('index', axis=1)
        # 删除用户部门为空的
        self.staff_df = self.staff_df.dropna(subset=['用户部门'], axis=0)

        self.staff_df['用户部门'] = [x.split(r'/') for _, x in self.staff_df['用户部门'].iteritems()]
        self.staff_df['depart_num'] = [len(x) for _, x in self.staff_df['用户部门'].iteritems()]
        depart_level = 4
        for i in range(depart_level):
            self.staff_df[f'depart{i + 1}'] = [x[i] if len(x) > i else '' for _, x in self.staff_df['用户部门'].iteritems()]

        print('Complementing interactions ......')
        with open(self.interaction_save_path, 'a', encoding='utf-8') as fp:  # 'a'表示append
            for e_idx in tqdm.tqdm(range(self.staff_df.shape[0])):
                entry = self.staff_df.loc[e_idx,]
                depart_len = len(entry['用户部门'])
                worker = entry['用户UID']
                # 找到上司并加入数据集
                # 只有部门长度大于1的才有leader！如果a部门长度比b部门长度多一位，且之前的都相同，则认为是b是a的上司
                if depart_len > 1:
                    cond1 = self.staff_df['depart_num'] == depart_len - 1
                    depart_list = [f'depart{x}' for x in range(1, depart_len)]
                    cond2 = True
                    for d in depart_list[:4]:  # 有253的个存在5个用户部门的
                        s1 = self.staff_df[d] == entry[d]
                        cond2 = cond2 & s1
                    leaders = self.staff_df[(cond1) & (cond2)]

                    for l_idx, leader in leaders.iterrows():
                        leader = leader["用户UID"]
                        if leader not in self.user or worker not in self.user:
                            # print(f"{leader} or {worker} not in training set!")
                            continue
                        else:
                            leader_id, w_id = self.user[leader], self.user[worker]
                            if (leader_id, w_id) not in self.interaction_set:
                                self.interaction_set.add((leader_id, w_id))
                                fp.write(str(leader_id) + ' ' + str(w_id) + '\n')
                                extra_relation_cnt += 1
                            else:
                                continue

                # 找到同部门同事并加入数据集
                condition1 = self.staff_df['depart_num'] == depart_len
                depart_list = [f'depart{x}' for x in range(1, depart_len + 1)]
                condition2 = True
                for d in depart_list[:4]:
                    s1 = self.staff_df[d] == entry[d]
                    condition2 = condition2 & s1
                depart_colleague = self.staff_df[(condition1) & (condition2)]

                for c_idx, colleague in depart_colleague.iterrows():
                    colleague = colleague["用户UID"]
                    if colleague not in self.user or worker not in self.user:
                        # print(f"{colleague} or {worker} not in training set!")
                        continue
                    else:
                        colleague_id, worker_id = self.user[colleague], self.user[worker]
                        if (colleague_id, worker_id) not in self.interaction_set:
                            self.interaction_set.add((colleague_id, worker_id))
                            fp.write(str(colleague_id) + ' ' + str(worker_id) + '\n')
                            extra_relation_cnt += 1
                        if (worker_id, colleague_id) not in self.interaction_set:
                            self.interaction_set.add((worker_id, colleague_id))
                            fp.write(str(worker_id) + ' ' + str(colleague_id) + '\n')
                            extra_relation_cnt += 1
            print(f'新增额外关系数据：{extra_relation_cnt}条')


    # 定义删除除字母,数字，汉字以外的所有符号的函数
    def remove_punctuation(self, line):
        line = str(line)
        if line.strip() == '':
            return ''
        rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
        line = rule.sub('', line)
        return line

    # 数据集划分
    def dataset_split(self, records_save_path, split_frac=(0.8, 0.1, 0.1)):
        print('Spliting dataset......')
        train_frac, dev_frac, test_frac = split_frac
        records = open(records_save_path, errors='ignore', encoding='utf-8').readlines()
        train_idx = int(train_frac * len(records))
        dev_idx = int((train_frac + dev_frac) * len(records))
        train = ''.join(records[: train_idx])
        dev = ''.join(records[train_idx: dev_idx])
        test = ''.join(records[dev_idx:])
        with open("../dataset/retweet_prediction/train.txt", 'w', encoding='utf-8') as fp:
            fp.write(train)
        with open("../dataset/retweet_prediction/dev.txt", 'w', encoding='utf-8') as fp:
            fp.write(dev)
        with open("../dataset/retweet_prediction/test.txt", 'w', encoding='utf-8') as fp:
            fp.write(test)

        # with open(records_save_path, errors='ignore', encoding='utf-8') as fp:
        # 	for idx, line in tqdm.tqdm(enumerate(fp)):
        # 		row = line.strip().split(' ')
        # 		if len(row) != 4:
        # 			title = [''.join(x) for x in row if row.index(x) not in [0, 1, len(row)-1]][0]
        # 			title = self.remove_punctuation(title)  # 去除标点以及进行分词
        # 			title = "".join([w for w in list(jieba.cut(title)) if w not in stopwords])
        # 			row = [row[0], row[1], title, row[len(row)-1]]
        # 			# print(row)
        # 		row[0] = int(row[0])
        # 		row[1] = int(row[1])
        # 		row[3] = int(row[3])
        # 		df.loc[idx] = row
        #
        # print(df.info())
        # train = df.sample(frac=0.7, random_state=123, axis=0)
        # test = df.loc[list(set(df.index) - set(train.index))]
        # train.to_csv("../dataset/retweet_prediction/train.txt", sep=' ', index=False, header=False)
        # dev.to_csv("../dataset/retweet_prediction/train.txt", sep=' ', index=False, header=False)
        # test.to_csv("../dataset/retweet_prediction/test.txt", sep=' ', index=False, header=False)
        # print(train.info())
        # print(test.info())


def main():
    # 加载初始文档
    data = pd.read_excel('../dataset/retweet_prediction/retweet_records.xlsx')
    staff = pd.read_excel('../dataset/retweet_prediction/staff.xlsx')
    records_save_path = '../dataset/retweet_prediction/retweets.txt'
    interaction_save_path = '../dataset/retweet_prediction/trust.txt'
    train_save_path = "../dataset/retweet_prediction/train.txt"
    ds = Dataset(data, staff)

    # 一对多 转化为 一对一
    flattened = True
    if not flattened:
        ds.flatten_record(records_save_path)

    # split the dataset
    split = False
    if flattened and not split:
        ds.dataset_split(records_save_path)

    # 生成关系网
    get_interaction = True
    if not get_interaction and split and flattened:
        ds.get_interaction_dataset(interaction_save_path, train_save_path)


if __name__ == '__main__':
    main()