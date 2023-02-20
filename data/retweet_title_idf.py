# 计算训练数据集公文标题的idf权重
import jieba, re
import math
import os
import pickle

# idf值统计方法
def train_idf(doc_list):
	# doc_list = open(file_path, encoding='utf-8', errors='ignore').readlines()
	idf_dic = {}
	# 总文档数
	tt_count = len(doc_list)

	# 每个词出现的文档数
	for doc in doc_list:
		for word in doc:
			idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

	# 按公式转换为idf值，分母加1进行平滑处理
	for k, v in idf_dic.items():
		idf_dic[k] = math.log(tt_count / (1.0 + v))

	# 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
	print("Total title count: " + str(tt_count))
	default_idf = math.log(tt_count / (1.0))
	return idf_dic, default_idf

def save_obj(obj, name ):
	with open('../dataset/retweet_prediction/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path, name):
	with open(file_path + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def main():
	# 标题路径
	doc_titles_save_path = "../dataset/retweet_prediction/doc_titles.txt"

	# 计算IDF
	idf_dic, default_idf = train_idf(doc_titles_save_path)
	save_obj(idf_dic, 'idf_dic')
	# idf_dic = load_obj('idf_dic')
	print(default_idf)


if __name__ == "__main__":
	main()