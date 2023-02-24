'''
用于下一环节处理人推荐的MHCN模型
把通道数改为2通道，删除通道purchase
'''
from data import retweet_title_idf
import os, pickle
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=Warning)
import numpy as np
from base.graph_recommender import GraphRecommender
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from util.loss_tf import bpr_loss
from data.social import Relation
from base.tf_interface import TFGraphInterface
from util.sampler import next_batch_pairwise
from util.conf import OptionConf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from gensim.test.utils import common_texts,get_tmpfile
from gensim.models import Word2Vec

# paper: Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. WWW'21

class MHCN(GraphRecommender):
	def __init__(self, conf, training_set, test_set, dev_data, **kwargs):
		GraphRecommender.__init__(self, conf, training_set, test_set, dev_data, **kwargs)  # 得到self.data
		args = OptionConf(self.config['MHCN'])
		self.n_layers = int(args['-n_layer'])
		self.ss_rate = float(args['-ss_rate'])
		self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
		self.data.by_item = True  # True为给每个item推荐候选用户，False为给每个用户推荐商品

	def print_model_info(self):
		super(MHCN, self).print_model_info()
		# print social relation statistics
		print('Social data size: (user number: %d, relation number: %d).' % (self.social_data.size()))
		print('=' * 80)

	def build_hyper_adj_mats(self):
		S = self.social_data.get_social_mat()
		Y = self.data.interaction_mat
		B = S.multiply(S.T)
		U = S - B
		C1 = (U.dot(U)).multiply(U.T)
		A1 = C1 + C1.T
		C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
		A2 = C2 + C2.T
		C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
		A3 = C3 + C3.T
		A4 = (B.dot(B)).multiply(B)
		C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
		A5 = C5 + C5.T
		A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
		A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
		A8 = (Y.dot(Y.T)).multiply(B)
		A9 = (Y.dot(Y.T)).multiply(U)
		A9 = A9 + A9.T
		# A10 = Y.dot(Y.T) - A8 - A9
		# addition and row-normalization
		H_s = sum([A1,A2,A3,A4,A5,A6,A7])
		H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1))
		H_j = sum([A8,A9])
		H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
		# H_p = A10
		# H_p = H_p.multiply(H_p>3)
		# H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
		return [H_s, H_j]

	def get_item_embeddings(self):
		# def get_masks(tokens, max_seq_length):
		#     """Mask for padding"""
		#     if len(tokens) > max_seq_length:
		#         raise IndexError("Token length more than max seq length!")
		#     return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))
		#
		# def get_segments(tokens, max_seq_length):
		#     """Segments: 0 for the first sequence, 1 for the second"""
		#     if len(tokens) > max_seq_length:
		#         raise IndexError("Token length more than max seq length!")
		#     segments = []
		#     current_segment_id = 0
		#     for token in tokens:
		#         segments.append(current_segment_id)
		#         if token == "[SEP]":
		#             current_segment_id = 1
		#     return segments + [0] * (max_seq_length - len(tokens))
		#
		# def get_ids(tokens, tokenizer, max_seq_length):
		#     """Token ids from Tokenizer vocab"""
		#     token_ids = tokenizer.convert_tokens_to_ids(tokens)
		#     input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
		#     return input_ids
		#
		# def get_embed(s, fun_tokenizer):
		#     stokens = fun_tokenizer.tokenize(s)
		#     stokens = ["[CLS]"] + stokens + ["[SEP]"]
		#     input_ids = get_ids(stokens, fun_tokenizer, max_seq_length)
		#     input_masks = get_masks(stokens, max_seq_length)
		#     input_segments = get_segments(stokens, max_seq_length)
		#
		#     pool_embs, all_embs = bert_model.predict([[input_ids], [input_masks], [input_segments]])
		#     return pool_embs, all_embs

		# 根据word2vec和tfidf获取句子矩阵
		def get_sentence_matrix(sentence_list):
			# 获取idf字典和默认idf值
			doc_titles_save_path = "./dataset/retweet_prediction/doc_titles_cut.txt"
			self.idf_dic, self.default_idf = retweet_title_idf.train_idf(self.data.train_titles)

			sentences_matrix = []
			# 平均特征矩阵
			index = 0
			for sent in sentence_list:
				words_matrix = []
				word_weight = []
				# 得出句子中各个词的特征向量，形成一个矩阵，然后与idf权重相乘，就得到该句子的特征向量
				for word in sent:
					if word in self.w2v_model.wv.key_to_index.keys():
						words_matrix.append(self.w2v_model.wv.get_vector(word))
						if word in self.idf_dic:
							word_weight.append(self.idf_dic[word])  # 保存对应文字的idf权重
						else:
							word_weight.append(self.default_idf)

				# 根据idf权重求句向量
				words_matrix_np = np.array(words_matrix)  # N行，300列， 300是w2v模型的vector_size，可变化
				word_weight_np = np.array(word_weight, ndmin=2)  # 1行N列，N表示句子字数
				sentence_vec = np.matmul(word_weight_np, words_matrix_np)  # 输出为 1 X vector_size的数组
				# 以下三行用于测试
				# if sentence_vec.shape != (1, self.emb_size):
				# 	print("sentence"+sent+"end, 长为",len(sent))
				# 	print(sentence_vec.shape)
				sentences_matrix.append(sentence_vec)
				index += 1
			sentences_matrix = np.asarray(sentences_matrix)
			sentences_matrix = sentences_matrix.squeeze(axis=1)
			return sentences_matrix

		# ==============hugging face 预训练模型================
		# BERT_PATH = r"D:\05 STARWARP\code\SELFRec_MHCN\model\pretrained_model"
		# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
		# model = TFBertModel.from_pretrained("bert-base-chinese")
		# # text = "Replace me by any text you'd like."
		# encoded_input = tokenizer(self.train_titles, return_tensors='tf')
		# pooled_embeddings, sequence_embeddings = model(encoded_input)
		# embeddings = tf.Variable(pooled_embeddings)

		# ================tensorflow-hub预训练模型==================
		# max_seq_length = self.emb_size - 2  # 两个位置要让给CLS和SEP符号
		# input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
		# input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
		# segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
		# bert_layer = hub.KerasLayer(BERT_PATH, trainable=False)
		# pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
		# bert_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids],
		#                                    outputs=[pooled_output, sequence_output])
		# tokenizer = FullTokenizer(r'bert_zh_L-12_H-768_A-12_1\assets\vocab.txt')

		# pooled_embeddings, sequence_embeddings = get_embed(self.data.train_titles, tokenizer)  # pool_out: shape=[batch, 768]；sequence_out: shape=[batch, 256, 768]
		# linear_layer_weights = tf.Variable(initializer([pooled_embeddings.shape[1], self.emb_size]))
		# linear_layer_biases = tf.Variable(initializer([self.emb_size]))
		# embeddings = tf.matmul(pooled_embeddings, linear_layer_weights) + linear_layer_biases  # 维度形状不确定，输出是[batch, self.emb_size]

		self.w2v_model = Word2Vec(self.data.train_titles, vector_size=self.emb_size)
		output_array = get_sentence_matrix(self.data.train_titles)
		embedding = tf.convert_to_tensor(output_array, dtype=tf.float32)
		return embedding

	# 相当于pytorch的forward函数
	def build(self):
		self.weights = {}
		self.n_channel = 3
		self.u_idx = tf.compat.v1.placeholder(tf.int32, name="u_idx")
		self.v_idx = tf.compat.v1.placeholder(tf.int32, name="v_idx")
		self.neg_idx = tf.compat.v1.placeholder(tf.int32, name="neg_holder")
		self.batch_labels = tf.compat.v1.placeholder(tf.int32, name="batch_labels")

		initializer = tf.contrib.layers.xavier_initializer()  # xavier初始化方法，正态分布的一种，使得每一层输出的方差应该尽量相等
		self.user_embeddings = tf.Variable(initializer([self.data.user_num, self.emb_size]))  # 创建了尺寸为(user_num, emb_size)的tensor，user_num为用户总数
		# 改造：item_embeddings
		# self.item_embeddings = tf.Variable(initializer([self.data.item_num, self.emb_size]))
		self.item_embeddings = self.get_item_embeddings()

		# define learnable paramters
		for i in range(self.n_channel):
			self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='g_W_%d_1' % (i + 1))
			self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]), name='g_W_b_%d_1' % (i + 1))
			self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='sg_W_%d_1' % (i + 1))
			self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]), name='sg_W_b_%d_1' % (i + 1))
		self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
		self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')
		tf_config = tf.compat.v1.ConfigProto()
		tf_config.gpu_options.allow_growth = True
		self.sess = tf.compat.v1.Session(config=tf_config)

		# define inline functions
		def self_gating(em, channel):
			return tf.multiply(em, tf.nn.sigmoid(tf.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' % channel]))

		def self_supervised_gating(em, channel):
			return tf.multiply(em, tf.nn.sigmoid(tf.matmul(em, self.weights['sgating%d' % channel]) + self.weights['sgating_bias%d' % channel]))

		def channel_attention(*channel_embeddings):
			weights = []
			for embedding in channel_embeddings:
				weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), 1))
			score = tf.nn.softmax(tf.transpose(weights))
			mixed_embeddings = 0
			for i in range(len(weights)):
				mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
			return mixed_embeddings, score

		# initialize adjacency matrices
		M_matrices = self.build_hyper_adj_mats()
		H_s = M_matrices[0]
		H_s = TFGraphInterface.convert_sparse_mat_to_tensor(H_s)
		H_j = M_matrices[1]
		H_j = TFGraphInterface.convert_sparse_mat_to_tensor(H_j)
		# H_p = M_matrices[2]
		# H_p = TFGraphInterface.convert_sparse_mat_to_tensor(H_p)
		R = TFGraphInterface.convert_sparse_mat_to_tensor(self.data.normalize_graph_mat(self.data.interaction_mat))
		# self-gating
		user_embeddings_c1 = self_gating(self.user_embeddings, 1)
		user_embeddings_c2 = self_gating(self.user_embeddings, 2)
		# user_embeddings_c3 = self_gating(self.user_embeddings, 3)
		simple_user_embeddings = self_gating(self.user_embeddings, 3)
		all_embeddings_c1 = [user_embeddings_c1]
		all_embeddings_c2 = [user_embeddings_c2]
		# all_embeddings_c3 = [user_embeddings_c3]
		all_embeddings_simple = [simple_user_embeddings]
		item_embeddings = self.item_embeddings
		all_embeddings_i = [item_embeddings]

		self.ss_loss = 0  # 自监督损失
		# multi-channel convolution 多通道卷积
		for k in range(self.n_layers):
			mixed_embedding = channel_attention(user_embeddings_c1, user_embeddings_c2)[0] + simple_user_embeddings / 2
			# Channel S
			user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s, user_embeddings_c1)
			norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
			all_embeddings_c1 += [norm_embeddings]
			# Channel J
			user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c2)
			norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
			all_embeddings_c2 += [norm_embeddings]
			# Channel P
			# user_embeddings_c3 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c3)
			# norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
			# all_embeddings_c3 += [norm_embeddings]
			# item convolution
			new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding)
			norm_embeddings = tf.math.l2_normalize(new_item_embeddings, axis=1)
			all_embeddings_i += [norm_embeddings]
			simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
			all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
			item_embeddings = new_item_embeddings

		# averaging the channel-specific embeddings
		user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
		user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
		simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
		item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
		# aggregating channel-specific embeddings
		self.final_item_embeddings = item_embeddings
		self.final_user_embeddings, self.attention_score = channel_attention(user_embeddings_c1, user_embeddings_c2)
		self.final_user_embeddings += simple_user_embeddings / 2

		# create self-supervised loss 自监督损失,以下是by_user的自监督损失
		self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings, 1), H_s)
		self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings, 2), H_j)
		self.ss_loss = self.ss_rate * self.ss_loss

		# embedding look-up 从one_hot到矩阵编码的转换过程需要在embedding进行查找，就是把对应的id的用户表示提取出来
		# self.final_item_embeddings是最终用户表示，self.neg_idx是负id
		# 以下是by_user的embedding
		# self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
		# self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
		# self.batch_pos_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)

		# 以下是by_item的embedding
		self.batch_neg_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.neg_idx)
		self.batch_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)
		self.batch_pos_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)

		# TODO: 将处理人和item embedding拼接起来，做二分类判断，得到logits
		self.concat_embeddings = tf.concat(values=[self.user_embeddings, self.item_embeddings], axis=1)
		self.fcn_w1 = tf.Variable(initializer([self.emb_size, 128]))
		self.fcn_b1 = tf.Variable(initializer([1, 128]))
		self.xw1_plus_b1 = tf.matmul(self.concat_embeddings, self.fcn_w1) + self.fcn_b1
		self.fcc_output1 = tf.sigmoid(self.xw1_plus_b1)

		self.fcn_w2 = tf.Variable(initializer([128, 2]))
		self.fcn_b2 = tf.Variable(initializer([1, 2]))
		self.xw2_plus_b2 = tf.matmul(self.fcc_output1, self.fcn_w2) + self.fcn_b2
		self.fcc_output2 = tf.sigmoid(self.xw2_plus_b2)
		self.logits = tf.argmax(input=self.fcc_output2, axis=1)  # TODO:要不要加tf.argmax

	# 自监督损失
	def hierarchical_self_supervision(self, em, adj):
		def row_shuffle(embedding):
			return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
		def row_column_shuffle(embedding):
			corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
			corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
			return corrupted_embedding
		def score(x1, x2):
			return tf.reduce_sum(tf.multiply(x1, x2), 1)
		user_embeddings = em
		# user_embeddings = tf.math.l2_normalize(em,1)
		edge_embeddings = tf.sparse_tensor_dense_matmul(adj, user_embeddings)
		# Local MIM
		pos = score(user_embeddings, edge_embeddings)
		neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
		neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
		local_loss = tf.reduce_sum(-tf.math.log(tf.sigmoid(pos - neg1)) - tf.math.log(tf.sigmoid(neg1 - neg2)))
		# Global MIM
		graph = tf.reduce_mean(edge_embeddings, 0)
		pos = score(edge_embeddings, graph)
		neg1 = score(row_column_shuffle(edge_embeddings), graph)
		global_loss = tf.reduce_sum(-tf.math.log(tf.sigmoid(pos - neg1)))
		return global_loss + local_loss


	def train(self):
		# TODO：labels需要placeholder，
		rec_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.batch_labels, logits=)
		# rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
		reg_loss = 0
		for key in self.weights:
			reg_loss += self.reg * tf.nn.l2_loss(self.weights[key])
		reg_loss += self.reg * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_neg_item_emb) + tf.nn.l2_loss(self.batch_pos_item_emb))
		total_loss = rec_loss + reg_loss + self.ss_loss
		opt = tf.compat.v1.train.AdamOptimizer(self.lRate)
		train_op = opt.minimize(total_loss)
		init = tf.compat.v1.global_variables_initializer()
		self.sess.run(init)  # 启动默认会话，训练神经网络

		# Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
		# session.run()函数：Runs operations and evaluates tensors in `fetches`， fetches是从计算图中取出对应变量的参数，
		# 可以是单个图元素、任意的列表、元组、字典等等形式的图元素。图元素包括操作、张量、稀疏张量、句柄、字符串等等。
		# TODO：修改
		for epoch in range(self.maxEpoch):
			for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
				user_idx, i_idx, j_idx = batch
				_, loss1, loss2 = self.sess.run([train_op, rec_loss, self.ss_loss], feed_dict={self.u_idx: user_idx, self.batch_labels: j_idx, self.v_idx: i_idx})
				print('training:', epoch + 1, 'batch', n, 'rec loss:', loss1, 'ssl loss',loss2)
			self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])
			self.fast_evaluation_by_item(epoch)
		self.U, self.V = self.best_user_emb, self.best_item_emb

	def train_by_item(self):
		rec_loss = bpr_loss(self.batch_item_emb, self.batch_pos_user_emb, self.batch_neg_user_emb)
		reg_loss = 0
		for key in self.weights:
			reg_loss += self.reg * tf.nn.l2_loss(self.weights[key])
		reg_loss += self.reg * (
					tf.nn.l2_loss(self.batch_item_emb) + tf.nn.l2_loss(self.batch_neg_user_emb) + tf.nn.l2_loss(
				self.batch_pos_user_emb))
		total_loss = rec_loss + reg_loss + self.ss_loss
		opt = tf.compat.v1.train.AdamOptimizer(self.lRate)
		train_op = opt.minimize(total_loss)
		init = tf.compat.v1.global_variables_initializer()
		self.sess.run(init)  # 启动默认会话，训练神经网络

		# Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
		# session.run()函数：Runs operations and evaluates tensors in `fetches`， fecches是从计算图中取出对应变量的参数，
		# 可以是单个图元素、任意的列表、元组、字典等等形式的图元素。图元素包括操作、张量、稀疏张量、句柄、字符串等等。
		for epoch in range(self.maxEpoch):
			for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
				user_idx, i_idx, j_idx = batch
				_, l1, l2 = self.sess.run([train_op, rec_loss, self.ss_loss],
										  feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
				print('training:', epoch + 1, 'batch', n, 'rec loss:', l1, 'ssl loss', l2)
			self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])
			self.fast_evaluation_by_item(epoch)
		self.U, self.V = self.best_user_emb, self.best_item_emb

	def save(self):
		self.best_user_emb, self.best_item_emb = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])

	def predict(self, u):
		u = self.data.get_user_id(u)
		return self.V.dot(self.U[u])

	def predict_by_item(self, v):
		v = self.data.get_item_id(v)
		return self.U.dot(self.V[v])  # 但是item可能不存在

