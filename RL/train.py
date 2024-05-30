import heapq
import json
import numpy as np
import random
import tensorflow as tf
from collections import deque
from RL.config import Config
from RL.metrics import cal_recall, cal_precision, cal_ndcg, cal_f1, cal_map
from RL.model import MyNet
from env import Simulator

def tmp_Q_eps_greedy(state, candidate, policy_net):
	policy_net1, policy_net2 = policy_net
	now_state = state.copy()
	mashup_id_list, selected_api_list, no_selected_api_list = now_state[0], now_state[1], now_state[2]
	actions = candidate.copy()
	all_ids = [i + 2906 for i in range(1322)]
	if len(actions) <= 2:
		return actions
	else:
		epsilon = 0.9
		# 贪婪策略
		coin = random.random()
		if coin < epsilon:
			cascading_api_list = []
			# 获取第一个api
			mask_ids1 = list(set(all_ids) - set(actions))
			out1 = policy_net1([tf.convert_to_tensor([mashup_id_list]),
								tf.convert_to_tensor([selected_api_list]),
								tf.convert_to_tensor([no_selected_api_list]),
								tf.convert_to_tensor([cascading_api_list])])
			out1 = out1.numpy()
			for mask in mask_ids1:
				out1[0][mask - 2906] = -10
			index1 = np.argmax(out1, axis=1)[0]
			a1 = all_ids[index1]
			if a1 not in actions:
				print(a1)
			# 缩小候选集空间
			actions.remove(a1)
			selected_api_list.append(a1)
			cascading_api_list.append(a1)
			# 获取第二个api
			mask_ids2 = list(set(all_ids) - set(actions))
			out2 = policy_net2([tf.convert_to_tensor([mashup_id_list]),
								tf.convert_to_tensor([selected_api_list]),
								tf.convert_to_tensor([no_selected_api_list]),
								tf.convert_to_tensor([cascading_api_list])])
			out2 = out2.numpy()
			for mask in mask_ids2:
				out2[0][mask - 2906] = -10
			index2 = np.argmax(out2, axis=1)[0]
			a2 = all_ids[index2]
			if a2 not in actions:
				print(a2)
			actions.remove(a2)
			cascading_api_list.append(a2)
			return cascading_api_list
		else:
			return list(random.sample(actions, 2))

def memory_sampling(memory, BATCH_SIZE):
	mini_batch = random.sample(memory, BATCH_SIZE)
	s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
	for transition in mini_batch:
		t_state, t_action, t_reward, t_next_state, t_done = transition
		s_lst.append(t_state)
		a_lst.append(t_action)
		r_lst.append(t_reward)
		s_prime_lst.append(t_next_state)
		done_mask_lst.append(t_done)
	return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst

def optimize_model(memory, policy_net, target_net, GAMMA, BATCH_SIZE):
	policy_net1, policy_net2 = policy_net
	target_net1, target_net2 = target_net
	state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory_sampling(memory, BATCH_SIZE)
	state_batch = np.array(state_batch, dtype=object)
	state_batch = [tf.convert_to_tensor(list(state_batch[:, 0])),
				   tf.ragged.constant(list(state_batch[:, 1])),
				   tf.ragged.constant(list(state_batch[:, -1])),
				   tf.convert_to_tensor([[]])]
	next_state_batch = np.array(next_state_batch, dtype=object)
	next_state_batch = [tf.convert_to_tensor(list(next_state_batch[:, 0])),
						tf.ragged.constant(list(next_state_batch[:, 1])),
						tf.ragged.constant(list(next_state_batch[:, -1])),
						tf.convert_to_tensor([[]])]

	next_state_values1 = target_net1(next_state_batch)
	next_state_values1 = next_state_values1.numpy()
	max_val_list1 = []
	for next_state_value in next_state_values1:
		max_val = max(next_state_value)
		max_val_list1.append(max_val)
	expected_state_action_values1 = next_state_values1
	for i in range(next_state_values1.shape[0]):
		for a in action_batch[i]:
			expected_state_action_values1[i][a - 2906] = max_val_list1[i] * GAMMA + reward_batch[i]
	expected_state_action_values1 = tf.convert_to_tensor(expected_state_action_values1)
	# loss 反传 梯度更新
	with tf.GradientTape() as tape:
		state_value1 = policy_net1(state_batch)
		loss1 = policy_net1.compiled_loss(expected_state_action_values1, state_value1)
	grad1 = tape.gradient(loss1, policy_net1.trainable_variables)
	policy_net1.optimizer.apply_gradients(zip(grad1, policy_net1.trainable_variables))
	policy_net1.compiled_metrics.update_state(expected_state_action_values1, state_value1)

	next_state_values2 = target_net2(next_state_batch)
	next_state_values2 = next_state_values2.numpy()
	max_val_list2 = []
	for next_state_value in next_state_values2:
		max_val = max(next_state_value)
		max_val_list2.append(max_val)
	expected_state_action_values2 = next_state_values2
	for i in range(next_state_values2.shape[0]):
		for a in action_batch[i]:
			expected_state_action_values2[i][a - 2906] = max_val_list2[i] * GAMMA + reward_batch[i]
	expected_state_action_values2 = tf.convert_to_tensor(expected_state_action_values2)
	with tf.GradientTape() as tape:
		state_value2 = policy_net2(state_batch)
		loss2 = policy_net2.compiled_loss(expected_state_action_values2, state_value2)
	grad2 = tape.gradient(loss2, policy_net2.trainable_variables)
	policy_net2.optimizer.apply_gradients(zip(grad2, policy_net2.trainable_variables))
	policy_net2.compiled_metrics.update_state(expected_state_action_values2, state_value2)


def train():
	memory = deque(maxlen=3000)
	config = Config()
	policy_net1 = MyNet(config)
	policy_net2 = MyNet(config)
	target_net1 = MyNet(config)
	target_net2 = MyNet(config)
	policy_net1.compile(optimizer='adam', loss='huber_loss', metrics=['accuracy'])
	policy_net2.compile(optimizer='adam', loss='huber_loss', metrics=['accuracy'])
	target_net1.compile(optimizer='adam', loss='huber_loss', metrics=['accuracy'])
	target_net2.compile(optimizer='adam', loss='huber_loss', metrics=['accuracy'])
	try:
		policy_net1.load_weights('../model/weight1/policy1')
		policy_net2.load_weights('../model/weight2/policy2')
		target_net1.load_weights('../model/weight3/target1')
		target_net2.load_weights('../model/weight4/target2')
		print("存在参数，加载成功!")
	except:
		print("不存在参数，加载失败!")
	policy_net = [policy_net1, policy_net2]
	target_net = [target_net1, target_net2]
	TARGET_UPDATE = 1000
	BATCH_SIZE = 128
	GAMMA = 0.9
	simulator = Simulator(mode='train')
	num_users = len(simulator)  # 2325
	total_step_count = 0
	with open('../data/top2_dict.json', 'r') as f:
		top5_dict = json.load(f)
	f.close()
	for e in range(500):
		preds = []
		trues = []
		total_round = 0
		for u in range(num_users):
			n_round = 0
			mashup_id, api_ids,  all_simulator_apis= simulator.get_data(u)
			trues.append(len(api_ids))
			pred = []
			all_apis = all_simulator_apis.copy()
			done = False
			mashup_id_list = [mashup_id]
			selected_api_list = []
			no_selected_api_list = []
			while not done:
				if n_round == 0:
					action = top5_dict[str(mashup_id)]
				else:
					# Recommendation using epsilon greedy policy
					action = tmp_Q_eps_greedy(state=[mashup_id_list, selected_api_list.copy(), no_selected_api_list], candidate=all_apis.copy(), policy_net=policy_net)
				# Compute reward
				reward = simulator.step([mashup_id_list, selected_api_list], action, n_round)
				# Determine whether you can stop
				next_select_api_list = selected_api_list.copy()
				next_no_select_api_list = []
				for a_action in action:
					if a_action in all_apis:
						all_apis.remove(a_action)
					if a_action in api_ids:
						next_select_api_list.append(a_action)
						pred.append(1)
					else:
						next_no_select_api_list.append(a_action)
						pred.append(0)
				if set(api_ids) == set(next_select_api_list) or len(all_apis) == 0:
					done = True
				if (done == True and n_round == 0) or (n_round == 1):
					preds.append(pred.copy())
				# Store transition to buffer
				Tuple = ([mashup_id_list, selected_api_list, no_selected_api_list], action, reward, [mashup_id_list, next_select_api_list, next_no_select_api_list], done)
				memory.append(Tuple)
				# Q learning
				# target update
				if len(memory) > 200:
					optimize_model(memory, policy_net, target_net, GAMMA, BATCH_SIZE)
				if len(memory) > 200 and total_step_count % TARGET_UPDATE == 0:
					target_net1.set_weights(policy_net1.get_weights())
					target_net2.set_weights(policy_net2.get_weights())
				total_step_count += 1
				# update state
				selected_api_list = next_select_api_list.copy()
				no_selected_api_list = next_no_select_api_list.copy()
				# print(selected_api_list, no_selected_api_list)
				n_round += 1
			total_round += n_round
			print('mashup_id:%d, round:%d' %(mashup_id, n_round))
		# calculate recall and average round of recommendation
		recall = 'epoch %d R@10:' % e + str(cal_recall(preds, trues))
		round = 'average round of epoch %d:' % e + str(total_round / num_users)
		print(recall)
		print(round)
		target_net1.set_weights(policy_net1.get_weights())
		target_net2.set_weights(policy_net2.get_weights())
		policy_net1.save_weights('../model/weight1/policy1')
		policy_net2.save_weights('../model/weight2/policy2')
		target_net1.save_weights('../model/weight3/target1')
		target_net2.save_weights('../model/weight4/target2')

def test(policy_net):
	simulator = Simulator(mode='test')
	num_users = len(simulator)
	with open('../data/top2_dict.json', 'r') as f:
		top5_dict = json.load(f)
	f.close()
	preds = []
	trues = []
	total_round = 0
	all_candidate_apis = [i + 2906 for i in range(1322)]
	for u in range(num_users):
		n_round = 0
		mashup_id, api_ids, all_simulator_apis = simulator.get_data(u)
		trues.append(len(api_ids))
		pred = []
		all_apis = all_candidate_apis.copy()
		done = False
		mashup_id_list = [mashup_id]
		selected_api_list = []
		no_selected_api_list = []
		while not done:
			if n_round == 0:
				action = top5_dict[str(mashup_id)]
			else:
				# Recommendation using epsilon greedy policy
				action = tmp_Q_eps_greedy(state=[mashup_id_list, selected_api_list.copy(), no_selected_api_list], candidate=all_apis.copy(), policy_net=policy_net)
			# Determine whether you can stop
			next_select_api_list = selected_api_list.copy()
			next_no_select_api_list = []
			for a_action in action:
				if a_action in all_apis:
					all_apis.remove(a_action)
				if a_action in api_ids:
					next_select_api_list.append(a_action)
					pred.append(1)
				else:
					next_no_select_api_list.append(a_action)
					pred.append(0)
			if set(api_ids) == set(next_select_api_list) or len(all_apis) == 0:
				done = True
			if (done == True and (n_round==0 or n_round == 1 or n_round == 2 or n_round == 3)) or (n_round == 4):
				preds.append(pred.copy())
			# update state
			selected_api_list = next_select_api_list.copy()
			no_selected_api_list = next_no_select_api_list.copy()
			n_round += 1
		total_round += n_round
		print('mashup_id:%d, round:%d' %(mashup_id, n_round))
	# calculate recall and average round of recommendation
	recall = 'Recall@10:' + str(cal_recall(preds, trues))
	precision = 'Precision@10:' + str(cal_precision(preds))
	ndcg = 'NDCG@10:' + str(cal_ndcg(preds))
	f1 = 'F1@10:' + str(cal_f1(preds, trues))
	map = 'MAP@10:' + str(cal_map(preds, trues))
	round = 'average round:' + str(total_round / num_users)
	print(precision)
	print(recall)
	print(ndcg)
	print(f1)
	print(map)
	print(round)

if __name__ == '__main__':
	print('Train...')
	train()
	'''config = Config()
	policy_net1 = MyNet(config)
	policy_net2 = MyNet(config)
	policy_net1.compile(optimizer='adam', loss='huber_loss', metrics=['accuracy'])
	policy_net2.compile(optimizer='adam', loss='huber_loss', metrics=['accuracy'])
	try:
		policy_net1.load_weights('../model/weight1/policy1')
		policy_net2.load_weights('../model/weight2/policy2')
		print("存在参数，加载成功!")
	except:
		print("不存在参数，加载失败!")
	policy_net = [policy_net1, policy_net2]
	test(policy_net)'''