"""
DynamicLight-Cycle
Input shape: [batch, max_lane*4]
Created by Liang Zhang
"""
from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda,  Subtract, Add, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
from tensorflow.keras import backend as K
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class DynamicAgentCycle(NetworkAgent):

    def build_network(self):
        ins0 = Input(shape=(self.max_lane*4, ))
        ins1 = Input(shape=(1, self.num_phases))

        feat1 = Reshape((self.max_lane, 4, 1))(ins0)
        feat1 = Dense(4, activation="sigmoid")(feat1)
        feat1 = Reshape((self.max_lane, 16))(feat1)

        lane_feats_s = tf.split(feat1, self.max_lane, axis=1)
        MHA1 = MultiHeadAttention(4, 8, attention_axes=1)
        Mean1 = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))
        phase_feats_map_2 = []
        for i in range(self.num_phases):
            tmp_feat_1 = tf.concat([lane_feats_s[idx] for idx in self.phase_map[i]], axis=1)
            tmp_feat_2 = MHA1(tmp_feat_1, tmp_feat_1)
            tmp_feat_3 = Mean1(tmp_feat_2)
            phase_feats_map_2.append(tmp_feat_3)

        # [batch, num_phase, dim]
        phase_feat_all = tf.concat(phase_feats_map_2, axis=1)
        selected_phase_feat = Lambda(lambda x: tf.matmul(x[0], x[1]))([ins1, phase_feat_all])
        selected_phase_feat = Reshape((16, ))(selected_phase_feat)

        hidden = Dense(20, activation="relu")(selected_phase_feat)
        hidden = Dense(20, activation="relu")(hidden)
        q_values = self.dueling_block(hidden)
        network = Model(inputs=[ins0, ins1],
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()
        return network

    def dueling_block(self, inputs):
        tmp_v = Dense(20, activation="relu", name="dense_values")(inputs)
        value = Dense(1, activation="linear", name="dueling_values")(tmp_v)
        tmp_a = Dense(20, activation="relu", name="dense_a")(inputs)
        a = Dense(self.num_action_dur, activation="linear", name="dueling_advantages")(tmp_a)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantages = Subtract()([a, mean])
        q_values = Add(name='dueling_q_values')([value, advantages])
        return q_values

    def phase_index2matrix(self, phase_index):
        # [batch, 1] -> [batch, 1, num_phase]
        lab = to_categorical(phase_index, num_classes=self.num_phases)
        return lab

    def choose_action(self, states, list_need):
        phase = []
        phase2 = []
        phase_feat = []
        for s, inter_id in zip(states, list_need):
            feat1 = s[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][1]]
            # feat0 = s[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][0]]
            tmp_idx = self.phase_control_policy(inter_id)
            phase.append([[tmp_idx]])
            phase2.append(tmp_idx)
            phase_feat.append(feat1)

        phase_feat2, phase_idx = np.array(phase_feat), np.array(phase)
        phase_matrix = self.phase_index2matrix(phase_idx)
        q_values = self.q_network.predict([phase_feat2, phase_matrix])
        action = self.epsilon_choice(q_values)
        return phase2, action

    def epsilon_choice(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_action_dur, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act

    def phase_control_policy(self, inter_id):
        idx = self.cyclicInd2[inter_id]
        if self.cyclicInd2[inter_id] == self.num_phases - 1:
            self.cyclicInd2[inter_id] = 0
        else:
            self.cyclicInd2[inter_id] += 1
        return idx

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        #  used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        _state = []
        _next_state = []
        _action1 = []  # phase index
        _action2 = []
        _reward = []
        #  use average reward
        for i in range(len(sample_slice)):
            state, action1, action2, next_state, reward, _ = sample_slice[i]
            _state.append(state)
            _next_state.append(next_state)
            _action1.append([[action1]])
            _action2.append(action2)
            _reward.append(reward)

        # well prepared states
        _state2 = np.array(_state)
        _next_state2 = np.array(_next_state)

        phase_matrix = self.phase_index2matrix(np.array(_action1))

        cur_qvalues = self.q_network.predict([_state2, phase_matrix])
        next_qvalues = self.q_network_bar.predict([_next_state2, phase_matrix])
        # [batch, 4]
        target = np.copy(cur_qvalues)
        for i in range(len(sample_slice)):
            target[i, _action2[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(next_qvalues[i, :])
        self.Xs = [_state2, phase_matrix]
        self.Y = target
