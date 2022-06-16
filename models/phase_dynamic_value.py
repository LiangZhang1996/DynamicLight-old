"""
DynamicLight that use Max State-Value to determine the phase
Input shape: [batch, max_lane*4]
Created by Liang Zhang
"""
from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda, Subtract, Add, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
from tensorflow.keras import backend as K
import numpy as np
import random
import tensorflow as tf


class DynamicAgentValue(NetworkAgent):

    def build_network(self):
        ins0 = Input(shape=(self.max_lane*4, ))

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
        phase_feat_all = MultiHeadAttention(4, 8, attention_axes=1)(phase_feat_all, phase_feat_all)

        hidden = Dense(20, activation="relu")(phase_feat_all)
        hidden = Dense(20, activation="relu")(hidden)
        q_values = self.dueling_block(hidden)
        network = Model(inputs=ins0,
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()
        return network

    def dueling_block(self, inputs):
        tmp_v = Dense(20, activation="relu", name="dense_values")(inputs)
        # [bath, phase, 1]
        value = Dense(1, activation="linear", name="dueling_values")(tmp_v)
        tmp_a = Dense(20, activation="relu", name="dense_a")(inputs)
        # [batch, phase, dur]
        a = Dense(self.num_action_dur, activation="linear", name="dueling_advantages")(tmp_a)
        mean = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(a)
        advantages = Subtract()([a, mean])
        # [batch, phase, num_dur]
        q_values = Add(name='dueling_q_values')([value, advantages])
        return q_values

    def choose_action(self, states, list_need):
        phase_feat = []
        for s in states:
            feat1 = s[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][1]]
            phase_feat.append(feat1)
        # [batch, phase, dur]
        q_values = self.q_network.predict(np.array(phase_feat))
        # values: [batch, 4]
        values = np.mean(q_values, axis=-1)
        # select phase
        phase = np.argmax(values, axis=-1)

        # select corresponding Q-values
        q_values_s = []
        for i in range(len(q_values)):
            if random.random() <= self.dic_agent_conf["EPSILON"]:
                tmp_p = np.random.randint(self.num_phases)
                phase[i] = tmp_p
            else:
                tmp_p = phase[i]
            q_values_s.append(q_values[i, tmp_p])
        # [batch, durs]
        q_values_s = np.array(q_values_s)
        action = self.epsilon_choice(q_values_s)
        return phase, action

    def epsilon_choice(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_action_dur, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act

    def phase_control_policy(self, feat0):
        if self.num_lane == 8:
            feat10 = feat0[1] + feat0[3]
            feat20 = feat0[5] + feat0[7]
            feat30 = feat0[0] + feat0[2]
            feat40 = feat0[4] + feat0[6]
        elif self.num_lane == 10:
            feat10 = feat0[1] + feat0[4]
            feat20 = feat0[7] + feat0[9]
            feat30 = feat0[0] + feat0[3]
            feat40 = feat0[6] + feat0[8]
        elif self.num_lane == 12:
            feat10 = feat0[1] + feat0[4]
            feat20 = feat0[7] + feat0[10]
            feat30 = feat0[0] + feat0[3]
            feat40 = feat0[6] + feat0[9]
        elif self.num_lane == 16:
            feat10 = feat0[1] + feat0[2] + feat0[5] + feat0[6]
            feat20 = feat0[9] + feat0[10] + feat0[13] + feat0[14]
            feat30 = feat0[0] + feat0[4]
            feat40 = feat0[8] + feat0[12]
        idx = np.argmax([feat10, feat20, feat30, feat40])

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
            _action1.append(action1)
            _action2.append(action2)
            _reward.append(reward)

        # well prepared states
        _state2 = np.array(_state)
        _next_state2 = np.array(_next_state)

        cur_qvalues = self.q_network.predict(_state2)
        next_qvalues = self.q_network_bar.predict(_next_state2)
        # [batch, 4]
        target = np.copy(cur_qvalues)
        for i in range(len(sample_slice)):
            target[i, _action1[i], _action2[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(next_qvalues[i, _action1[i], :])
        self.Xs = _state2
        self.Y = target
