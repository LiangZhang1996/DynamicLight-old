from .config import DIC_AGENTS
from .cityflow_env import CityFlowEnv
import time
import os
import copy


class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
            start_time = time.time()
            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=self.cnt_round,
                    intersection_id=str(i)
                )
                self.agents[i] = agent
            print("Create intersection agent time: ", time.time()-start_time)

        self.env = CityFlowEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf
        )

    def generate(self):
        reset_env_start_time = time.time()
        state, step_time, list_need = self.env.reset()
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()
        while step_time < self.dic_traffic_env_conf["RUN_COUNTS"]:
            step_start_time = time.time()
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
                phase_action, duration_action = self.agents[i].choose_action(state, list_need)
            # print(phase_action, duration_action)
            next_state, step_time, list_need = self.env.step(phase_action, duration_action)

            print("time: {0}, running_time: {1}".format(self.env.get_current_time(),
                                                        time.time()-step_start_time))

            state = next_state

        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("start logging.......................")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time
        self.env.end_cityflow()
        print("reset_env_time: ", reset_env_time)
        print("running_time_total: ", running_time)
        print("log_time: ", log_time)
