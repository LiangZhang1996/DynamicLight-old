from models.phase_dynamic_0_1 import DynamicAgent01
from models.phase_dynamic_0_2 import DynamicAgent02
from models.phase_dynamic_0_3 import DynamicAgent03
from models.phase_dynamic_0_4 import DynamicAgent04
from models.phase_dynamic_0_5 import DynamicAgent05
from models.phase_dynamic_cycle import DynamicAgentCycle
from models.phase_dynamic_1_1 import DynamicAgent11
from models.phase_dynamic_value import DynamicAgentValue
from models.phase_dynamic_lite import DynamicAgentLite
from models.phase_dynamic_lite_2 import DynamicAgentLite2
from models.phase_dynamic_multi import DynamicAgentMulti
from models.phase_dynamic_multi_a import DynamicAgentMultiA
from models.phase_dynamic_mp import DynamicAgentMP



DIC_AGENTS = {
    "Dyn01": DynamicAgent01,
    "Dyn02": DynamicAgent02,
    "Dyn03": DynamicAgent03,
    "Dyn04": DynamicAgent04,
    "Dyn05": DynamicAgent05,
    "Dyn11": DynamicAgent11,
    "DynC": DynamicAgentCycle,
    "DynV": DynamicAgentValue,
    "DynMP": DynamicAgentMP,
    "DynLite": DynamicAgentLite,
    "DynLite2": DynamicAgentLite2,
    "DynM": DynamicAgentMulti,
    "DynMA": DynamicAgentMultiA
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {

    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "MAX_LANE": 16,

    "OBS_LENGTH": 167,
    "OBS_LENGTH_Q": 80,
    "MIN_ACTION_TIME": 15,
    "MEASURE_TIME": 15,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "ACTION_DURATION": {
                0: 10,
                1: 15,
                2: 20,
                3: 25,
                4: 30,
                5: 35,
                6: 40
            },

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle",
        "lane_num_vehicle_downstream",
        "traffic_movement_pressure_num",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "pressure",
        "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}
