# 1. Introduction

Official code for article <DynamicLight: Dynamically tuning traffic signal duration based on deep reinforcement learning for signalized intersections>.


# 2. Requirements

`python3.6`,`tensorflow=2.4`, `cityflow`, `pandas`, `numpy`

[`cityflow`](https://github.com/cityflow-project/CityFlow.git) needs a linux environment, and we run the code on Manjaro Linux.

# 3. Quick start

## 3.1 Base experiemnts

- For `DynamicLight`, run:
```shell
python run_dynamic.py
```
- For `DynamicLight-Lite`, run:
```shell
python run_dynamic_lite.py
```
- For tranfer, configure and run `run_test.py`


## 3.2 Extended experiemnts

### 3.2.1 Different state representations

- For `number of vehicles`, `traffic movement pressure`, and `queue length`, you should change the used feature name in `run_dynamic1.py` and run it.
- For `number of vehicles under segmented road`, just run `run_dynamic.py`.

### 3.2.2 Different neural networks

As described in the article,
- For `Network 1`, run `run_dynamic.py`
- For `Network 2`, change the model name as `DynM` in `run_dynamic_multi.py` and run it
- For `Network 3`, change the model name as `DynMA` in `run_dynamic_multi.py` and run it

### 3.2.3 Different duration action spaces

You should configure the action space in `run_dynamic.py` and run it.

### 3.2.4 Different phase control methods

- For `M-QL`, run `run_dynamic.py`
- For `Efficient-MP`, run `run_dynamic_mp.py`
- For `M-SV` with queue length as the reward, configure the reward and run `run_dynamic_value.py`
- For `M-SV` with presure as the reward, configure the reward and run `run_dynamic_value.py`

### 3.2.5 Differnt feature fusion methods

Configure the model name as <`Dyn01`, `Dyn02`,`Dyn03`,`Dyn05`> in `run_dynamic.py` and run it.
You can refer to the `DynamicLight/models` for more details.

### 3.2.6 DynamicLight-Cycle

- Direct train: run `run_dynamic_cycle.py`
- Under transfer: first run `run_dynamic.py` to well train `DynamicLight`; next configure and run `run_test.py` to transfer it.

# 4. Baseline methods

For the baseline methods, refer to [Efficient-XLight](https://github.com/LiangZhang1996/Efficient_XLight.git) and [Advanced-XLight](https://github.com/LiangZhang1996/Advanced_XLight.git)







