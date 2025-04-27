# DemoTuner: Efficient DBMS Knobs Tuning via LLM-Assisted Demonstration Reinforcement Learning
Database systems offer hundreds of parameters that critically impact performance. However, automatic tuning remains challenging due to the vast configuration space and complex parameter interactions. Traditional reinforcement learning (RL) methods suffer from low sample efficiency and slow training, partly due to the lack of guidance during exploration.

DEMOTune is the first approach to bring demonstration learning into database configuration tuning.
By leveraging domain knowledge and high-quality tuning hints, DEMOTune pretrains the agent with high-quality configurations, and later balances domain-guided exploration with self-exploration. This reduces training overhead while improving tuning effectiveness.

Our key contributions:

Knowledge Extraction: Use large language models (LLMs) to mine detailed domain knowledge from expert texts, including parameter recommendations.

Demonstration Learning: Incorporate domain knowledge into the RL agent’s exploration through demonstrations.

Training Enhancements: Apply action-guidance and reward-shaping techniques to speed up and improve training.

We evaluated DEMOTune on MySQL and PostgreSQL under three different workloads.
Results show that DEMOTune significantly outperforms state-of-the-art methods in effectiveness, efficiency, and adaptability.

![DEMOTune Overview](./overview1.png)
# start
## Envirnoment Version
PostgreSQL (16.1)
MySQL (8.0.36)

## Install dependencies
pip install -r requirements.txt

## Benchmark
The main benchmark we employed for configuration evalu-
ation is YCSB (Yahoo! Cloud Serving Benchmark), which is a
frequently-used benchmarking tool for performance evaluation
of DBMSs.After 安装，需要在pg.ini或mysql.ini中修改YCSB的安装路径YCSB_path

## Main Code
tuning_run.py 加上配置设定文件的路径并--collect 用于收集生成演示数据
mysql.ini 和 pg.ini中设定YCSB负载的类型和大小
tuning_run.py 加上配置设定文件的路径 并在s1中设定训练相关选项进行离线训练

## Acknowledge
We would like to express our special thanks to the open-source benchmarks, traces and codes of these papers or repositories:
https://github.com/brianfrankcooper/YCSB
https://github.com/cmu-db/benchbase
