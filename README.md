# DemoTuner: Efficient DBMS Knobs Tuning via LLM-Assisted Demonstration Reinforcement Learning
**DemoTuner** is the first approach that introduces **demonstration learning** into database configuration tuning.

Database systems expose hundreds of parameters that critically impact performance. However, automatic tuning remains challenging due to the vast configuration space and complex parameter interactions. Traditional reinforcement learning (RL) methods often suffer from low sample efficiency and slow training, partly due to the lack of guidance during exploration.

**DemoTuner** addresses these challenges by incorporating expert knowledge—mined automatically using LLMs—into the RL training process, 
improving both the efficiency and effectiveness of tuning.

We evaluated **DemoTuner** on **MySQL** and **PostgreSQL** under three different workloads.  
Experimental results show that DemoTuner significantly outperforms state-of-the-art methods in terms of **effectiveness**, **efficiency**, and **adaptability**.
## Key Contributions

- **Knowledge Extraction:** Utilize large language models (LLMs) to mine detailed domain knowledge from expert texts, including parameter recommendations.
- **Demonstration Learning:** Incorporate domain knowledge into the RL agent’s exploration through demonstrations.
- **Training Enhancements:** Apply action-guidance and reward-shaping techniques to accelerate and improve training.



![DEMOTune Overview](./overview1.png)
# start
## Environment Version  
PostgreSQL (16.1)  
MySQL (8.0.36)

## Install dependencies
```bash
pip install -r requirements.txt

## Benchmark
The main benchmark we employed for configuration evaluation is YCSB (Yahoo! Cloud Serving Benchmark), which is a frequently used benchmarking tool for performance evaluation of DBMSs. After installation, you need to modify the YCSB installation path in pg.ini or mysql.ini by setting the YCSB_path.

## Main Code
Run `tuning_run.py` with the configuration file path and the `--collect` option to collect and generate demonstration data. Set the YCSB workload in `mysql.ini` or `pg.ini`. Then, run `tuning_run.py` with the configuration file path and set the training-related options in section `s1` to perform offline training.

## Acknowledge
We would like to express our special thanks to the open-source benchmarks, traces and codes of these papers or repositories:

- [YCSB（Yahoo! Cloud Serving Benchmark）](https://github.com/brianfrankcooper/YCSB)
- [Benchbase](https://github.com/cmu-db/benchbase)



