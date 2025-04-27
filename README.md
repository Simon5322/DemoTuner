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
