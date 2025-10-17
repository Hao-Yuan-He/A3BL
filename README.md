# A3BL Integration for ABLkit

This branch adapts A3BL to the latest ABLkit (https://github.com/AbductiveLearning/ABLkit) for compatibility and easier use within the ABL community.

Two new experiments are added: 

- We have extended our experiments to the more realistic `BDD-OIA` task, a multi-label autonomous driving benchmark for studying reasoning systems in real-world, high-stakes scenarios.

- We add an multi-task setting addition variant, which is combine two addition with mod together. This experiment is proposed to support our learnability analysis in a recently paper "A learnability analysis on neuro-symbolic learning" accepted by NeurIPS'2025.

For reproducible experiments corresponding to the ICML paper, please refer to the ICML branch: https://github.com/Hao-Yuan-He/A3BL/tree/icml.

This branch also includes performance improvements over the ICML version, including vectorized operations in the abduction process and updates aligned with the latest ABLkit API, resulting in faster execution.