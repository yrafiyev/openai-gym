# [Cart-Pole-v1](https://gym.openai.com/envs/CartPole-v1/)

---

## Description
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled
by applying a force of +1 or -1 to the cart. The pendulum starts upright,
and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright.

---

## Setup
* Deep Neural Network layers and sizes:
    * Input Layer: 4
    * First hidden layer: 12 (configurable)
    * Second hidden layer: 12 (configurable)
    * Third hidden layer: 12 (configurable)
    * Output Layer: 2


---

## Learning Algorithm
Agent was trained using Q-learning algorithms described in [this article](http://www.davidqiu.com:8888/research/nature14236.pdf).

## Results

![Results](https://github.com/yrafiyev/openai-gym/tree/master/cart-pole/batch_32_layer_12.jpg)