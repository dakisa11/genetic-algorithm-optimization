# Genetic Algorithm Optimization

Neural network training using a **Genetic Algorithm**, implemented from scratch in Python

---

## Overview

This project implements a **generational genetic algorithm** used to optimize the weights of a **feed‑forward neural network** for function approximation (regression).

Instead of using backpropagation, the neural network is trained **exclusively using a genetic algorithm**, making this approach applicable even when gradients are unavailable or unreliable.

---

## 🧠 Problem Description

The goal is to approximate an unknown function based on sampled input–output pairs. The following benchmark functions are supported:

* **Sinusoidal function**
* **Rastrigin function**
* **Rosenbrock function**

Only a **subset of sampled points** is provided during training. The trained neural network is then evaluated on an unseen test set.

---

## Genetic Algorithm

Each individual in the population represents **one neural network**, encoded as a real‑valued chromosome containing **all network weights**.

### Genetic Algorithm Components

* **Population** – a set of neural networks
* **Fitness function** – inverse of Mean Squared Error (MSE)
* **Selection** – fitness‑proportional (roulette‑wheel)
* **Crossover** – arithmetic mean of parent chromosomes
* **Mutation** – additive Gaussian noise
* **Elitism** – best individuals copied directly to next generation

### Hyperparameters

All GA parameters are configurable via command line:

* `popsize` – population size
* `elitism` – number of elite individuals
* `p` – mutation probability per weight
* `K` – mutation noise standard deviation
* `iter` – number of generations

---

## Neural Network

A fully connected feed‑forward neural network with sigmoid activation is used.

### Supported Architectures

Some of the architectures are:

* `5s`   → input → 5 → σ → output
* `20s`  → input → 20 → σ → output
* `5s5s` → input → 5 → σ → 5 → σ → output

Where:

* `σ` is the logistic sigmoid
* output layer is linear

Weights are initialized from a normal distribution with standard deviation `0.01`.

---

## Dataset Format

Datasets are provided in **CSV format**.

* First row is a header
* Last column is the **target value**
* All other columns are input features

### Example: Sinusoid

```
x,y
3.469,-0.795
1.626,0.971
2.995,-0.349
```

### Example: Rastrigin

```
x1,x2,y
0.939393939394,0.373737373737,18.7532101448
0.535353535354,-0.191919191919,26.508872037
-0.0909090909091,0.555555555556,21.3012973177
```

### Example: Rosenbrock

```
x1,x2,y
1.63636363636,-2.0,2188.47954375
-0.262626262626,-1.87878787879,380.971295363
-1.07070707071,1.07070707071,4.86097610378
```

---

## Running the Project

Example command:

```bash
python solution.py \
  --train sine_train.txt \
  --test sine_test.txt \
  --nn 5s \
  --popsize 10 \
  --elitism 1 \
  --p 0.1 \
  --K 0.1 \
  --iter 10000
```

---

## Output Format

During training, the algorithm reports training error every 2000 generations:

```
[Train error @2000]: 0.002106
[Train error @4000]: 0.001565
[Train error @6000]: 0.001097
[Train error @8000]: 0.000891
[Train error @10000]: 0.000830
[Test error]: 0.000433
```

Final output reports **test error** of the best evolved network.

---

## Goals of the Project

* Understand genetic algorithms beyond toy examples
* Apply evolutionary optimization to neural networks
* Explore alternatives to gradient‑based learning
* Gain experience with stochastic optimization

---

## Technologies

* Python 3
* NumPy (optional, minimal use)
* No ML frameworks (no TensorFlow, PyTorch, sklearn)

---

## Possible Improvements

* Fitness visualization over generations
* Parallel fitness evaluation
* Multi‑objective optimization

---

## References

* Goldberg – *Genetic Algorithms in Search, Optimization and Machine Learning*
* Rosenbrock & Rastrigin benchmark functions
* University Artificial Intelligence coursework
