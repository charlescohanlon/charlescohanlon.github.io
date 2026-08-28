---
title: Motivating My Current Research Stance
description: ""
date: 2026-06-06
tags: []
draft: true
---
# Motivating My Research Stance: prerequisites of superintelligence

The following is by no means a rigorous argument. I try hard not to overclaim. I do not have the wisdom or experience required to bet against trillion-dollar frontier laboratories. Although I echo the ideas of some who do. I merely seek to explain what my experience interacting with Large Language Model (LLM) based AI and its literature has led me to believe.

> ...if I asked you about art, you'd probably give me the skinny on every art book ever written. \[...] But I'll bet you can't tell me what it smells like in the Sistine Chapel. You've never actually stood there and looked up at that beautiful ceiling...
>
> — Sean Maguire, _Good Will Hunting_

### At a High-Level
LLMs are real-valued affine transformations fed to non-linear functions in a high-dimensional space. (This is a fancy way of saying they're a bunch of multiplications, additions, and mathematical functions applied to lists of numbers.) The key is that the coefficients that define the transformations are tuned such that applying the operations in sequence yields a probability distribution over all possible words (actually, tokens, which tend to be sub-words) that the LLM could say. Essentially, LLMs produce the most commonly said word they've seen said before, given the previous words they've said. Vision language models (VLMs) extend this paradigm to include images. They consume images as pixel patches, flattened in raster order: top-left to bottom-right. 

This approach has taken us pretty far. We learned in the mid-2010s that scaling network size led to "emergent" capabilities. Models began to draw on concepts from across domains; they appeared to generalize and extrapolate. Their frequentist modeling of word dependencies makes them effective natural-language search engines. They're remarkably capable of answering in-distribution queries, which is helpful when their training distribution reflects the majority of human knowledge. Yet, they seem to exhibit some fundamental limitations.

### Ungrounded Perception
LLMs/VLMs perceive raw data tokens as their base unit. They're not grounded in reality. If you were to show a VLM a paper airplane just as it's thrown and ask about its trajectory, best case, the model admits it can't answer. A response anywhere close to correct would be completely by chance. The models can regurgitate the aerodynamic equations governing the system or generate a plausible image of the airplane in flight, but they lack a semantic understanding of the aircraft's state.

An illustrative thought experiment I used with a friend: Imagine a dog. It could be any dog, but I have pictured a German Shepherd here.

![[German Shepherd.jpeg]]

Now tell me what the RGB values are for the top-left pixel. You can't. VLMs do this. They see in unsigned 8-bit integer triples.
### A Weak Mental Model
Opponents of LLMs/VLMs also often claim they lack a mental model of the world. In this section, I use "model" to refer to an internal, abstract representation of reality. Humans develop these through experience observing cause and effect. You might call it intuition or common sense. In daily life, we plan, reason, and evaluate against these mental models.

I believe that to say LLMs have no mental model whatsoever is too extreme. If you were to sufficiently describe the current state of reality and ask it what the most likely next state would be, it could answer correctly. Especially if there's a high level of abstraction and the state is simple. E.g., "There's a ball lying motionless on the ground. I pick it up. What happens next?"

![[Weak Model of World.png]]

It's correct answer implies the existence of at least a weak implicit mental model of the world.

That said, when asked to answer a query that'd benefit from explicit simulation, LLMs fall short. Reasoning about the trajectory of a paper airplane means forward simulation. Given initial conditions: throw angle, force, fold geometry, air currents, it requires unraveling hypothetical state-action sequences through a mental model, which LLMs/VLMs are architecturally incapable of. When they reason about physical scenarios, they're pattern matching against their training data.

### Sample Inefficient
Yann LeCun uses the driving example: adolescents can learn to drive a car in about 20 hours of practice, whereas current popular machine learning (ML) methods demand far more data, to the point that even the rarest situations are encountered frequently during training ([LeCun, 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)). Both systems pre-trained. A human adolescent has years of experience in the world as they live their lives. ML systems are shown billions of examples — trillions of tokens. Yet there remains an inherent disparity between the two forms of intelligence. 

### My Belief
The human brain operates on sensory perception, constructs a strong mental model, and exhibits powerful generalization, but it is not the pinnacle of intelligence. Deep neural networks surpass human-level performance on certain tasks. LLMs have unparalleled fact recall and learn comparatively quickly. In the search space of all forms of intelligence, neither is Pareto optimal. I believe a true super intelligence should be.

**inductive bias** · _noun._ · machine learning ([Mitchell, 1980](https://www.cs.cmu.edu/~tom/pubs/NeedForBias_1980.pdf))
The set of assumptions that \[a] learner uses to predict outputs of given inputs that it has not encountered.

Inductive biases (IBs) are design decisions that incentivize a model to learn one data-fitting function over another. They are not strict constraints, but soft ones. I seek to explore IBs that enable neural-network-based forms of intelligence to address the fundamental limitations of current methods. 

The following are some IBs I believe are critical to explore,
- Rich multimodal input signals — for a model to be extraordinarily knowledgeable of the physical world, it must be grounded in it.
- Planning/reasoning as a means for producing complex trajectories or ideas — planning is an iterative method that develops a solution incrementally. The capability of current ML methods to plan is emergent. It should be innate.
- Hierarchical modeling — models should operate across time horizons. What is its immediate next action? What is its goal an hour from now? A week from now?

### World Models and Joint Embedding Predictive Architectures
World Models (WMs) and Joint Embedding Predictive Architectures (JEPAs) are promising IBs that aim to address the limitations of LLMs. They embrace multimodal input signals, planning, and hierarchical modeling.

WMs and JEPAs perform inference by predicting in a semantic latent space. They're trained to minimize the distance between a predicted representation of the $i$th input perturbed $\hat{s}_y(i)$, and a representation of the perturbation $s_y(i)$, over all examples in a dataset,

$$\frac{1}{M} \sum_{i=1}^M D(\hat{s}_y(i), s_y(i)) = \frac{1}{M} \sum_{i=1}^M \sum_{j \in B_i} ||\hat{s}_{y_j} - s_{y_j}||_2^2$$

For instance, I-JEPA perturbs input images by removing patches of pixels from the image. It minimizes the distance in representation space between it's prediction given remaining patches, and the representation of *only* the pixel patches cut out.

The encoder serves to extract high-level meaningful semantic features from the raw data. It lets the predictor — the model performing the regression task — only concern itself with the information imbalance between what's given and what it must predict.

This forces the neural network to learn 
### Addendum
I've come across the argument that the human brain surpasses neural networks in intelligence per watt.  However, LLMs simulate intelligence on general-purpose, unspecialized machines. Even the GPU, which is purpose-built for matrix multiplication (the backbone operation of these systems), can perform other operations. Whereas the brain substrate's sole purpose is to facilitate action potentials between neurons (i.e., think). Ultra-specialization allows it to run on mere watts, while LLMs demand gigawatts. Several groups have recognized this and are working to burn models directly into hardware. Custom accelerators can, in theory, close the energy-efficiency gap.