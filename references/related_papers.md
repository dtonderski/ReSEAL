# Related papers
#### Object Goal Navigation using Goal-Oriented Semantic Exploration (AKA SemExp) [[arxiv]](https://arxiv.org/abs/2007.00643) [[GitHub]](https://github.com/devendrachaplot/Object-Goal-Navigation)
This paper propose a modular system for the object goal navigation problem (ObjectNav), that splits the object goal navigation problem into a semantic mapping module and a goal-oriented semantic policy module

#### SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency [[arxiv]](http://arxiv.org/abs/2212.00922)
This paper presents a framework for an agent to explore in environment in order to improve its perception in that environment. Thus the agent "actively" learns its environment, which improves ObjectNav

#### Navigating to Objects in the Real World [[arxiv]](http://arxiv.org/abs/2212.00922) 
This paper compares methods for solving the ObjectNav, evaluated on robots in the real world. They show that SemExp performs well thanks to the decoupling of semantic mapping and planning

#### ObjectNav Revisited: On Evaluation of Embodied Agents Navigating to Objects [[arxiv]](http://arxiv.org/abs/2006.13171)
This paper is a summary of ObjectNav, published in 2020 (before SemExp and SEAL)

#### ROS-X-Habitat: Bridging the ROS Ecosystem with Embodied AI [[arxiv]](http://arxiv.org/abs/2109.07703) [[GitHub]](https://github.com/ericchen321/ros_x_habitat)
ObjectNav is often simulated in an environment known as AI Habitat [[link]](https://aihabitat.org).
This paper introduces an interface between AI Habitat and ROS, allowing a ROS-based planner to interact with the Habitat simulator

#### Semantic Curiosity for Active Visual Learning [[arxiv]](http://arxiv.org/abs/2006.09367)
This paper proposes a reward for learning an exploration policy that plans trajectories to improve perception.
The reward improves perception by penalising inconsistent object labels.
This is comparable to the "Gainful Curiosity" reward proposed in the SEAL paper

#### Instance-Specific Image Goal Navigation: Training Embodied Agents to Find Object Instances [[arxiv]](http://arxiv.org/abs/2211.15876)
This paper tackles the image-goal problem (ImageNav).
ImageNav is the second part to the Habitat Navigation Challenge 2023.
This paper may be useful if we choose to participate in this challenge


# Other resources
#### Habitat Navigation Challenge [[Webpage]](https://aihabitat.org/challenge/2023/) [[GitHub]](https://github.com/facebookresearch/habitat-challenge/tree/navigation-challenge-2023)
🚨 Submission deadline: 15/05/2023

#### NeurIPS 2023 [[link]](https://nips.cc)
🚨 Abstract submission deadline: 11/05/2023

🚨 Paper submission deadline: 17/05/2023
