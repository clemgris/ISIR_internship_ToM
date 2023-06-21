# Environment
The environment in this setup is a 1D tabular representation of a toy consisting of N buttons, with M buttons that play music and the rest doing nothing. This setup is based on the cognitive science paper by Hyowon Gweon (2021) [Inferential social learning: cognitive foundations of human social learning and teaching](https://www.sciencedirect.com/science/article/pii/S1364661321001789)). The purpose of using this setup was to highlight the Theory of Mind mechanism in the interaction of young children.

In the experiment, the first phase involves one child (referred to as the teacher) observing another child (referred to as the learner) playing with the toy in a specific configuration (which determines the location of the M musical buttons). In the second phase, the toy is reset to a different configuration, and the teacher must demonstrate the minimum sequence of buttons to ensure that the learner only plays the musical buttons after observing the demonstration.
# Learner
We define different type of learner by the initial distributions over the possible configurations of the environement (beliefs) modelling prior on the number of musical buttons M of the toy.
# Teacher
## Bayesian ToM
In

    bayesian_ToM.py
    
We implemented the Theory of Mind (ToM) mechanism as a Bayesian process. The teacher maintains a belief (probability distribution) over a finite and known set of possible learner types and updates these beliefs based on observations of the learner in the first phase.
In the second phase, the teacher selects the demonstration to show based on its beliefs, using either maximum a posteriori or weighted maximum utility criteria.
## ToMNet
In 

    neural_network_ToM.py
    
We modelled ToM mechanism using an adapted version of the policy prediction model ToMNet proposed by Neil C. Rabinowitz et al. (2018) in the paper [Machine theory of mind](https://arxiv.org/abs/1802.07740). The model takes as inputs previous trajectories of the learner in different environenments (toy configurations) and a possible demonstration for the second environment and outputs the predicted policy of the learner on the second environment after seen the demonstrattion.
In the second phase, the teacher choses the demonstration that maximizes their utility (reward of the learner minus the cost of showing the demonstration) based on the predicted reward derived from prediction model. This replication of the planning mechanism behind Theory of Mind aligns with the concept discussed by Mark K.Ho in the papper (2022) [People construct simplified mental representations to plan](https://arxiv.org/abs/2105.06948).
### Data generation
     python save_dataset.py --n_buttons 20 --n_music 3 --num_past 1 --max_steps 50 --max_steps_current 0 --n_agent_train 1000 --n_agent_test 100 --n_agent_val 100 --saving_name dataset
### Training
     python train.py --n_epochs 50 --basic_layer ResConv --e_char_dim 8 --batch_size 1 --data_path ./data/dataset
# Experiments
All the experiments are conducted in the notebook

    experiments.ipynb
    
We tried the robustness of both methods using useen learner type (out of the Bayesian belief's support and out of the training set) as we tested the adaptability of the teacher opposed to a changeable learner. This notebook also includes the evaluation of the different teachers.
