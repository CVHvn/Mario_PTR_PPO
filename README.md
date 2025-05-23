# Mario_PTR_PPO
Playing Super Mario Bros using Proximal Policy Optimization with Prioritized Trajectory Replay (PTR-PPO)

## Introduction

My PyTorch Proximal Policy Optimization with Prioritized Trajectory Replay (PTR-PPO) implement to playing Super Mario Bros. My code is modify from [PTR-PPO paper](https://arxiv.org/pdf/2112.03798). This paper combine [PPO paper](https://arxiv.org/abs/1707.06347) with [Prioritized Experience Replay paper](https://arxiv.org/pdf/1511.05952). But I couldn't find any reference code, and I don't understand some points in the paper (it’s also possible that those points in the paper aren't as good as my modifications), so I made some adjustments compared to the original paper.
<p align="center">
  <img src="demo/gif/1-1.gif" width="200">
  <img src="demo/gif/1-2.gif" width="200">
  <img src="demo/gif/1-3.gif" width="200">
  <img src="demo/gif/1-4.gif" width="200"><br/>
  <img src="demo/gif/2-1.gif" width="200">
  <img src="demo/gif/2-2.gif" width="200">
  <img src="demo/gif/2-3.gif" width="200">
  <img src="demo/gif/2-4.gif" width="200"><br/>
  <img src="demo/gif/3-1.gif" width="200">
  <img src="demo/gif/3-2.gif" width="200">
  <img src="demo/gif/3-3.gif" width="200">
  <img src="demo/gif/3-4.gif" width="200"><br/>
  <img src="demo/gif/4-1.gif" width="200">
  <img src="demo/gif/4-2.gif" width="200">
  <img src="demo/gif/4-3.gif" width="200">
  <img src="demo/gif/4-4.gif" width="200"><br/>
  <img src="demo/gif/5-1.gif" width="200">
  <img src="demo/gif/5-2.gif" width="200">
  <img src="demo/gif/5-3.gif" width="200">
  <img src="demo/gif/5-4.gif" width="200"><br/>
  <img src="demo/gif/6-1.gif" width="200">
  <img src="demo/gif/6-2.gif" width="200">
  <img src="demo/gif/6-3.gif" width="200">
  <img src="demo/gif/6-4.gif" width="200"><br/>
  <img src="demo/gif/7-1.gif" width="200">
  <img src="demo/gif/7-2.gif" width="200">
  <img src="demo/gif/7-3.gif" width="200">
  <img src="demo/gif/7-4.gif" width="200"><br/>
  <img src="demo/gif/8-1.gif" width="200">
  <img src="demo/gif/8-2.gif" width="200">
  <img src="demo/gif/8-3.gif" width="200">
  <img src="demo/gif/8-4.gif" width="200"><br/>
  <i>Results</i>
</p>

## Motivation

I'm passionate about reinforcement learning and Mario (I find Mario to be a sufficiently challenging environment to test algorithms). My goal is to learn more reinforcement learning algorithms and techniques. Implementing these algorithms also helps me improve my coding skills. I aim to train an agent that can easily complete all 32 levels of Mario using reinforcement learning. I've implemented several RL algorithms ([A2C](https://github.com/CVHvn/Mario_A2C), [PPO](https://github.com/CVHvn/Mario_PPO), [PPO-RND](https://github.com/CVHvn/Mario_PPO_RND)) to play Mario and have successfully completed it with PPO-RND. However, I feel that these algorithms are not yet powerful enough to train on all 32 stages simultaneously. They require tuning hyperparameters to complete difficult stages (even though I have enough experience to choose hyperparameters for challenging stages!). Therefore, I will continue to learn and implement more advanced RL algorithms. In my experience, off-policy algorithms perform poorly when training agents to play Mario. However, I still like the idea of Experience Replay. I think it could help improve on-policy algorithms like PPO (combining on-policy and off-policy). Initially, I planned to implement ACER, but I found it difficult to understand and implement, and the results were almost similar (only slightly better) compared to PPO. So, I decided to explore and implement PPO-PTR, combining PPO with experience replay.

## How to use it

You can use my notebook for training and testing agent very easy:
* **Train your model** by running all cell before session test
* **Test your trained model** by running all cell except agent.train(), just pass your model path to agent.load_model(model_path)

Or you can use **train.py** and **test.py** if you don't want to use notebook:
* **Train your model** by running **train.py**: For example training for stage 1-4: python train.py --world 1 --stage 4 --num_envs 8
* **Test your trained model** by running **test.py**: For example testing for stage 1-4: python test.py --world 1 --stage 4 --pretrained_model best_model.pth --num_envs 2

## Trained models

You can find trained model in folder [trained_model](trained_model)

## Hyperparameters

I selected hyperparameters based on the default parameters of PPO, PER, and the PPO-PTR paper:
- With PER: alpha = 0.7, eps = 0.01 same as [Howuhh PER](https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py). This algorithm don't use beta because we weight sample by importance sampling instead of PER weight computed by priority and beta.
- Memory size = 256. I try 256, 512 and 1024. But 256 work better.
- eps_marg = 0.2 same as PPO-PTR paper.
- ratio between on-policy and off-policy training is 2:8 same as PPO-PTR paper.
- I kept the gamma values for the stages according to the hyperparameters from my PPO project: 0.9 or 0.99.
- I only tuned the batch size for stages 8-1 and 8-4: using 256 instead of 64.
- I still use entropy target = 0.05, I believe this is an important parameter to stabilize learning. Note: I calculate entropy target with old_policy backed up before training instead of behavior policy. 
- The default parameter set (except for gamma) worked for 30/32 stages. Due to resource limitations, I couldn’t test with only one gamma value, so I had to refer to previous projects (stages that couldn’t be completed with 0.9 were tested with 0.99).

You can refer to the hyperparameters in the table below.

| World | Stage | num_envs | learn_step | batchsize | on-epoch | off-epoch | lambda | gamma | learning_rate | target_kl | clip_param | max_grad_norm | norm_adv | V_coef | entropy_coef | loss_type | per_eps | per_alpha | per_beta | eps_marg | memory_size | training_step | training_time |
|-------|-------|----------|------------|-----------|----------|-----------|--------|-------|---------------|-----------|------------|---------------|----------|--------|--------------|-----------|---------|------------|----------|-----------|--------------|---------------|---------------|
| 1     | 1     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 104954        | 4:07:44       |
| 1     | 2     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 337885        | 15:52:13      |
| 1     | 3     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 533502        | 22:06:32      |
| 1     | 4     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 46076         | 1:57:32       |
| 2     | 1     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 415724        | 15:06:07      |
| 2     | 2     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 497663        | 17:39:54      |
| 2     | 3     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 120297        | 4:34:15       |
| 2     | 4     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 86015         | 3:31:16       |
| 3     | 1     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 126464        | 4:33:51       |
| 3     | 2     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 67053         | 3:02:46       |
| 3     | 3     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 138742        | 4:04:53       |
| 3     | 4     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 106495        | 3:11:51       |
| 4     | 1     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 134124        | 3:55:59       |
| 4     | 2     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 436208        | 15:47:05      |
| 4     | 3     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 106484        | 4:21:14       |
| 4     | 4     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 219638        | 7:33:17       |
| 5     | 1     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 156160        | 5:47:13       |
| 5     | 2     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 326645        | 9:55:26       |
| 5     | 3     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 214010        | 8:17:16       |
| 5     | 4     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 205299        | 6:57:02       |
| 6     | 1     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 72187         | 3:01:48       |
| 6     | 2     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 367614        | 14:29:09      |
| 6     | 3     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 158714        | 6:11:25       |
| 6     | 4     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 137216        | 5:22:12       |
| 7     | 1     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 179706        | 7:05:49       |
| 7     | 2     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 595449        | 16:05:53      |
| 7     | 3     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 217597        | 7:54:26       |
| 7     | 4     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 120787        | 4:52:59       |
| 8     | 1     | 16       | 512        | 256       | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 2455542       | 3 days, 19:55:54  |
| 8     | 2     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 759799        | 21:39:34      |
| 8     | 3     | 16       | 512        | 64        | 2        | 8         | 0.95   | 0.9   | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 263168        | 9:13:53       |
| 8     | 4     | 16       | 512        | 256       | 2        | 8         | 0.95   | 0.99  | 7e-5      | 0.05      | 0.2        | 0.5           | FALSE    | 0.5    | 0.01         | huber     | 0.01    | 0.7        | 0.4      | 0.2       | 256          | 1532832       | 2 days, 12:02:26  |

## Questions

* Can my code guarantee completion of all stages?
  - My hyperparameters don't guarantee completion of all stages, but except for stages 8-1 and 8-4, I completed all other stages with just first training run. Update: hard stage include 1-3, 5-3, 8-1, 8-4 maybe take longer time to completed when I rerun with different hyperparam (I am very lucky with first run).
  - For stage 8-1, I had to try 3 sets of hyperparameters (batchsize 64 + gamma 0.9, batchsize 64 + gamma 0.99, batchsize 256 + gamma 0.9) and I succeeded with the last one (batchsize 256 + gamma 0.9). I didn't train multiple times so I'm not sure about the success rate. But based on my experience with other RL algorithms, I think the success rate is very high. If you're not too unlucky, you'll probably succeed on the first try!
  - Stage 8-4 is almost impossible to complete unless you're lucky. I've tried many hyperparameters and run multiple times with each set. I was lucky when Mario learned to do a double jump (game hack?). If you retrain, I think you won't be able to complete stage 8-4.

* How long does it take to train the agent?
  - From a few hours to a few days. It depends on the hardware. I used different hardware during training and trained multiple agents at the same time, so the speed was affected. You can refer to the table in the **Hyperparameters** section.

* What can you do to improve my code?
  - You can separate the testing part into a separate thread or process. I'm not good at multithreading so I didn't do that!
  - You can try different sets of hyperparameters!

* Comparison with PPO?
  - I find the algorithm more stable than PPO, completing 30/32 stages in a single run.
  - It can complete stage 8-4 (although it was lucky). I've tried PPO with many hyperparameters but couldn't complete it!
  - This algorithm is slower and consumes more RAM than PPO because it uses replay and needs to retrieve data from the replay as well as recalculate advantages for data from the replay.
  - The algorithm also introduces many new hyperparameters, and many important hyperparameters like alpha of PER are not even mentioned in the paper (the paper is unprofessional!). This leads to more hyperparameter tuning!

* What did I adjust compared to the paper?
  - The paper trains the critic network with a 1-step TD error. I use a value loss similar to my PPO code.
  - Because the authors use a 1-step TD error, they only weight that sample by action_prob / old_action_prob. I use ρ marg to weight samples similarly to the actor loss.
  - The authors don't specify the hyperparameter alpha for PER, so I use the recommended value of 0.7. The authors also don't specify how to randomly sample from PER, so I kept it the same as the original PER.
  - In the paper, the authors use the symbol b to denote the behavior policy (saved to the replay) and old_pi to denote the policy before training, and pi is the policy being trained (since the agent is trained over multiple batches and epochs, old_pi will be different from pi). The paper doesn't specify whether ratio of PPO r = pi / old_pi or r = pi / b, but according to PPO theory and my experiments, the algorithm only works when r = pi / b.
  - With the algorithm in the paper, the authors back up old_pi before each training with data taken from the replay. But I found this to be unstable, so I only back up old_pi before each training and only update old_pi when it's time for the next training (old_pi will be backed up and kept unchanged before training 2 on-policy epochs and 8 off-policy epochs).

## Acknowledgements
With my code, I can completed 32/32 stages of Super Mario Bros.

## Requirements

* **python 3>3.6**
* **gym==0.25.2**
* **gym-super-mario-bros==7.4.0**
* **imageio**
* **imageio-ffmpeg**
* **cv2**
* **pytorch** 
* **numpy**

## Reference
* [Howuhh PER](https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py)
* [CVHvn A2C](https://github.com/CVHvn/Mario_A2C)
* [Stable-baseline3 ppo](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO)
* [uvipen PPO](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)
* [lazyprogrammer A2C](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl3/a2c)