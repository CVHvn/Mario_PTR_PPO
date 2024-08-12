# Mario_PTR_PPO
Playing Super Mario Bros using Proximal Policy Optimization with Prioritized Trajectory Replay (PTR-PPO)

## Introduction

My PyTorch Proximal Policy Optimization with Prioritized Trajectory Replay (PTR-PPO) implement to playing Super Mario Bros. My code is modify from [PTR-PPO paper](https://arxiv.org/pdf/2112.03798). This paper combine [PPO paper](https://arxiv.org/abs/1707.06347) with [Prioritized Experience Replay paper](https://arxiv.org/pdf/1511.05952). But I couldn't find any reference code, and I don't understand some points in the paper (it’s also possible that those points in the paper aren't as good as my modifications), so I made some adjustments compared to the original paper.

Đây là code PyTorch Proximal Policy Optimization with Prioritized Trajectory Replay (PTR-PPO) để chơi Super Mario Bros của tôi. Code của tôi được điều chỉnh từ [PTR-PPO paper](https://arxiv.org/pdf/2112.03798). Paper này kết hợp [PPO paper](https://arxiv.org/abs/1707.06347) và [Prioritized Experience Replay paper](https://arxiv.org/pdf/1511.05952). Nhưng tôi không thể tìm được mã nguồn của paper và tôi không hiểu một số điểm trong paper (cũng có thể các điểm tôi không hiểu không tốt bằng thay đổi của tôi) nên tôi đã thực hiện 1 số điều chỉnh so với paper gốc.
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

Tôi có niềm đam mê với reinformance learning và cả mario (tôi thấy mario là 1 môi trường đủ khó để test thử các thuật toán). Mục tiêu của tôi là học hỏi nhiều kiến thức và thuật toán reinformance learning hơn. Đồng thời việc implement các thuật toán này cũng giúp tôi cải thiện kỹ năng code. Tôi muốn train 1 agent có thể hoàn tất cùng lúc 32/32 level một cách dễ dàng bằng reinformance learning. Tôi đã triển khai một số thuật toán RL ([A2C](https://github.com/CVHvn/Mario_A2C), [PPO](https://github.com/CVHvn/Mario_PPO), [PPO-RND](https://github.com/CVHvn/Mario_PPO_RND)) để chơi mario và đã hoàn thành nó với PPO-RND. Tuy nhiên tôi cảm thấy các thuật toán này vẫn chưa đủ mạnh để train 32 stages cùng lúc, các thuật toán cần tinh chỉnh siêu tham số để hoàn thành các stages khó (dù tôi đã có đủ kinh nghiệm để chọn siêu tham số cho các stage khó!). Nên tôi sẽ tiếp tục tìm hiểu và triển khai các thuật toán RL nâng cao hơn. Theo kinh nghiệm của tôi, các thuật toán off-policy cho kết quả rất tệ khi huấn luyện agent chơi ppo. Tuy nhiên tôi vẫn thích ý tưởng về Experience Replay, tôi nghĩ nó sẽ giúp improve các thuật toán on-policy như PPO (kết hợp on-policy và off-policy). Ban đầu tôi định thử implement ACER, nhưng tôi thấy nó khó hiểu và khó code, kết quả cũng gần như tương đồng (chỉ tốt hơn 1 chút) so với PPO. Nên tôi đã quyết định tìm hiểu và thử implement PPO-PTR, cải tiến kết hợp PPO với experience replay. 

## How to use it

You can use my notebook for training and testing agent very easy:
* **Train your model** by running all cell before session test
* **Test your trained model** by running all cell except agent.train(), just pass your model path to agent.load_model(model_path)

Or you can use **train.py** and **test.py** if you don't want to use notebook:
* **Train your model** by running **train.py**: For example training for stage 1-4: python train.py --world 1 --stage 4 --num_envs 8
* **Test your trained model** by running **test.py**: For example testing for stage 1-4: python test.py --world 1 --stage 4 --pretrained_model best_model.pth --num_envs 2

Bạn có thể sử dụng notebook của tôi để train và test agent rất dễ dàng:
* **Để train agent**, chỉ cần run tất cả cell trước session test
* **Để test agent**, chỉ cần run tất cả cell trừ cell agent.train() và thêm đường dẫn pretrained model vào agent.load_model(model_path)

Hoặc bạn có thể dùng **train.py** và **test.py** nếu bạn không muốn dùng notebook:
* **Để train agent**, run file **train.py**: ví dụ để training stage 1-4: python train.py --world 1 --stage 4 --num_envs 8
* **Test your trained model**, run file **test.py**: ví dụ để testing for stage 1-4: python test.py --world 1 --stage 4 --pretrained_model best_model.pth --num_envs 2

## Trained models

You can find trained model in folder [trained_model](trained_model)

Bạn có thể sử dụng các model đã được tôi huấn luyện tại folder [trained_model](trained_model)

## Hyperparameters

## Questions

* Can my code guarantee completion of all stages?
  - My hyperparameters don't guarantee completion of all stages, but except for stages 8-1 and 8-4, I completed all other stages with just first training run.
  - For stage 8-1, I had to try 3 sets of hyperparameters (batchsize 64 + gamma 0.9, batchsize 64 + gamma 0.99, batchsize 256 + gamma 0.9) and I succeeded with the last one (batchsize 256 + gamma 0.9). I didn't train multiple times so I'm not sure about the success rate. But based on my experience with other RL algorithms, I think the success rate is very high. If you're not too unlucky, you'll probably succeed on the first try!
  - Stage 8-4 is almost impossible to complete unless you're lucky. I've tried many hyperparameters and run multiple times with each set. I was lucky when Mario learned to do a double jump (game hack?). If you retrain, I think you won't be able to complete stage 8-4.

* How long does it take to train the agent?
  - From a few hours to a few days. It depends on the hardware. I used different hardware during training and trained multiple agents at the same time, so the speed was affected. You can refer to the table in the Hyperparameters section.

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

* Code của tôi có đảm bảo hoàn tất tất cả các stages không?
  - Siêu tham số của tôi không đảm bảo hoàn tất các stage, nhưng trừ stage 8-1 và stage 8-4 thì tôi hoàn thành tất cả stage còn lại chỉ với 1 lần training.
  - Stage 8-1 tôi phải chỉnh thử 3 siêu tham số (batchsize 64 + gamma 0.9, batchsize 64 + gamma 0.99, batchsize 256 + gamma 0.9) và tôi hoàn thành với siêu tham số cuối cùng (batchsize 256 + gamma 0.9). Tôi không train nhiều lần nên không chắc về tỷ lệ huấn luyện thành công. Nhưng với kinh nghiệm của tôi với các thuật toán RL khác, tôi nghĩ tỷ lệ thành công rất cao, nếu bạn không quá xui thì chắc chắn sẽ thành công ngay lần đầu!.
  - Stage 8-4 gần như không thể hoàn thành trừ khi bạn may mắn. Tôi đã thử nhiều siêu tham số và run nhiều lần với mỗi bộ siêu tham số. Tôi đã may mắn khi mario học được cách thực hiện double jump (hack game?). Nếu bạn train lại, tôi nghĩ bạn sẽ không thể hoàn thành stage 8-4.

* Thời gian để train agent?
  - Từ vài giờ đến vài ngày. Tùy theo phần cứng, tôi dùng nhiều phần cứng khác nhau trong quá trình training và train nhiều agent cùng lúc nên tốc độ bị ảnh hưởng. Bạn có thể tham khảo trong bảng ở phần **Hyperparameters**

* Bạn có thể làm gì để cải thiện code của tôi?
  - Bạn có thể tách phần testing ra thành 1 thread hoặc process riêng. Tôi không giỏi code đa lường nên tôi không làm vậy!
  - Bạn có thể thử các bộ siêu tham số khác!

* So sánh với PPO?
  - Tôi thấy thuật toán ổn định hơn PPO, hoàn tất 30/32 stages trong lần 1 lần run duy nhất.
  - Có thể hoàn tất stage 8-4 (dù may mắn). Tôi đã thử PPO với nhiều bộ siêu tham số nhưng không thể hoàn thành!
  - Thuật toán này chậm hơn và tốn nhiều ram hơn PPO vì dùng replay và phải lấy dữ liệu từ replay cũng như tính toán lại advantages cho dữ liệu từ replay.
  - Thuật toán cũng giới thiệu rất nhiều siêu tham số mới, nhiều siêu tham số quan trọng như alpha của PER thậm chí không được nói đến trong paper (paper thiếu chuyên nghiệp!). Dẫn đến bạn phải tuning siêu tham số nhiều hơn! 

* Tôi đã điều chỉnh gì so với paper?
  - Paper train critic network với 1 step td error. Tôi sử dụng value loss tương tự như code PPO của tôi.
  - Vì tác giả sử dụng 1 step td error, họ sẽ chỉ đánh weight sample đó bằng action_prob / old_action_prob. Tôi dùng ρ marg để weight các sample tương tự như actor loss.
  - Tác giả không nói rõ về siêu tham số alpha của PER nên tôi dùng tham số khuyến nghị là 0.7. Tác giả cũng không nói rõ về cách random sample từ PER nên tôi giữ nguyên như PER nguyên bản.
  - Trong paper, tác giả sử dụng ký hiệu b để chỉ behavior policy (save vào replay) và old_pi để chỉ policy trước khi train, pi là policy đang train (vì agent được train qua nhiều batch và epoch nên old_pi sẽ khác pi). Trong paper không nói rõ về ratio của PPO r = pi / old_pi hay r = pi / b nhưng theo lý thuyết PPO và thực nghiệm của tôi thì thuật toán chỉ work khi r = pi / b.
  - Với thuật toán trong paper, tác giả backup old_pi trước mỗi lần trainning với data được lấy từ replay. Nhưng tôi thấy việc này không ổn định, tôi chỉ backup old_pi trước mỗi lần train và chỉ update old_pi khi tới lần train tiếp theo (old_pi sẽ được backup và giữ nguyên trước khi train 2 epochs on-policy và 8 epochs off-policy)

## Requirements

* **python 3>3.6**
* **gym==0.25.2**
* **gym-super-mario-bros==7.4.0**
* **imageio**
* **imageio-ffmpeg**
* **cv2**
* **pytorch** 
* **numpy**

## Acknowledgements
With my code, I can completed 32/32 stages of Super Mario Bros.

Tôi đã hoàn thành 32/32 stages của Super Mario Bros với thuật toán này.

## Reference
* [Howuhh PER](https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py)
* [CVHvn A2C](https://github.com/CVHvn/Mario_A2C)
* [Stable-baseline3 ppo](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO)
* [uvipen PPO](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)
* [lazyprogrammer A2C](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl3/a2c)