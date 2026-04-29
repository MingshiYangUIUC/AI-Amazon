A simple Amazon chess player [Easyamaze](https://en.botzone.org.cn/game/ranklist/59463fb292a2ea07a5c6b5a8?page=0#66fcbae6bae30059c85f35ba) trained by reinforcement learning.

This is only limited experimental training with a super small neural network, but the logic is very similar to https://github.com/MingshiYangUIUC/AI-Doudizhu/.   

NOW: Training data is generated through policy-guided self-play, where a lightweight policy model is used to prioritize and prune the large action space, enabling faster and more diverse data generation while retaining a value network for final evaluation.

Email mingshi3@illinois.edu if you have questions...

---

### To train a model from scratch

- Recommended to use linux or WSL, requires `torch`

- Install pybind11 and run `cpp/_compile.py` to compile cpp code (action generation)    
    `action_module.xxx` will show up after successful run

- Run `train_test.py`   
    Configs are set following `if __name__ == "__main__"`    
    Models will be saved under `models` after configurated number of episodes
