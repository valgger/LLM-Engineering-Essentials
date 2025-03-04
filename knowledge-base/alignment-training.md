# Alignment and Advanced Training Techniques

## What makes an LLM? Part 3: the problem of alignment

$\symbol{"221E}$

After SFT, an LLM is able to respond to complex instructions, but we still haven't reached “perfection” just yet. First of all, we need to make the LLM harmless. What are we getting at here? Well, to give a few examples, at this point, a model could:

* Create explicit content – maybe even in response to an innocent prompt!
* Explain how to make a violent weapon …or how to destroy humankind.
* Create [a research paper on the benefits of eating crushed glass](https://thenextweb.com/news/meta-takes-new-ai-system-offline-because-twitter-users-mean).
* Sarcastically scold users for their ignorance of elementary mathematics before solving their arithmetic problem, yikes!

Certainly, this is not the behavior that we expect from a good LLM! Beyond being helpful, an AI assistant should also be harmless, non-toxic, and so on. It shouldn’t generate bad things out of nowhere, and further, it should refuse to generate them even if prompted by a user.

That said, pre-training datasets are huge, and they are seldom curated well enough to exclude harmful content, so a pre-trained LLM can indeed be capable of some awful things. Further, SFT doesn't solve this problem. Therefore, we need an additional mechanism to ensure the **alignment** of an LLM with human values.

At the same time, in order to create a good chat model, we also need to ensure that the LLM behaves in a conversation more or less like a friendly human assistant, keeping the right mood and tone of voice throughout. Here, we again need to introduce human preferences into the LLM somehow. **Chat** models you may come across usually differ from pre-trained or purely **Instruct** models because of this additional conversational alignment training.

In this section, we'll discuss how this is done. For now, we'll omit the technical and mathematical details in favor of high-level understanding. (We'll have a more in-depth treatment of alignment training in a separate long read.)

## Human preference dataset 

To align an LLM with human preference, we must show what humans deem right or wrong, and for this, we’ll need a dataset. Usually this consists of a tuple of three elements, that is, the following triples

`(prompt, preferred continuation, rejected continuation)`

This can be collected by prompting the LLM, obtaining two (or more) continuations and then asking people which one they would prefer (or by ranking them). For instance, the criteria could be: helpfulness, engagement, toxicity, although there are many, many more possibilities for this.

As an example, OpenAI collected a dataset of 100K–1M comparisons when they created ChatGPT, and they asked their labellers to rank many potential answers:

<center>
<img src="https://drive.google.com/uc?export=view&id=12oohICfFLbLV3BtCCFuo7ieqqHd0KTwb" width=800 />
</center>

Of course, this was expensive; using another powerful LLM to rank completions is a cheaper alternative.

From there, we need to use this preference dataset to fine-tune our LLM and one way to start this is by training a reward model.

## Trainable reward model 

A trainable **reward model** formalizes human preferences in an LLM-training-friendly way. Usually this involves a function $r(x, y)$ taking a prompt $x$ as input with its continuation $y$ and outputting a number. More accurately, a reward model is a neural network which can be trained on triples

`(prompt, preferred continuation, rejected continuation)`

by encouraging

$$r(prompt, preferred\ continuation) > r(prompt, rejected\ continuation).$$

That is, $r(x, y)$ is a **ranking model**. 

Now, we want to train our LLM to get the maximum possible reward, and it turns out that we need **Reinforcement Learning** (**RL**) to do that.


## Why do we need RL?

Let's ask the question this way: why can't we train the LLM to maximize reward using supervised fine-tuning?

This is because supervised fine-tuning, as we've already discussed, trains an LLM to produce specific completions for specific prompts. However, this is not in the spirit of alignment training! Instead, we want to teach the LLM to produce completions with maximal possible reward. To do that, we need to:

* Make the LLM produce completions
* Judge them with the reward model
* Suggest the LLM to improve itself based upon the reward it received

And conveniently, this is exactly what RL does!

## Reinforcement learning in a nutshell

Imagine you want to train an AI bot to play [Prince of Persia](https://www.youtube.com/watch?v=FGQmtlxllWY) (the 1989 game). In this game, the player character (that is, the titular prince) can:

* Walk left or right, jump and fight guards with his sword
* Fall into pits, get impaled on spikes, or killed by guards
* Run out of time and lose
* Save the princess and win

<center>
<img src="https://drive.google.com/uc?export=view&id=1jye3OWWmdRr2dWSDQw5NUSFa_jOyJfJK" width=600 />
</center>

The simplest AI bot would be a neural network that takes the current screen (or maybe several recent screens) as an input and predicts the next action – but how to train it?

A supervised learning paradigm would probably require us to play many successful games, record all the screens, and train the model to predict the actions we chose. But there are several problems with this approach, including the following:

* The game is quite long, so it'd simply be too tiresome to collect a satisfactory number of rounds.
* It's not sufficient to show the right ways of playing; the bot should also learn the moves to avoid.

* The game provides a huge number of possible actions on many different screens. It’s reasonable to expect that successful games played by experienced gamers won't produce data with the level of necessary diversity for the bot to "understand" the entire distribution of actions.

So, these considerations have us move to consider training the bot by **trial-and-error**:

1. Initializing its behavior ("**policy**") somehow.
2. Allowing it to play according to this policy, checking various moves (including very awkward ones) and to enjoy falling to the bottom of a pit, and so on.
3. Correct the policy based on its success or failures.
4. Repeat step 2 and 3 until we're tired of waiting or the bot learns to play Prince of Persia like a pro.

Let's formalize this a bit using conventional RL terminology:

* The (observed) **state** is the information we have about the game at the present moment. In our case, this is the content of the current screen.
* The **agent** is a bot which is capable of several **actions**.
* The **environment** is the game. It defines the possible states, the possible actions, and the effects of each action on the current state – and which state will be the next.
* The **policy** is the (trainable) strategy the bot uses for playing. In our case, this is the neural network that predicts actions given the current state, or the state history.
* The **reward** is the score that we assign to the states. For example, defeating a guard, progressing to a next level, or winning the game might have positive rewards, while falling into a pit or getting wounded by a guard would mean negative rewards.

<center>
<img src="https://drive.google.com/uc?export=view&id=1PqJaNX6bvSPHIc6JCNStIo7v7YKzTKHM" width=600 />
</center>

The goal of the training is finding a policy that maximizes reward, and there are many ways of achieving this.
We'll now see that alignment training has much relevance with Prince of Persia.

## The idea behind RLHF

**RLHF** (**Reinforcement Learning with Human Feedback**) is the training mechanism that:

* Created [InstructGPT](https://openai.com/index/instruction-following/) (an Instruct model) from GPT-3
* Created [ChatGPT](https://openai.com/index/chatgpt/) (a Chat model) from GPT-3.5, the more advanced version of GPT-3
* A training mechanism which has since been used to train many more top LLMs.

As suggested by its name, RLHF is a **Reinforcement Learning** approach, and as such, it involves the following:

* An **agent**: that is, our LLM,
* An observed **state**: the prompt and the part of the completion that has already been generated
* **Actions**: generation of the next token
* **Reward**: reward model score

<center>
<img src="https://drive.google.com/uc?export=view&id=10N0GI3exb4WdJ16S2lt-regJ4eamqgBF" width=600 />
</center>

Roughly speaking, we want do the following:

1. The **agent** (our LLM) generates the next token $y_{t+1}$ based on the **current observed state**: prompt $x$, the current completion $y_1\ldots y_t$.
2. The current completion is updated to $y_{1:(t+1)} = y_1\ldots y_ty_{t+1}$. It will now be part of the **next state**.
3. The **reward model** returns the score $r(x, y_{1:(t+1)})$.
4. The weights of the LLM are updated to maximize $r$ (in RL terms: we are updating the agent's **policy**).
5. Return to step 1 and continue until we generate the `<EOS>` token.

Here, step 4 is the most involved. We would very much like to just update the LLM weights through

$$r(x, y_1\ldots y_ty_{t+1}) = r(x, \mathrm{\color{magenta}{LLM}}(y_1\ldots y_t))\longrightarrow\max,$$

but this wouldn't work so simply. Traditionally, PPO (Proximal Policy Optimization) with some additional modifications, is used instead. We'll omit the details here and revisit RLHF technicalities as well as the optimization process in the Math of RLHF and DPO long read. Still, let’s note several important things here:

**Note 1**. RLHF is not the name of a particular algorithm. Rather, it’s a particular task formulation where the reward used in training (reward model) is an approximation of the “true” reward, which lives in the human minds (that is, real human preferences).

**Note 2**. RLHF fine tuning doesn't require pairs `(prompt, completion)` for training. The dataset for RLHF consists of prompts only, and the completions are generated by the model itself as part of the trial-and-error.

**Note 3**. OpenAI used 10K–100K prompts for the RLHF training stage of ChatGPT, which is comparable to the SFT dataset size. Moreover, the prompts were high-quality and written by experts and they were different from both the SFT and reward model training prompts.

In practice, RLHF may produce some effect even after training on about 1K prompts. Moreover, it is often trained on the same prompts that were used for reward modeling (because of the lack of data). But the quality of these prompts matters, and the less you have of them, the more you should be concerned about the quality.

**Note 4.** While RLHF improves alignment with human preferences, it doesn't directly optimize output correctness and plausibility. This means that alignment training can harm the LLM quality. So while a model that refuses to answer any question will never tell a user how to make a chemical weapon so it's perfectly harmless – although still utterly useless for helpful purposes, too.

To address this issue, we try to ensure that the RLHF-trained model doesn't diverge much from its SFT predecessor. This is often enforced by adding a regularization term to the optimized function:

$$ℒ(X) = \sum_{i=1}^Nr(x_i, LLM(x_i)) - dist(trained\\_LLM, frozen\\_SFT\\_LLM),$$

where $dist$ is some kind of distance; Kullback-Leibler divergence between the predicted probability distributions is popular in this role. This way, we maximize the reward while keeping the distance low.

<details>
    <summary> **Beware: math! Click at your own risk!** </summary>

In math terms, the loss is: 

$$ℒ_{RLHF} = 𝔼_{x\sim 𝒟, y\sim\pi_{\theta}(y|x)}\left[r(x, y)\right] - \beta 𝔻_{{KL}}\left[\pi_{\theta}(y|x)||\pi_{{SFT}}(y|x)\right],$$

where

- $\pi_{\theta}(y|x)$ is the probability distribution of completion $y$ given the prompt $x$, predicted by the trainable LLM
- $\pi_{SFT}(y|x)$ is the same, but for the frozen after-SFT LLM

You can read $𝔼_{x\sim 𝒟, y\sim\pi_{\theta}(y|x)}$ as:
- we iterate over prompts $x$ from the preference fine-tuning dataset
- for each of them we generate a completion $y$ using the trainable LLM
- we calculate $r(x, y)$ for all the pairs we've got, and we average these values.


</details>

## Direct Preference Optimization

Reinforcement Learning is a capricious tool, so there have been several attempts at getting rid of it for alignment training. The most popular one right now is [**DPO**](https://arxiv.org/pdf/2305.18290) (**Direct Preference Optimization**). Let’s try to briefly summarize the differences:

1. **RLHF**:
   * Trains an external reward model to approximate human preference
   * It then fine-tunes the LLM to maximize this synthetic reward using a trial-and-error, on-policy regime

2. **DPO** (**Direct preference optimization**):
   * Uses some math to suggest an internal reward model that doesn't take into account human preferences. It turns out to be very simple and it roughly says: 
     "*A continuation* $y$ *is better for a prompt* $x$ *if* $y$ *is more likely to be generated from* $x$ *by the LLM*".
   * It then takes the `(prompt, preferred_continuation, rejected_continuation)` dataset and trains the LLM on it in a simple supervised way to ensure that, roughly speaking, the preferred continuation is more likely to be generated from the prompt than the rejected one*.

Again, more about this in the Math of RLHF and DPO long read.

<center>
<img src="https://drive.google.com/uc?export=view&id=1QWNCO_CUhaTvh4OwhGScCl107LZ7cbJl" width=600 />
</center>

<details>
    <summary> **Beware: math! Click at your own risk!** </summary>
The actual loss function for DPO is

$$ℒ_{\mathrm{DPO}} = 𝔼_{(x, y_a, y_r)\sim 𝒟}\,\sigma\left(\beta\log\frac{\pi_{\theta}(y_a|x)}{\pi_{SFT}(y_a|x)} - \beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{{SFT}}(y_r|x)}\right),$$

where $y_a$ stands for the accepted (preferred) completion and $y_r$ for the rejected one. 
</details>

