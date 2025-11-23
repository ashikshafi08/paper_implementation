# Paper Reading & Re-Implementation Checklist

Rough list of papers and ideas I want to read and code up, mostly pulled from the Annotated Research Paper Implementations project.
I’ll just flip `[ ]` → `[x]` when something’s done and maybe drop a quick note next to it if there’s anything interesting or painful.

---

## ✨ Transformers

- [ ] **JAX Implementation** — [JAX: composable transformations of Python+NumPy programs (Frostig et al., 2018)](https://arxiv.org/abs/1812.01815) *(framework background)*
- [ ] **Multi-Headed Attention** — [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [ ] **Triton Flash Attention** — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [ ] **Transformer Building Blocks** — [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [ ] **Transformer XL** — [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., 2019)](https://arxiv.org/abs/1901.02860)
- [ ] **Relative Multi-Headed Attention** — [Self-Attention with Relative Position Representations (Shaw et al., 2018)](https://arxiv.org/abs/1803.02155)
- [ ] **Rotary Positional Embeddings (RoPE)** — [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [ ] **Attention with Linear Biases (ALiBi)** — [Train Short, Test Long: Attention with Linear Biases (Press et al., 2021)](https://arxiv.org/abs/2108.12409)
- [ ] **RETRO** — [Efficient Retrieval Augmented Generation (Borgeaud et al., 2022)](https://arxiv.org/abs/2112.04426)
- [ ] **Compressive Transformer** — [Compressive Transformers for Long-Range Sequence Modelling (Rae et al., 2019)](https://arxiv.org/abs/1911.05507)
- [ ] **GPT Architecture** — [Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [ ] **GLU Variants** — [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](https://arxiv.org/abs/1612.08083) *(plus follow-up variants)*
- [ ] **kNN-LM (Generalization through Memorization)** — [Generalization through Memorization: Nearest Neighbor Language Models (Khandelwal et al., 2020)](https://arxiv.org/abs/1911.00172)
- [ ] **Feedback Transformer** — [The Feedback Transformer (Feng et al., 2020)](https://arxiv.org/abs/2002.09402)
- [ ] **Switch Transformer** — [Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)
- [ ] **Fast Weights Transformer** — [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236)
- [ ] **FNet** — [FNet: Mixing Tokens with Fourier Transforms (Lee-Thorp et al., 2021)](https://arxiv.org/abs/2105.03824)
- [ ] **Attention Free Transformer** — [An Attention Free Transformer (Zhai et al., 2021)](https://arxiv.org/abs/2105.14103)
- [ ] **Masked Language Model** — [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [ ] **MLP-Mixer** — [MLP-Mixer: An all-MLP Architecture for Vision (Tolstikhin et al., 2021)](https://arxiv.org/abs/2105.01601)
- [ ] **Pay Attention to MLPs (gMLP)** — [Pay Attention to MLPs (Liu et al., 2021)](https://arxiv.org/abs/2105.08050)
- [ ] **Vision Transformer (ViT)** — [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [ ] **Primer EZ** — [Primer: Searching for Efficient Transformers for Language Modeling (So et al., 2021)](https://arxiv.org/abs/2109.08668)
- [ ] **Hourglass** — [Stacked Hourglass Networks for Human Pose Estimation (Newell et al., 2016)](https://arxiv.org/abs/1603.06937) *(confirm this matches the intended “Hourglass” variant)*

## ✨ Low-Rank Adaptation

- [ ] **LoRA** — [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)

## ✨ Eleuther GPT-NeoX

- [ ] **Generate on a 48GB GPU** — implementation notes (no single canonical paper; see [GPT-NeoX-20B technical report](https://arxiv.org/abs/2204.06745) for architecture)
- [ ] **Finetune on two 48GB GPUs** — implementation guide (refer to above GPT-NeoX report)
- [ ] **LLM.int8()** — [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022)](https://arxiv.org/abs/2208.07339)

## ✨ Diffusion Models

- [ ] **DDPM** — [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [ ] **DDIM** — [Denoising Diffusion Implicit Models (Song et al., 2021)](https://arxiv.org/abs/2010.02502)
- [ ] **Latent Diffusion Models** — [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752)
- [ ] **Stable Diffusion** — same as above LDM paper + Stability AI implementation notes

## ✨ Generative Adversarial Networks

- [ ] **Original GAN** — [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [ ] **DCGAN** — [Unsupervised Representation Learning with Deep Convolutional GANs (Radford et al., 2016)](https://arxiv.org/abs/1511.06434)
- [ ] **CycleGAN** — [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Zhu et al., 2017)](https://arxiv.org/abs/1703.10593)
- [ ] **WGAN** — [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875)
- [ ] **WGAN-GP** — [Improved Training of Wasserstein GANs (Gulrajani et al., 2017)](https://arxiv.org/abs/1704.00028)
- [ ] **StyleGAN 2** — [Analyzing and Improving the Image Quality of StyleGAN (Karras et al., 2020)](https://arxiv.org/abs/1912.04958)

## ✨ Sequence & Vision Architectures

- [ ] **Recurrent Highway Networks** — [Recurrent Highway Networks (Zilly et al., 2017)](https://arxiv.org/abs/1607.03474)
- [ ] **LSTM** — [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [ ] **HyperNetworks / HyperLSTM** — [HyperNetworks (Ha et al., 2017)](https://arxiv.org/abs/1609.09106) and [HyperNetworks for LSTM (Ha et al., 2016)](https://arxiv.org/abs/1609.09106)
- [ ] **ResNet** — [Deep Residual Learning for Image Recognition (He et al., 2016)](https://arxiv.org/abs/1512.03385)
- [ ] **ConvMixer** — [Patches Are All You Need? (Trockman & Kolter, 2022)](https://arxiv.org/abs/2201.09792)
- [ ] **Capsule Networks** — [Dynamic Routing Between Capsules (Sabour et al., 2017)](https://arxiv.org/abs/1710.09829)
- [ ] **U-Net** — [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [ ] **Sketch RNN** — [A Neural Representation of Sketch Drawings (Ha & Eck, 2017)](https://arxiv.org/abs/1704.03477)

## ✨ Graph Neural Networks

- [ ] **GAT** — [Graph Attention Networks (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)
- [ ] **GATv2** — [How Attentive are Graph Attention Networks? (Brody et al., 2022)](https://arxiv.org/abs/2105.14491)

## ✨ Reinforcement Learning

- [ ] **PPO + GAE** — [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) and [Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)
- [ ] **DQN (Dueling + Prioritized + Double)** — [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) plus extensions: [Dueling Network Architectures (Wang et al., 2016)](https://arxiv.org/abs/1511.06581), [Prioritized Experience Replay (Schaul et al., 2016)](https://arxiv.org/abs/1511.05952), [Double DQN (Van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)

## ✨ Counterfactual Regret Minimization

- [ ] **CFR** — [Regret Minimization in Games with Incomplete Information (Zinkevich et al., 2007)](https://poker.cs.ualberta.ca/publications/2014techreport.pdf)
- [ ] **Kuhn Poker** — reference rules (no dedicated paper; see classic game theory texts)

## ✨ Optimizers

- [ ] **Adam** — [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)](https://arxiv.org/abs/1412.6980)
- [ ] **AMSGrad** — [On the Convergence of Adam and Beyond (Reddi et al., 2018)](https://openreview.net/forum?id=ryQu7f-RZ)
- [ ] **Adam with Warmup / Noam** — [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [ ] **Rectified Adam** — [On the Variance of the Adaptive Learning Rate and Beyond (Liu et al., 2019)](https://arxiv.org/abs/1908.03265)
- [ ] **AdaBelief** — [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients (Zhuang et al., 2020)](https://arxiv.org/abs/2010.07468)
- [ ] **Sophia-G** — [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training (Liu et al., 2023)](https://arxiv.org/abs/2305.14342)

## ✨ Normalization Layers

- [ ] **BatchNorm** — [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)
- [ ] **LayerNorm** — [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
- [ ] **InstanceNorm** — [Instance Normalization: The Missing Ingredient for Fast Stylization (Ulyanov et al., 2017)](https://arxiv.org/abs/1607.08022)
- [ ] **GroupNorm** — [Group Normalization (Wu & He, 2018)](https://arxiv.org/abs/1803.08494)
- [ ] **Weight Standardization** — [Micro-Batch Training with Batch-Channel Normalization and Weight Standardization (Qiao et al., 2019)](https://arxiv.org/abs/1903.10520)
- [ ] **Batch-Channel Normalization** — same paper as above (Qiao et al., 2019)
- [ ] **DeepNorm** — [DeepNet: Scaling Transformers to 1,000 Layers (Wang et al., 2022)](https://arxiv.org/abs/2203.00555)

## ✨ Distillation

- [ ] **Knowledge Distillation** — [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)

## ✨ Adaptive Computation

- [ ] **PonderNet** — [PonderNet: Learning to Ponder (Banino et al., 2021)](https://arxiv.org/abs/2107.05407)

## ✨ Uncertainty

- [ ] **Evidential Deep Learning** — [Evidential Deep Learning to Quantify Classification Uncertainty (Sensoy et al., 2018)](https://arxiv.org/abs/1806.01768)

## ✨ Activations

- [ ] **Fuzzy Tiling Activations** — [Fuzzy Tiling Activations (Alcorn et al., 2021)](https://arxiv.org/abs/2106.07447)

## ✨ Language Model Sampling

- [ ] **Greedy Sampling** — technique (refer back to GPT/BERT papers)
- [ ] **Temperature Sampling** — technique; see [Empirical Evaluation of Generic Sampling Methods (Ackley et al., 1985)](https://www.sciencedirect.com/science/article/pii/S0893608085800100) for temperature interpretation
- [ ] **Top-k Sampling** — [The Curious Case of Neural Text Degeneration (Holtzman et al., 2020)](https://arxiv.org/abs/1904.09751)
- [ ] **Nucleus Sampling** — same Holtzman et al. (2020)

## ✨ Scalable Training / Inference

- [ ] **Zero3 Memory Optimizations** — [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning (Rajbhandari et al., 2021)](https://arxiv.org/abs/2104.07857) *(ZeRO stage 3 variant)*

---
Notes for future me:
- it’s fine if more papers/ideas get added over time, this list is supposed to grow
- if an item is more like “implementation notes” than a real paper, I’ll link whatever primary reference I actually used
- feel free to be messy here; the goal is to make it easy to pick up where I left off, not to be polished  🎯