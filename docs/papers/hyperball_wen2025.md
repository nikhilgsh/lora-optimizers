# Fantastic Pretraining Optimizers and Where to Find Them 2.1: Hyperball Optimization

Authors: Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, Percy Liang

## TL;DR

We propose an optimizer wrapper called Hyperball that normalizes the Frobenius norm of both weights and optimizer updates of all matrices in the neural network throughout training instead of using weight decay. This operation leads to 20-30% speedup over weight decay and hyperparameter transfer across widths and depths. 

## Section 1: Motivation

In our previous paper Fantastic Pretraining Optimizers and Where to Find Them, we observed that the speedups of matrix-based optimizers including Muon over AdamW shrink from 30% to only 10% as model size and data scale grow. We've been searching for a way to keep those speedups at higher compute since.



It turns out the solution is extremely simple. We introduce a a simple optimizer wrapper that enforces constant weight and update norms, transforming any base optimizer into its hyperball variant (e.g., Muon → Muon Hyperball). This small change leads to two empirical benefits: (1) It preserves optimizer speedups across scales, and (2) it allows hyperparameters to transfer without retuning.



## Section 2: Hyperball Optimization

Most modern LLM training uses weight decay, which controls the size of the weights implicitly. Let ⁍ be the weight matrix at step ⁍, ⁍ be the update provided by a base optimizer (e.g., from Adam), ⁍ be the learning rate, and ⁍ be the weight decay coefficient. The standard update rule is:

$$ W_{t+1} = (1 - \eta \lambda) W_t - \eta u_t $$

Here ⁍ adds the new update information and typically leads to increasing weight norm without weight decay. The term ⁍) softly controls the norm by shrinking the weights towards zero every step.

Hyperball replaces this soft control on weight norm with an explicit constraint. It decouples the magnitude of the weights from the direction of the update entirely. To define the update, we first introduce the following notation:

The Hyperball update rule is defined as:

$$ W_{t+1} = R \cdot \text{Normalize}\left(W_t - \eta R \cdot \text{Normalize}(u_t) \right)  $$

Geometrically, Hyperball constrains the optimization trajectory to lie strictly on the surface of a hypersphere with radius ⁍. The update takes a step of length ⁍ in the direction defined by the normalized update ⁍, and the result is immediately projected back onto the sphere. This ensures that the norm of the weights and updates remains constant, while the optimizer purely navigates the direction of the updates.

[image]

Here ⁍ can be the optimizer update from any optimizer. In this blog, we focus on two variants: Adam-Hyperball (AdamH) and Muon-Hyperball (MuonH). 

Empirical Tips:

- Hyperball is applied to all non-embedding matrices in the neural network. All remaining parameters, including those in RMSNorm and the word embedding layers, are still optimized using Adam. When we use MuonH for the rest of projection, we use AdamH for the LM head.

- The step size ⁍ intuitively represent how strongly the current update should influences the next weights. We empirically find that Hyperball optimizers prefer a learning rate between 2.5e-3 to 1e-2. It also has better hyperparameter transfer property as will discuss in Section 3.2.

- The radius ⁍ fixes the weight norm and can be set once. In our experiments, we set it to be the initial norm ⁍ where we randomly initialize each parameters with a standard deviation of ⁍. We expect Hyperball to tolerate a wide range of initialization schemes, since it keeps the relative update size the same across initializations.





## Section 3: Experiments

### Section 3.1: Empirical Speedup

We evaluated Hyperball's training speedup across multiple scales and settings. Unless otherwise specified, we uses a Qwen3-like architecture with QK-Norm and train on a mixture of DCLM-baseline, StarCoder and ProofPile 2. For the concrete hyperparameters, one can refer to the corresponding Wandb runs by clicking on the curves. 

Head-to-head Comparison with Weight Decay. In the below run where we compare Muon and MuonH on a 1.2B model. We show that MuonH reaches 0.03 lower final loss while maintaining a constant weight norm through out training. One important thing to note is that Hyperball optimizers typically start with a higher loss but overtake baseline methods once the learning rate decays.

[embed: ]

Quantitative Speedup on 1.2B Models. We then quantify optimizer speedup by training 1.2B-parameter Qwen3 models. We train models over 4 Chinchilla ratios: 1× means the token budget is 20x the non-embedding parameters and 2×/4×/8× mean training on 2/4/8 times more tokens than that reference. We then fit a scaling law for AdamW and compute to reach the same final evaluation loss, how many tokens would AdamW need compared to our new optimizer / Muon? 

This is the same setup as in our previous paper (shown in the left plot) except that now we incorporate QK-Norm in the new architecture. We use the same hyperparameters configuration as our previous papers and re-tune learning rates for all the optimizers. The original Muon runs which only shows 10% speedup in this model scale, similar to the result in our original paper. In the meantime, our Hyperball optimizers demonstrates 20-30% speedup and the speedup increases with respect to the training duration. See the original runs here.

[image]



Validation in Marin’s speedrun  We further test both AdamH and MuonH in the setting of Marin’s speedrun, where we train on the FineWeb-Edu data for different models in the 1x Chinchilla regime. Both AdamH and MuonH show persistent speedup over their weight decay counterparts with increasing scales and can match models with 10% more parameters in 1x Chinchilla regime. 

[embed: ]

Scale up to 8B model. We further scaled MuonH up to 8B parameter models in Marin Ferries and compare MuonH with our previous AdamW baseline. We observe a surprising 0.04 loss improvement for the 8B experiments. Note that there is a catch that both MuonH and AdamW adopts a manual chosen hyperparameters so likely there are rooms to improve both of them.

[embed: ]

Over-training Stress Test. Our previous paper observed that optimizer’s speedup can diminish with increasing training durations. We show here Hyperball maintained its performance advantages for overtrained 130M models even when we pushed training far beyond typical Chinchilla budgets. 

[embed: ]

### Section 3.2: Hyperparameter Transfer (Depth & Width)

Hyperball's second compelling property is its ability to transfer hyperparameters across different model architectures without retuning. Optimal hyperparameters often changes with scales, even when one adopted initialization schemes like MuP (e.g. Atil et.al and Fan et.al). This makes it challenging to find suitable hyperparameters for large model training. 

Hyperball enables hyperparameter transfer by explicitly control the effective step size in the direction space. The key insight behind this is that the ratio between weight norm and update norm is the main factor governing in optimization dynamics according to Spectral Condition. This approach maintains the model in a feature learning "sweet spot" during the optimization process.

We validated this experimentally across two dimensions. In our depth scaling experiments, we fixed the hidden dimensions at ⁍ while varying the number of layers from L=4 to 512 for 10B tokens. Below, we plot the optimal learning rate for different widths using a multiplicative grid with ratio  ⁍. The maximal drift of the optimal learning rate window was only 1.4x. 

[image]

Similarly, in our width scaling experiments, we fixed the number of layers at ⁍ while varying hidden dimensions from ⁍ to ⁍ for 10B tokens and plot the optimal learning rates with respect to different widths. Again, the maximal drift of the optimal learning rate window was just 1.4x for both AdamH and MuonH (refer to runs here).

[image]



## Section 4: Mechanism Behind Hyperball

### Sec 4.1: RMSNorm's Rescaling Parameter Preserves Representation Power

One might worry that fixing weight norms would limit what the network can learn. Fortunately, this isn't the case for architectures that use RMSNorm with rescaling parameters ⁍. 

$$ \mathrm{RMSNorm}(h; \gamma) = \gamma\odot h / \|h\|_{\mathrm{rms}} $$

For many linear weights in Transformers, the input is preprocessed by RMSNorm before feeding it into the corresponding linear layer. Concretely, with ⁍ as the weight matrix, the following structure is commonly seen:

$$ f(h; W, \gamma) = W \times \mathrm{RMSNorm}(h; \gamma) = W (\gamma\odot h / \|h\|_{\mathrm{rms}})  $$

The key observation is that ⁍ —scaling the weights and inversely scaling the rescaling parameter produces the same output. This means fixing the norm of ⁍ doesn't restrict the function class the network can represent.

### Sec 4.2: Why Hyperball Speeds Up Training

Hyperball achieves faster training by cleanly separating two questions: "how big are the weights?" and "how fast do their directions change?"

Weight decay couples these two factors. With standard weight decay, there are two forces deciding the weight norm. Due to the existence of gradient noise, optimizer update has a stable angle with the weight, and causes each weight matrix's norm to increase. Meanwhile, the weight decay term shrink the weight norm towards 0. The two effects will eventually balance and each weight matrix's norm will drift toward an equilibrium value determined by hyperparameters including learning rate and weight decay factor. This equilibrium norm then implicitly sets the relative step size ⁍, which controls how quickly weight directions evolve (Li. et.al., Simon et.al).

But neural networks are approximately scale-invariant. Since features are typically normalized (via LayerNorm or RMSNorm), performance depends much more on the direction of weights than their absolute scale. What really matters for learning is the relative step size—how much the weight directions rotate each step.

Hyperball decouples scale from directional learning speed. By explicitly fixing the weight norm and normalizing updates, Hyperball gives direct control over the directional update speed. You can schedule this speed independently (e.g., with linear or cosine schedules), allowing the model's features to evolve at the optimal rate throughout training. This decoupling is what enables faster, more efficient learning.



## Section 5: Related Methods

While less common in modern large-scale LLM training, normalization and constraints on weight matrices have a rich history.

- Reparameterization and Standardization

- Constraints in Generative Models

- Constraints in Update 

- Fixed Norms in LLM Pretraining

- (Spectral-related) Manifold Optimization

[image]

## Acknowledgement

The authors would like to thank Songlin Yang, Zihan Qiu, and Liliang Ren for motivating this blog post into existence. To some extent, this work is a proof of concept to show that it is possible to remove weight decay altogether by designing the optimizer to explicitly control weight norms. The authors would also like to thank William Held, David Hall, Suhas Kotha, Tatsunori Hashimoto, Jason Lee, Zhiyuan Li, Lijie Chen, Huaqing Zhang, Jiacheng You, Jeremy Bernstein, Shu Zhong and Samuel Schoenholz for helpful discussions. We would like to specially thank Google TRC compute for making all the experiments possible.

---

## Citations

If this work is helpful to you, please consider citing:

```
@online{wen2025hyperball,
  title   = {Fantastic Pretraining Optimizers and Where to Find Them 2.1: Hyperball Optimization},
  author  = {Wen, Kaiyue and Dang, Xingyu and Lyu, Kaifeng and Ma, Tengyu and Liang, Percy},
  year    = {2025},
  month   = {12},
  day     = {15},
  url     = {https://tinyurl.com/muonh},
  urldate = {2025-12-15}
}
```

