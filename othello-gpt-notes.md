
**Abstract:**
**Key Question:** What is the source of LLMs range of capabilities ? 
- Is it memorization of a collection of surface statistics?
- Is it an internal representation of the process that generates the sequence they see?
**Experiment**: Use a variant of a GPT model with the task to predict legal moves in the Othello board game.
**What they found ?** Evidence of an emergent nonlinear internal representation of the board state
- Representation used to control the output of the network.
- Produce latent saliency maps that help explain predictions

Key Questions:
- What is the othello board games ?
- What is an internal representation ? What makes it linear or nonlinear?
- What is a latent saliency map?

**Introduction:**
- **Question:** Next word prediction are powerful, how the performance emerges from sequence predictions remains ?
- **Argument**: Training a sequence modelling task is inherently limiting
	- Philosophical: Bender & Koller, 2020
	- Mathematical: Merrill et al., 2020
- **Argument**:  LLMs might do more than collect correlations, but build interpretable world models
		- **World Models** are understandable representations of the process producing the sequences they are trained on.
			- Example: Internal Representations of simple concepts such as color, direction or tracking boolean states
			- Another view is that the model store internal variables that map cleanly to real-world problems.
- **Paper**: Toshniwal et al. (2021) --> approach to study emergence of world states, and explores chess move sequences
	- How? Analyze LLM architecture behavior in a well-understood and constrained setting
	- Predicts legal chess moves with high accuracy. Analyzing predicted moves, it appears that the LLM tracks the board state
	- They do not explore form of the internal representations
- Questions I have:
	- What is out of distribution data problem?
	
*Othello as a testbed for interpretability*
- **Goal of paper:** The internal representation is the focus of this paper
	- We can think of the board as a world, games are testbeds to explore world representations by AI. They chose a simple game Othello (similar to chess) to explore how the AIs think when playing this
- **Training Dynamics**
	- First they trained a language model with move transcripts (Othello GPT) with legal moves, the model has no prior knowledge
	- Each token is a tile where the player places the disk
	- They do not train the model for strategically good moves or to wing the game
-  Next step is to look for world representations that might be used by the network
	- World in this context is the current board position
	- **Question**: can we identify a board state representation that produces next move prediction
		- To answer the question they train a set of probes (i.e. classifies) that infer board state from internal network activations
			- Question: What is a classifier?
	- The probe is a standard tool to analyze neural networks as seen in multiple papers
		- Papers: (Alain & Bengio, 2016; Tenney et al., 2019; Belinkov, 2016)
- a) Using this method they found evidence for an emergent world representation
	- b) Non-linear prove is able to predict the board state with high accuracy, linear probes however produces poor results 
	- **Questions**: Why linear probes produces poor results?
- c) Provide evidence that the representation plays a causal role in the networks predictions
	- Tool: Intervention technique that modifies internal activations to correspond to counterfactual states
- d) Knowledge of internal world models can be used as an **interpretability too**l, they use activation-intervention technique to create **latent salient maps**, which provide insight in how network makes predictions
	- **Questions:** what are latent salient maps

*"Language Modelling of Othello Game Transcript*
- LLMs show evidence of learning valid chess moves by observing transcripts in training data (Toshniwal et al., 2021)
- Othello was chosen because it has a Goldilocks zone not to simple and not to complex - has a sufficiently large game tree to avoid route memorization
	- **Question**: if the problem is sufficiently simple, are there easier models to build ? If the trees are more simple for example? Min Max?
- Othello's Rules
	- 8x8 board, with white and black disks
	- Objective is to have the majority of one's color discs in the board at the end of the game
	- Game Rules
		- (A) The board is always initialized with four discs (two black, two white) placed in the center of the board
		- (B) Black always moves first
		- (C) Every move must flip one or more opponent discs by outflanking—or sandwiching—the opponent
		- (D) A game ends when there are no more legal moves.
- Datasets for training
	- Used two sets of training data for the system
	- They call the "Championship" and "Synthetic"
	- Each captures different objectives, data quality and quantity
	- Championship data reflects strategic moves by expert human players
	- Synthetic data is larger set consisting of legal but otherwise random moves
	- Championship:
		- two online sources --> containing 7k and 132k games, they are combined and split randomly using 8:2 training and validation sets
	- Synthetic:
		- Created a synthetic dataset with 20 million games for training and 3 million for validation 
		- Computed the dataset by uniformly sampling leaves from Othello game tree
			- Question: what is uniformly sampling leaves from the Game tree ? How do you do that?
		- Data distribution show no strategy 
- Model and training
	- Goal is to study how much Othello-GPT can learn from pure sequence information, so they provide few inductive biases as possible
		- Question: What are inductive biases?
	- Use only sequential tile indices as input to the model
	- Each games is treated as a sentence tokenized with a vocabulary of 60 words
	- Trained an 8-layer GPT Model (Radford et al., 2018; 2019; Brown et al., 2020)
		- 8 head attention mechanism and 512- dimensional hidden space
		- Question: what the is this 8 head stuff
	- Training was performed in autoregressive fashion
		- **Question:** What is Autoregressive training? 
		- **Answer:** The model is trained to predict the next word (or move) based only on previous ones.
	- Input Representation:
		- Input is a partial game - which is a sequence of moves: $\left\{ y_t \right\}_{t=0}^{T-1}$
		- Each move $y_t$ is mapped to a **trainable word embedding**: 60 possible words/moves → 60 vectors
			- These embeddings form a sequence $\left\{ x^0_t \right\}_{t=0}^{T-1}$, which are the inputs to the transformer
			- Question: What is a trainable word embedding
	- Transformer processing
		- Each input vector $x^0_t$ passes **sequentially through 8 transformer layers**.
		- The representation after layer $l$ at position $t$ is denoted $x^l_t$.
		- **Causal Masking** is applied:
			- In transformers trained **autoregressively**, the model should only look at **past and current tokens**, not future ones.
			- This is enforced using a **causal mask**: it blocks information from future positions.
			- $x^l_t = \text{feature of token } t \text{ after layer } l$
			- It is the **representation (or hidden state)** of the token at position $t$ after passing through layer $l$ of the transformer.
			- At **layer** $l$, when computing $x^l_t$, the model can **only use the outputs from layer** $l-1$ at **time steps $≤ t$**.
			- This means:
				- $x^l_t \text{ is computed using } \left\{ x^{l-1}_0, x^{l-1}_1, …, x^{l-1}_t \right\}$
				- It **cannot** see $x^{l-1}{t+1}, x^{l-1}{t+2}, \ldots$, ensuring the model doesn’t “cheat” by looking ahead.
	- Prediction and training:
		- The final output from the last layer at the last position, $x^8_{T-1}$, is used to predict the **next move** $\hat{y}_T$.
		- A **linear classifier** (a final layer of weights) turns this vector into **logits** over all possible moves.
		- The model is trained using **cross-entropy loss**, comparing the predicted move vs. the true next move, and optimizing via **gradient descent**.
		- Questions:
			- What is a linear classifier?
				- A **linear classifier** is the simplest form of a classifier in machine learning. It makes predictions based on a **linear combination** of the input features.
				- A linear classifier:
					- Takes an input vector $x$
					- Multiplies it by a **weight matrix** $W$
					- Adds a **bias** $b$
					- Outputs a **logit vector** — one score per class
						- $\text{logits} = W x + b$
			- What are logits?
				- A **logit** is the raw (un-normalized) output score from a model **before** applying a softmax function.
				- It can be a scalar, vector, matrix, tensor
			- What is cross-entropy loss?
				- A mechanism to measure how well a classification model's predicted probability distribution matches true labels
				- Cross-entropy compares two probability distributions:
					- The **true distribution** (which is 1 for the correct class, 0 for others)
					- The **predicted distribution** from the model
					- It **punishes incorrect and overconfident predictions** more than uncertain ones.
				- Formula:
					- $\mathcal{L} \;=\; -\sum_{i=1}^{N} y_i \,\log\bigl(\hat{y}_i\bigr)$
					- $\mathcal{L} \;=\; -\sum_{t=1}^{T} \sum_{i=1}^{N} y_{t,i}\,\log\bigl(\hat{y}_{t,i}\bigr).$
					- $N$ is the number of classes.
					- $y_i$ is the true label (one-hot, so only one $y_k=1$).
					- $\hat{y}_i$ is the predicted probability for class i.
					- summing over a sequence of length $T$ (e.g. in language modeling)
	- Evaluating model's prediction adhering to the rules of Othello
		- Asked OthelloGPT to predict the next legal move conditioned by the partial game before that move
			- Error rate of 0.01% for synthetic
			- Error rate of 5.17% for championship
			- Error rate of untrained 93.29%
	- Is the model memorizing the steps
		- They created a skewed data set with truncated node C5, removing a quarter of the whole game tree and it still yielded 0.02% error rate
		- This is evidence that it is not memorizing, then what is it happening
	- **Exploring internal representations**
		- Does the model compute internal representations of the game state?
		- Tool used --> "Probe"
		- **A probe:** is a classifier or regressor whose input consists of internal activations of a network, and which is trained to predict a feature of interest, e.g., part of speech or parse tree depth (Hawitt & Manning 2019)
		- If a probe is able to be trained and is accurate then there is proof of internal representation
		- To study this they train probes to predict board state from network internal activations after a given sequence of moves
		- We take this autoregressive network activation $x^{\,l}_{t}$ to summarize the partial sequence of steps $y_{⩽t}$
		- Output is a three way categorical probability distribution  $p_{θ}(x^{l}_{t})=(p_{1},p_{2},p_{3})$
	- What is RELU, non-linear, and linearity of a classifier?
	- *Meta: Going forward I will use the framework Claim - Evidence - Mechanism - Questions for each subsection (CEMQ)*
	- **Linear Probes Have High Error Rates ?**
		- **Claim:** Linear classifier probes have poor relative accuracy. If there is a board state representation it is not of simple linear forms.
		- **Evidence:** Error rates where always higher than 20% for linear probes, even when compared to a baseline probes with randomly initiated networks
		- **Mechanism:** Trained probes on internal representations to show p(tile state). Function: Internal presentation --> Probe --> Board State  Prediction
			- Further explanation:
				- Simple diagnostic model you attach to an internal representation $x^{\,l}_{t}$ (hidden state of token $t$ at layer $l$)
				- You train only this probe to predict some property
			- **Probe function:**
			$$p_\theta(x^l_t) \;=\; \mathrm{softmax}\!\bigl(W\,x^l_t\bigr),
	\quad
	\theta = \{\,W \in \mathbb{R}^{F \times 3}\}.
	$$
				- Where $x^{t}_{l}∈R^{F}$ $is your **input vector** (of dimension F).
				- $W∈R^{F×3}$ is the **weight matrix** you learn (that’s your $\theta$).
				- The product $W x^{l}_{t}$ gives you a length-3 vector of raw scores (one for each of the 3 possible moves)
				- **Softmax** turns those 3 scores into probabilities that sum to 1:
					- $$
	\mathrm{softmax}(z)_i \;=\; \frac{\exp\!\bigl(z_i\bigr)}{\sum_{j=1}^{K} \exp\!\bigl(z_j\bigr)},
	\quad
	\text{for }i=1,\dots,K.
	$$
					- It turns each score (including negatives) into a positive number, exaggerates the differences
					- Normalization - divides by the sum:
						- Scales all exponentiated scores so they add up to 1
						- Converts raw scores into valid probability distributions over K classes
			- Error rates never dip below 20%
			- Baseline of probes trained on randomly initialized network (what does this mean?)
		- **Questions:**
			1. **How are the “baseline” probes trained?**  
			    Take the internal activation $x^{\,l}_{t}$ (layer $l$, step $t$) and train a 3-class classifier whose output $p_\theta(x^{\,l}_{t})$ is a categorical distribution over $\{\text{black},\text{white},\text{empty}\}$. They use an 8:2 train/validation split of $(x^{\,l}_{t},\text{tile label})$ pairs. The _baseline_ is the exact same probe trained on activations from a **randomly initialized** GPT; only the probe is trained, the network is not.
	    
			2. **Baseline, nonlinear, linear probes — what’s the difference?**  
				- Linear: information can be extracted with a simple weighted sum (e.g., $y= ax +b$ )
					- Probe architecture -->  $Linear = pθ(x)=softmax(Wx)$. 
				- Non-linear: Need more complex functions to extract information (e.g., neural networks)
					- Probe architecture:  $pθ(x)$ = $softmax(W_{1} ReLU(W_{2}x))$ )
				- Baseline refers to the features you are probing from a random neural network vs. trained Othello GPT network
		    3. **Why softmax for probabilities? What assumptions?**
			    Softmax turns unbounded scores into a normalized 3-class categorical distribution, plays nicely with cross-entropy (maximum-likelihood for a categorical target), is differentiable, and lets you compare relative evidence between classes. 
			    Core assumptions: 
				    (i) classes are mutually exclusive (each tile is exactly one of {black, white, empty}); 
				    (ii) examples are treated as i.i.d. for training; and 
				    (iii) the model family (linear or MLP) is adequate to map activations to class logits. In the paper, they explicitly frame the probe output as a 3-way categorical probability distribution
		3.2 Nonlinear probes have lower error rates
			**Claims:** Nonlinear probes have lower error rates than linear probes. Probe may be recovering an nontrivial representation of board state in the network activations
			**Evidence:** Lower error rates compared to both baselines and linear probes
			**Mechanism:** Trained 2-layer MLP as a probe.
				Function: 2- layer MLP
				- Two layers refer to two affine layers with a nonlinearity in between (plus softmax at the end if you want probabilities)
				- Affine map is a linear map (transformation) plus a shift of the like $f(x) = Wx + b$
		4.0 Validating probes with interventional experiments?
			Idea: Prove that the the internal representation has a causal effect on the model prediction
			