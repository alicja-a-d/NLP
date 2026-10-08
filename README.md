# Natural Language Processing 2026: Assignments 2 and 4 (Group 33)

Coursework for the Natural Language Processing course (Vrije University, 2026). This repository holds the code, figures and written reports for two assignments:

| | Assignment | Topic | Main techniques |
|---|---|---|---|
| A2 | Text statistics, language models and word vectors | Zipf's law, n-gram language models, count-based word vectors, PMI | Tokenisation, add-alpha smoothing, perplexity, co-occurrence matrix + truncated SVD, GloVe, PMI/PPMI |
| A4 | Neural networks and neural dependency parsing | Optimisation and regularisation, MLP warm-up, transition-based dependency parser, error analysis | SGD, Adam, dropout, MLP, arc-standard transitions, UAS, POS-feature analysis |

**Authors (Group 33):** Raghav Prabhakar, Amee Tran, Alicja Dorobis

**Course aims** (from the course description): learn the fundamentals for developing a system for a common NLP task (pre-processing, classification set-up and evaluation, neural networks, pre-trained models), and learn to systematically analyse and interpret NLP models (linguistic challenges, comparing models, interpreting outputs).

---

## Assignment 2: Text Processing, N-gram Language Models and Word Vectors

Report: `GROUP33_A2_NLP2026.pdf`

### A2.1 Text processing and Zipf's law
- A "word" is defined as a lowercased token after punctuation removal and whitespace splitting, with optional stopword removal. This differs from a lemma (canonical form) and a type (unique word form).
- Compared Brown corpus genres (news against fiction). News has more formal, domain-specific terms; fiction has more pronouns and dialogue. Both follow Zipf's law (a few very frequent words, a long tail of rare ones). Without stopwords, differences between genres become much clearer.
- Extended to Hindi and Telugu using NLTK's Indian language corpus. Tokens were dominated by stopwords, which a team member who reads Hindi verified. This points to the need for better tokenisation and stopword lists for these languages.

### A2.2 N-gram language modelling
- Built unigram, bigram and trigram models from relative frequencies using a sliding window and the Markov assumption, with sentence boundary markers.
- Handled unseen n-grams with add-alpha (k-adding) smoothing. Increasing alpha flattens the distribution: for example P(`<s>` the) fell from 0.013 to 0.005 when alpha went from 0 to 5.
- Evaluated with perplexity on a test set. Surprisingly, the trigram model had the highest perplexity (626.68) against bigram (320.78) and unigram (155.42), because higher-order models suffer from data sparsity on a small corpus.
- Discussed what n-grams capture (local syntax) and what they miss (long-distance and hierarchical structure, semantics), and how neural language models address this.

### A2.3 Word vectors (Alicja)
What we did:
1. **Data selection:** the Brown corpus (romance and humour genres) and the Indian language corpus (Hindi), limited by computational constraints.
2. **Preprocessing:** added sentence start and end tags to every sentence.
3. **Co-occurrence matrix and verification:** checked counts for a frequent pair ("old man") and a rare pair ("neither liked") to confirm the matrix reflects the text.
4. **Dimensionality reduction:** started with a hand-written SVD, then switched to `TruncatedSVD`, which is much more efficient on large matrices.
5. **Visualisation:** sampled every 10th word of the 200 most frequent terms; translated Hindi tokens with a translation package so non-Hindi readers could interpret the plot.
6. **Comparison with GloVe:** compared our sparse, count-based vectors with pre-trained GloVe vectors (Brown corpus only).
7. **Stopword experiment:** compared embeddings with and without stopwords.
8. **Semantic tests:** synonym/antonym triples and analogies.

What we found:
- Count-based SVD vectors reflect corpus variance: distance correlates with frequency and co-occurrence strength more than with meaning. Social and family words (mother, father, wife) separate from high-frequency nouns (man, things).
- In 2 of 5 synonym/antonym triples the antonym was closer than the synonym (sad/happy and love/hate). Antonyms appear in the same contexts ("happy, not sad"), so distributional vectors capture topic similarity, not negation. This supports the distributional hypothesis (Firth) and its limits, and ties to the symbol grounding problem (Harnad).
- Analogy tests showed gender and occupation stereotypes learned from the training data.
- Stopwords dominate the vector space (the top 20 words were all stopwords); removing them lets content words such as "time", "man" and "thought" drive the variance. The sentence-start tag remained an outlier in both plots.
- GloVe plots form a ring because vectors are L2-normalised, so cosine similarity organises the space.
- The Hindi analysis was limited because the group could not apply manual stopword removal, which shows that preprocessing needs language knowledge.

### Bonus: PMI and PPMI
Pointwise mutual information compares how often two words co-occur against what independence would predict. High PMI pairs are fixed collocations (rare names that almost always appear together); low or negative PMI pairs are frequent function-word pairs ("of the"). PPMI sets negative values to zero, leaving a cleaner view of real associations. The results show the unigram independence assumption does not hold.

### Files
- `NLP_2026_A2.ipynb`, `problem1.ipynb`, `pmi.ipynb`: notebooks with the code
- `results/`, `visualization/`: figures and outputs
- `Raghav - The Starting Questions - Group Work.pdf`: group-work questions
- `GROUP33_A2_NLP2026.pdf`: written report

---

## Assignment 4: Neural Networks and Neural Dependency Parsing

Report: `Report.pdf`. The assignment combines conceptual machine-learning questions, an MLP warm-up and a neural transition-based dependency parser.

### Part 2: Machine learning and neural networks (Alicja)
Written explanations of three core training techniques:
- **Stochastic gradient descent:** approximates the gradient from a random subset of examples. The direction is only an estimate, but each step is far cheaper than an exact gradient, which is what makes large parsing corpora tractable.
- **Adam:** combines **momentum** (a weighted average of past gradients, which damps oscillations and smooths the descent) with an **adaptive learning rate** per parameter (dividing by the root of a rolling average of squared gradients, which normalises update sizes).
- **Dropout:** randomly deactivates nodes during training, which reduces co-adaptation and acts like training an ensemble of thinned networks. It is switched off at evaluation, using the full network with scaled weights to approximate the ensemble average.

### Part 3: Multilayer perceptron warm-up (Alicja)
Completed the MLP notebook (`A4_MLP_group.ipynb`) and wrote short answers to its questions.

### Part 4: Neural transition-based dependency parser
- Worked through the arc-standard transition system (SHIFT, LEFT-ARC, RIGHT-ARC) on a sample sentence, using a stack and a buffer; parsing is linear in sentence length.
- Data in CoNLL format (token, POS tags, head index, dependency label); gold files used for evaluation.
- Implemented the transition system, minibatch parsing and the neural parser, in the style of Chen and Manning (2014).
- **Results:** best development UAS 88.55% after 10 epochs; test UAS 88.76% using the best checkpoint.
- Discussed what UAS captures and misses: it counts correct heads but does not weight errors by how much they change the meaning (for example, a wrong long-distance attachment).

### Part 5: Error analysis
- Classified parser errors in four sentences: verb phrase attachment, modifier attachment, prepositional phrase attachment and coordination.
- Parsed the garden-path sentence ("The horse raced past the barn fell") with UDPipe, showed the incorrect and correct trees, and explained why a greedy transition parser struggles (it commits too early); discussed backtracking as a remedy (Dary et al.).
- Probed the trained parser to test whether POS tags help: they resolve lexical ambiguity ("They can fish"), signal coordination and verbal structure, but a wrong tag ("up" tagged as a preposition instead of a particle in "looked up the answer") leads to a wrong attachment.

--

## Setup

```bash
git clone <repository-url>
cd <repository-name>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Replace the placeholders with the real repository details and add a `requirements.txt`. The datasets (Brown corpus and Indian language corpus via NLTK, GloVe vectors, CoNLL dependency data) are provided by the course or by NLTK and are not redistributed here.

## Key references
- Jurafsky and Martin, *Speech and Language Processing*, Chapter 3 (n-grams).
- Pennington, Socher and Manning (2014). GloVe: Global vectors for word representation.
- Chen and Manning (2014). A fast and accurate dependency parser using neural networks.
- Kingma and Ba (2017). Adam: A method for stochastic optimization.
- Srivastava et al. (2014). Dropout: A simple way to prevent neural networks from overfitting.
- Dary, Petit and Nasr (2022). Dependency parsing with backtracking using deep reinforcement learning.
- Firth (1957) and Harnad (1990) for the distributional hypothesis and the symbol grounding problem.

The full reference lists are in each report.
