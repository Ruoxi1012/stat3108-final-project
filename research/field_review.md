# Field Review: Canine Vocal Affect Recognition

Research snapshot: **2026-10-11**

## Executive conclusion

The field has established that dog vocalizations contain acoustic information related to recording context, caller identity, breed, and human-perceived affect. It has **not** established a reliable audio-only “dog emotion translator.” The strongest open research problem for this course project is generalization: performance often drops sharply when the test dog was not present during training, emotion labels have uncertain ground truth, and high reported accuracies are frequently based on private datasets, random clip splits, heavily augmented source clips, or binary tasks.

The recommended project is therefore:

> **Audit and improve cross-dog and cross-breed prediction of behavior-annotated arousal and valence, with calibrated uncertainty and an abstention option.**

This framing is scientifically defensible, feasible on a small public dataset, visibly innovative, and well suited to a 10-minute presentation.

## 1. What “emotion” means in this project

Animal emotion cannot be read directly from audio. A dimensional framework describes affect using **valence** (negative to positive) and **arousal** (low to high), while behavioral, neural, and physiological observations act as indirect evidence. Mendl, Burman, and Paul explicitly note that conscious experience cannot be assessed directly and propose valence-arousal as an integrative framework ([2010](https://doi.org/10.1098/rspb.2010.0303)). A review of canine positive-emotion assessment concludes that no single method reliably establishes dog emotion and recommends combining behavioral, physiological, endocrine, and neural measures ([Csoltova & Mehinagic, 2020](https://doi.org/10.3389/fpsyg.2020.02131)).

Therefore our supervised target is precisely:

> **Human consensus labels of vocal arousal and valence, derived from observable behavior and video context, predicted from audio alone.**

It is not a clinical diagnosis, a literal translation, or direct access to an animal's private state.

## 2. What is already known

### 2.1 Acoustic cues do carry affect-related information

Controlled listening studies show that humans use pitch, tonality, call duration, and inter-bark interval when judging dog vocalizations. Low-pitched, noisy, rapidly repeated barks tend to be perceived as aggressive, while higher-pitched, more tonal, slower sequences tend to be perceived as fearful, playful, or happy ([Pongracz, Molnar, & Miklosi, 2006](https://doi.org/10.1016/j.applanim.2005.12.004)). Human ratings of dog and human non-verbal vocalizations also show shared relationships between basic acoustic structure, perceived valence, and perceived intensity ([Farago et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3917336/)).

These studies justify MFCC/eGeMAPS baselines and acoustic-cue analysis. They do not prove that a short bark uniquely identifies a dog's actual emotion.

### 2.2 Identity and context are entangled

Molnar et al. classified more than 6,000 Mudi barks with 43% context accuracy and 52% individual-dog accuracy, demonstrating that the same waveform contains both context and identity information ([2008](https://doi.org/10.1007/s10071-007-0129-9)). Perez-Espinosa et al. later reported strong dog dependence: context F-measure could exceed 0.80 in dog-specific settings but stayed at or below roughly 0.57 when leaving one dog out ([2015](https://doi.org/10.13053/rcs-100-1-6)).

This is the central confound for our project. A clip-level random split may reward dog or source recognition rather than affect recognition.

### 2.3 Traditional affective-computing features are weak but meaningful baselines

The Emotional Dog Corpus (EmoDog) contains 226 sequences from 12 Mudi dogs, seven contexts, and five perceived-emotion classes rated by professional trainers. Under leave-one-dog-out evaluation, eGeMAPS with a linear SVM achieved 28.5% unweighted average recall for five emotions (20% chance), while the best context result was 32.9% UAR ([Hantke, Cummins, & Schuller, 2018](https://doi.org/10.1109/ICASSP.2018.8461757)).

The result matters because it uses a dog-disjoint protocol and shows that honest performance is modest. It also makes eGeMAPS + SVM a more defensible baseline than accuracy on an arbitrary Kaggle collection.

### 2.4 Pretrained representations help, but have not solved affect generalization

Abzaliev, Perez Espinosa, and Mihalcea applied wav2vec 2.0 to 8,034 dog-vocalization segments and evaluated identity, breed, sex, and context. They used dog-grouped ten-fold validation for non-identity tasks. On a four-context task, pretrained wav2vec 2.0 reached 62.18% accuracy and improved minority-class F1 over training from scratch ([LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1432/)). The dataset contains 74 dogs but is distributed only by request.

Dog2vec is a HuBERT-based, canine-specific encoder continually pretrained on about 150 cleaned hours selected from more than 6,000 hours of online dog video. With frozen encoders and a common small head, Dog2vec outperformed wav2vec 2.0, HuBERT, AVES, and BioLingual across breed, identity, bark-type, and detection tasks; for example, it reached 0.7138 macro F1 on bark type versus 0.6318 for the best AVES variant ([Li, Zhu, & Wu, 2025](https://doi.org/10.21437/Interspeech.2025-1287)). It has not been benchmarked in that paper on arousal-valence prediction, creating a natural transfer-learning question for this project.

### 2.5 EmotionalCanines is the best feasible primary dataset

[EmotionalCanines](https://doi.org/10.1145/3746027.3758286) provides 1,400 clips (about 35 minutes) from 199 dogs: 700 Husky clips and 700 Shiba Inu clips. Each breed has 600 training and 100 test clips; train and test contain different dogs. Labels are three-level arousal and three-level valence, assigned from videos by three annotators using behavioral guidance and majority agreement. The preliminary annotation study reported Fleiss' kappa 0.30 for arousal and 0.42 for valence—fair-to-moderate agreement, which is itself scientifically important.

The paper's honest dog-disjoint test results are difficult: best reported accuracies were approximately 50%–57%, depending on breed and target. In contrast, ordinary ten-fold cross-validation using similar-distribution clips produced roughly 54%–79% accuracy. The authors explicitly attribute this gap to the absence of individual-dog overlap in the fixed test set. The dataset is public under CC BY-NC-SA 4.0 at [GitHub](https://github.com/tmdang1101/EmotionalCanines).

Limitations include only two breeds, online-video selection bias, uncontrolled recording conditions, categorical labels derived from video context, no public annotator-level distributions in the released CSV files, and no dog/source IDs beyond the fixed train/test partition.

## 3. Recent results that require cautious interpretation

### DogEmotionSound and Mamba models

Yang et al. introduced a private 4,229-sample dataset with five labels and reported 91.97% accuracy using a Mamba architecture and synchrosqueezed STFT features ([2024](https://doi.org/10.1016/j.eswa.2024.125213)). A related 2026 paper reported 92.71% on a 4,230-sample dataset ([Yang et al., 2026](https://doi.org/10.1016/j.engappai.2026.115679)). The original paper points to a GitHub repository, but that repository was unavailable when checked on 2026-10-11, and the dataset is not an accessible benchmark. The headline number is therefore not directly reproducible or comparable to EmotionalCanines' unseen-dog test.

### Multi-QuadEmoNet

Multi-QuadEmoNet reports 95% accuracy on a binary Dog Mood Audio Dataset containing “angry” and “sad” classes ([Pearline et al., 2026](https://doi.org/10.3389/fvets.2026.1799281)). The paper states that each dog class began with 100 internet-sourced examples and was augmented to 1,000, then uses an 80:20 split. It does not report a dog- or source-disjoint split. If augmented siblings can cross the split, results can be strongly inflated. The task is also binary and the data provenance and ground-truth process differ from EmotionalCanines, so 95% is not evidence that broad dog affect recognition is solved.

### CREMD

CREMD contains 923 dog videos and 12,794 distinct crowd annotations across presentation modes ([Baek, Cao, & Blackwell, 2026](https://arxiv.org/abs/2602.15349)). Visual context increased agreement; audio increased confidence for anger and fear, but the authors call the audio effect inconclusive because they lacked a no-context/audio-only condition and had limited clean audio. This supports our cautious interpretation: context and annotator background shape the label, and audio alone is an incomplete view.

## 4. Evidence matrix

| Work | Data / split | Task and method | Main result | Limitation relevant to us |
|---|---|---|---|---|
| Pongracz et al. 2006 | Artificial sequences from Mudi barks | Human affect ratings vs pitch, tonality, rhythm | Basic acoustic parameters systematically alter perceived emotion | Measures perception of controlled sequences, not ground-truth inner state |
| Molnar et al. 2008 | >6,000 barks, six contexts | Handcrafted features + ML | 43% context; 52% individual | Identity is highly learnable and can confound context/affect |
| Perez-Espinosa et al. 2015 | Induced Mudi barks; several validation schemes | Context classification | Per-dog F1 >0.80; unseen-dog F1 <=~0.57 | Strong individual dependence |
| Hantke et al. 2018 | 226 sequences, 12 dogs; leave-one-dog-out | eGeMAPS/ComParE/BoAW + SVM | 28.5% UAR for five perceived emotions | Tiny, single-breed, perceived labels |
| Abzaliev et al. 2024 | 8,034 labeled segments, 74 dogs; grouped CV | wav2vec 2.0 transfer | 62.18% accuracy on four contexts | Context, not arousal-valence; data by request |
| Yang et al. 2024 | 4,229 private clips; split details not a public benchmark | SST-STFT + Mamba | 91.97% accuracy | Dataset and current code unavailable; cross-dog validity unclear |
| EmotionalCanines 2025 | 1,400 clips, 199 dogs, two breeds; fixed unseen dogs | Handcrafted, pretrained, CNN/ViT baselines | Fixed-test best ~50%–57%; ordinary CV up to ~79% | Only two breeds; fair rater agreement; missing public dog IDs |
| Dog2vec 2025 | ~150 h clean canine pretraining audio | Frozen dog-specific SSL encoder | Best across five canine classification tasks; +8.2% relative overall | Not evaluated on affect labels; training repo has no explicit software license |
| Multi-QuadEmoNet 2026 | 100 originals/class expanded to 1,000/class | MFCC + LSTM/GRU | 95% on binary dog mood | Random 80:20; augmentation/source leakage not ruled out |
| CREMD 2026 | 923 videos, 12,794 annotations | Human annotation study | Context raises agreement; audio effect inconclusive | Not an audio-only benchmark; subjective labels |

## 5. The research gap

Five gaps recur across the literature:

1. **Evaluation gap:** random clip splits can mix the same dog, source video, or augmented family across train and test. Many papers do not report group-disjoint evaluation.
2. **Domain-shift gap:** even unseen-dog tests are usually within breed; cross-breed affect transfer is rarely measured.
3. **Representation gap:** dog-specific SSL representations now exist, but their value for arousal-valence recognition has not been established.
4. **Target-structure gap:** low/medium/high and negative/neutral/positive are ordered and related, but EmotionalCanines baselines treat them as independent nominal classes.
5. **Reliability gap:** published dog-affect systems emphasize accuracy, not calibration, uncertainty, abstention, or severe-error cost.

## 6. Our innovation

### Primary innovation

Use the same lightweight prediction head to compare handcrafted, human-speech, general/bioacoustic, and dog-specific representations under three increasingly difficult settings:

1. official within-breed, unseen-dog tests;
2. cross-breed zero-shot transfer (Husky to Shiba and Shiba to Husky);
3. pooled multi-breed training with the official held-out dogs.

Then add a joint ordinal arousal-valence head and calibrated selective prediction to the best representation.

### Why this is not a duplicate

EmotionalCanines released the data and a broad baseline sweep, but did not report Dog2vec affect transfer, a cross-breed transfer matrix, joint ordinal modeling, calibration, or abstention. Dog2vec established canine-specific representation quality, but did not test affect. Our project connects those two advances and asks a new reliability question.

### How AI is leveraged

- **Self-supervised foundation models:** transfer knowledge learned from large unlabeled canine or bioacoustic corpora into a 1,400-clip labeled problem.
- **Multi-task ordinal learning:** use shared acoustic structure while respecting the order of the labels. CORAL-style ordinal heads are an appropriate reference because ordinary cross-entropy ignores label order ([Cao, Mirjalili, & Raschka, 2020](https://doi.org/10.1016/j.patrec.2020.11.008)).
- **Uncertainty-aware AI:** temperature scaling provides a simple calibration baseline ([Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)); selective prediction reports the trade-off between coverage and error instead of forcing a guess on every clip ([Geifman & El-Yaniv, 2019](https://proceedings.mlr.press/v97/geifman19a.html)).
- **AI-assisted development:** generative AI may help write tests, experiment configurations, and documentation, but it will not create ground-truth labels or replace source verification.

## 7. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Labels are human interpretations | Claims overstate animal cognition | Say “behavior-annotated vocal affect”; report agreement and uncertainty |
| Small fixed test sets (100/breed) | High variance | Bootstrap 95% confidence intervals; avoid tiny claimed gains |
| Dog/source IDs absent from public CSV | Cannot construct new true group folds | Treat the official test as the only dog-disjoint benchmark; do not overclaim random-CV leakage |
| Breed correlates with label distribution | Pooled model learns breed priors | Report each breed, balanced metrics, and cross-breed transfer |
| Pitch/time augmentation changes affect cues | Label corruption | Use conservative gain/noise augmentation first; treat pitch/time changes as ablations |
| Dog2vec/fairseq setup is brittle | Time loss | Start with eGeMAPS and one maintained encoder; keep Dog2vec as the high-value comparison |
| Demo invites “translator” interpretation | Unsafe communication | Display probability, confidence, limitations, and abstain state; no advice or diagnosis |

## 8. Decision

Proceed with **EmotionalCanines as the primary dataset** and the generalization/reliability framing. Do not build the project around an unlabeled Kaggle “happy/sad/aggressive” collection or a private benchmark. The central success criterion is not beating a 95% headline number; it is producing a reproducible and honest answer about what survives on unseen dogs and breeds.

## Core references

1. [Mendl, Burman, & Paul (2010), animal emotion framework](https://doi.org/10.1098/rspb.2010.0303)
2. [Pongracz, Molnar, & Miklosi (2006), acoustic parameters and human ratings](https://doi.org/10.1016/j.applanim.2005.12.004)
3. [Molnar et al. (2008), context and identity classification](https://doi.org/10.1007/s10071-007-0129-9)
4. [Perez-Espinosa et al. (2015), individual dependence](https://doi.org/10.13053/rcs-100-1-6)
5. [Hantke, Cummins, & Schuller (2018), EmoDog](https://doi.org/10.1109/ICASSP.2018.8461757)
6. [Csoltova & Mehinagic (2020), canine emotion assessment review](https://doi.org/10.3389/fpsyg.2020.02131)
7. [Abzaliev, Perez Espinosa, & Mihalcea (2024), wav2vec 2.0 dog-bark classification](https://aclanthology.org/2024.lrec-main.1432/)
8. [Dang et al. (2025), EmotionalCanines](https://doi.org/10.1145/3746027.3758286)
9. [Li, Zhu, & Wu (2025), Dog2vec](https://doi.org/10.21437/Interspeech.2025-1287)
10. [Baek, Cao, & Blackwell (2026), CREMD](https://arxiv.org/abs/2602.15349)
