# STAT UN3108 Final Project - Durable Project Memory

Last updated: 2026-10-18

## Active team and repository

- Public GitHub repository: `https://github.com/Ruoxi1012/stat3108-final-project`
- Default branch: `main`
- Team members: Ruoxi Zhao and Cici Zeng
- Current repository state as of 2026-09-18: initial course-required directory scaffold exists; README project title, overview, and goal/question are not yet filled in.
- Preserve visible, meaningful contributions from both members through feature branches, pull requests, and commits over time.

## Selected project direction (recorded 2026-10-11)

- Working title: **Bark Beyond Identity: Auditing and Improving Cross-Dog Generalization in Canine Vocal Affect Recognition**.
- Task framing: predict behavior- and context-annotated arousal and valence from short dog vocalizations. Do not describe the system as reading a dog's private emotional state.
- Primary benchmark: EmotionalCanines (ACM MM 2025), which provides 1,400 vocalization sequences from 199 Huskies and Shiba Inus and official tests containing dogs not present in training.
- Central research gap: high results from clip-level or augmentation-heavy splits may reflect dog, breed, source, or duplicated-audio cues rather than affect; generalization to unseen dogs and breeds is still weak.
- Confirmed innovation plan: (1) paired original/noisy/enhanced audio experiments, (2) evaluation on unseen dogs and cross-breed transfer without claiming coverage of every breed, and (3) confidence-aware output containing the predicted state, class probabilities, and a validation-calibrated reliable/uncertain decision.
- Current phase: data pipeline and audit complete; baseline modeling is next. No model results have been produced yet.

## Week 2 checkpoint (2026-10-18)

- EmotionalCanines was retrieved at commit `6b73c5fce55fc132e97db6a76899f8594a955f59`.
- All 1,400 released WAV files decode as mono 16 kHz audio.
- The exact decoded-audio audit found five duplicate groups. Two cross the
  official Shiba train/test boundary, three have conflicting labels, and one
  group has both problems.
- The generated manifest excludes five training rows from model fitting: the
  two train-side test duplicates, both members of a conflicting Husky training
  pair, and one repeated same-label Husky training copy. This leaves 1,195
  eligible training rows. All 200 official test rows remain for the primary
  score, with a duplicate-sensitivity analysis required.
- A licensed ESC-50 wind clip produced verified 10 dB and 0 dB paired smoke
  examples. This verifies generation and provenance, not classifier robustness.
- The clean Python environment and four audio-pipeline tests pass. A three-slide
  presentation checkpoint covers motivation, the official split and audit, and
  the paired-noise design.

## Purpose and source of truth

This file is the persistent, searchable memory for the STAT UN3108 (Applied Deep Learning and AI, Fall 2026, Section 001) final project. It summarizes two archived course documents:

- `docs/course-materials/Lecture4.pdf` - 67 lecture slides on the final project, Git/GitHub, and AI-assisted coding.
- `docs/course-materials/Final_Project_Deliverables.pdf` - 10-page formal deliverables and grading document.

When exact wording, layout, QR codes, or linked examples matter, consult the archived PDFs. The formal deliverables PDF should take precedence over the lecture summary if the two differ. Future instructor announcements may supersede both.

## What the project actually asks us to do

Choose a substantial AI/deep-learning topic and carry it through a complete evidence-based project cycle:

1. Define a clear, scoped problem, research question, hypothesis, or development goal.
2. Situate it in prior work and explain why the problem and any limitations of existing work matter.
3. Select one of the explicitly accepted paths:
   - reproduce or investigate a paper/result;
   - build a working system;
   - conduct a rigorous learning-based study, survey, or investigation.
4. Implement or investigate the approach with explicit models, data, tools, metrics, design choices, and modifications.
5. Evaluate it quantitatively where appropriate and qualitatively through examples, error analysis, limitations, and reflection.
6. Iterate meaningfully rather than producing one end-of-term result.
7. Explain the work clearly in a 10-minute poster/slide presentation, a concise two-page report, and a reproducible public GitHub repository.
8. Maintain visible, sustained, meaningful Git commits from every group member.
9. Attend the presentation sessions and have each student thoughtfully review two other groups.

The project is therefore not judged mainly on whether a published number is matched exactly. A reasonable implementation or investigation, honest analysis of discrepancies, meaningful iteration, and clear understanding are the central goals.

## Grade structure and deadlines

| Component | Weight | Due / timing | Essential requirement |
|---|---:|---|---|
| Poster or slides presentation | 60% | December 8 or 10, 2026 | 10-minute presentation; poster or slides accepted |
| Two-page project summary | 20% | December 13, 2026 | PDF, maximum two text pages excluding images and references |
| Public GitHub repository | 10% | December 15, 2026 | Reproducible repo; sustained individual commit history is central |
| Peer review and attendance | 10% | Presentation sessions | Each student reviews two other groups and asks at least one question per review |

Additional timing rules:

- Sign up for a presentation slot using the instructor's `Final Project Presentation Times` sheet.
- The instructors aim for 9-10 groups per day, maximum 10, and may rebalance days after signup.
- Poster PDF is due through GitHub by the group's presentation day.
- The report goes to both Gradescope and GitHub.
- The repository link goes to Gradescope by December 15.

## Highest-leverage grading strategy

- Put the largest share of effort into the presentation because it is 60% of the grade.
- For a built system, a smooth, well-explained live demo is the most reliable presentation booster. It should handle representative edge cases and support the story rather than replace evaluation.
- A learning-based study/survey does not need a demo and is not penalized for lacking one; it is judged instead on depth, synthesis, understanding, and clarity.
- Start the public repository immediately. The repo score is chiefly a history of individual contributions over time, not a polished final upload.
- Favor multiple meaningful experimental or design iterations. The report grading guidance explicitly rewards evidence of iteration.
- Optimize every artifact around one coherent arc: motivation -> question/goal -> method -> evidence -> interpretation -> limitations/reflection.

## Presentation / poster contract (60%)

### Format and logistics

- Either a traditional poster or PowerPoint/slides is accepted.
- Presentation limit: 10 minutes. Rehearse to stay within time.
- Poster maximum: 36 x 24 inches; landscape preferred.
- A poster may be one printed poster or collated A4 sheets.
- Submit a PDF with all fonts and images embedded.
- If using a poster, bring a printed version to class.
- The entire group must attend the full poster session and be ready for questions.

### Required content

1. Title and authors, including all group members and affiliation.
2. Introduction/background/motivation:
   - problem and why it matters;
   - clear goal, hypothesis, or research question;
   - specific result to reproduce or investigate, if applicable;
   - chosen paper and its main contributions, if applicable.
3. Methodology:
   - models, datasets, tools, and approach;
   - important modifications and design choices;
   - extensions or experiments beyond the original work.
4. Results:
   - graphs, tables, charts, images, or demo evidence;
   - comparison with original findings, baselines, or expectations.
5. Conclusion with key takeaways.
6. Future work (optional).
7. References for the original paper and every resource used.

### Design requirements

- Body text should be at least 24 pt on a poster; headings larger.
- Use concise bullets, readable fonts, strong contrast, and minimal dense prose.
- Every visual needs a caption and should be referenced in the surrounding explanation.
- Interpret patterns and discrepancies; offer hypotheses for unexpected findings.

### Exact presentation rubric

Each of 12 criteria is scored 1-5, totaling 60 points, which directly corresponds to the presentation's 60% weight:

1. Context and background.
2. Research questions/goals and scope.
3. Technical contributions.
4. Practical impact, including application, scalability, efficiency, or UX where relevant.
5. Quantitative results, baselines, measurements, and where appropriate ablations/significance.
6. Qualitative analysis, including cases, errors, limitations, and future work.
7. Visual clarity.
8. Effective visuals and text economy.
9. Clear, well-paced verbal summary within the allotted time.
10. Live demo booster (optional for learning-based projects).
11. Thoughtful engagement with reviewers and real-time problem solving.
12. Overall interest, engagement, and value.

Scoring labels: 1 poor, 2 limited, 3 good/meets expectations, 4 very good/above average, 5 amazing/exceptionally clear, creative, and insightful.

## Two-page report contract (20%)

- Due December 13, 2026.
- PDF, maximum two pages of text; images and references are excluded from that limit.
- Recommended: standard IEEE conference template, 10 pt, two columns.
- Also acceptable: single-column, 11-12 pt, single-spaced.
- Required filename: `group_topic_2page_report.pdf`.
- Submit to Gradescope and GitHub.
- Shorter and focused is preferred. A longer version may be kept separately, but the required submission is the concise summary.

Required sections:

1. Introduction - problem, motivation, and paper/method/contributions if applicable.
2. Chosen Result - exact target, why it matters, and relevant source figure/table/equation if applicable.
3. Methodology - architecture/approach, datasets, metrics, and modifications.
4. Results and Analysis - comparison to source/baselines/expectations, discrepancies, challenges, and meaning.
5. Reflections - lessons, takeaways, future extensions, and what could be improved.
6. References - paper, data, tools, frameworks, and other resources.

The highest-value report space is Methodology, Results and Analysis, and Reflections. Grading considers the challenge/interest of the problem, thoughtfulness of the approach, awareness of prior methods and limitations, obstacles and iteration, suitable metrics, insightful visualizations or toy examples, theoretical assumptions/limitations where relevant, group size where appropriate, and evidence of meaningful iteration.

Note for non-reproduction projects: the formal template uses reproduction-oriented language, but the overview explicitly permits systems and rigorous studies/surveys. Map "Chosen Result" to the central result, claim, capability, or research question, and compare against suitable baselines or expectations. Confirm this mapping with instructors if the fit remains ambiguous.

## GitHub repository contract (10%)

- Public repository; add all group members as collaborators.
- Submit the link through Gradescope by December 15, 2026.
- Central grading signal: each member's visible, sustained, meaningful contributions over time.
- One large end-of-term upload is specifically identified as weak evidence and will score poorly.
- Organization, documentation, and reproducibility still matter.

Required top-level structure:

```text
README.md
code/
data/
results/
poster/
report/
LICENSE
.gitignore
```

If data cannot be committed, `data/README.md` should explain exactly how to obtain it.

README content:

1. Introduction and selected paper/topic.
2. Live demo link and setup, if one exists.
3. Chosen result and its significance.
4. Repository organization.
5. Re-implementation/investigation details: models, data, tools, metrics, challenges, modifications.
6. Exact reproduction steps: commands, dependencies, arguments, environment, and compute/GPU requirements.
7. Results/insights and expected end result.
8. Conclusion and lessons.
9. References.
10. Acknowledgements, including course context and relevant tool use.

The README is an accessible snapshot, not a duplicate full report; aim for roughly one or two lines or figures per section where possible.

## Peer review and attendance contract (10%)

For every student individually:

- attend the presentation/poster sessions;
- review two assigned groups, normally on the day the student's own group is not presenting;
- engage with each presentation and ask at least one question;
- submit one short review form per assigned presentation;
- give thoughtful, specific, respectful feedback on clarity, rigor, originality, and visual design.

Perfunctory reviews or missed sessions reduce this score. The lecture indicates the completed form filename pattern `CUID_ProjectID_review.pdf`; verify the exact form workflow when the instructors release it.

## Required working process

Recommended team Git loop from the lecture:

1. Clone the team repository once per person.
2. Create a feature branch for each scoped task.
3. Commit small, meaningful units with clear messages.
4. Push the branch.
5. Open a pull request and have a teammate review it.
6. Sync/rebase the feature branch as appropriate, then merge into `main`.
7. Pull `main` and repeat.

Do not rewrite teammates' published history casually. Resolve conflicts by understanding and editing the conflicting lines, then stage and commit the resolution.

## Responsible AI-assisted development

- Plan before coding; use small, scoped tasks with explicit constraints and verification steps.
- Give coding agents repository context through `AGENTS.md` and point them to relevant files.
- Create Git checkpoints before and after tasks.
- Review every diff, run tests, and do not merge code the team does not understand.
- Never paste or commit secrets/API keys.
- Acknowledge AI tools in the deliverables.
- Every team member remains responsible for the correctness of their commits and must be able to explain submitted code and results during Q&A.

## Definition of done

The project is ready only when all of the following are true:

- The problem, scope, and success criteria are explicit.
- The chosen project path and its central result/question are clear.
- Baselines or expectations and evaluation metrics are defined before final experimentation.
- Results are reproducible from documented commands in a clean environment.
- Quantitative and qualitative evidence are both included where appropriate.
- Failures, limitations, discrepancies, and iterations are honestly analyzed.
- Every member has a sustained trail of meaningful commits.
- The 10-minute presentation has been rehearsed and answers all 12 rubric criteria.
- Any system demo is stable, short, explained, and has a fallback recording/screenshots if feasible.
- The report satisfies the page, format, filename, section, and submission rules.
- The public repository satisfies structure, README, license, data-access, and reproducibility rules.
- Poster/report PDFs and repository link are submitted to the correct platforms by their deadlines.
- Every student completes attendance and two substantive peer reviews.

## Current unknowns to resolve later

- Final encoder subset after compute and dependency checks.
- Whether near-duplicate or source-video review reveals additional overlap
  beyond the five exact decoded-audio groups.
- Exact presentation day and slot.
- Available GPU/compute budget and maximum feasible fine-tuning scope.
- Instructor confirmation of how a non-reproduction project should label the report's "Chosen Result" section, if applicable.
- Final peer-review form and assignment details.
- Whether any later course announcement changes dates, submission links, or formatting rules.
