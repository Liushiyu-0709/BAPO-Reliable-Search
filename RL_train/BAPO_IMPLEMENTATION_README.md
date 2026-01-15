# BAPO Code Implementation Details

The modifications upon standard GRPO code implementation based on verl framework are mainly concentrated in two aspects:
- **Reward Scoring Component**: Enhanced reward computation with correctness-based and boundary-aware scoring
- **Resampling Rollout Component**: Adaptive oversample mechanism

## Reward Scoring Component

BAPO's reward computation consists of both correctness-based reward (including format correctness and outcome correctness) and boundary-aware reward.

### Scoring Chain Architecture

The BAPO reward scoring follows this hierarchical chain:

```
workers/reward_manager/re_search_faithful.py:
ReSearchRewardManagerWithSaveFaithful.__call__()
    ↓
utils/reward_score/__init__.py:
faithful_compute_score()
    ↓
utils/reward_score/re_search_faith.py:
compute_score() - Main scoring logic with BAPO rewards
```

### Correctness-based Reward Scoring (`compute_score_single`)

Each response is evaluated based on:

- **Format Validation**: Checks for proper XML tags (`<think>`, `<answer>`, `<search>`, `<result>`, `<python>`)
- **Answer Extraction**: Extracts content from `<answer>` tags and `\boxed{}` format
- **F1 Score Calculation**: Computes semantic similarity with ground truth

**Scoring Rules:**
- `-1`: Format errors or extraction failures
- `0`: Wrong answer but correct format
- `F1 score`: Correct answer (0-1 range)

### Boundary-aware Reward Scoring (`compute_score`)

After computing the correctness-based reward scores, boundary-aware rewards are calculated based on the correctness situation.

The basic boundary-aware reward only penalizes IDK (I Don't Know) responses in groups without correct answers:

```python
for group_start in range(0, len(scores), rollout_size):
        group_end = group_start + rollout_size
        group_ground_truths = ground_truth_ls[group_start:group_end]
        normalized_ground_truths = [
            " ".join(gt) if isinstance(gt, list) else gt
            for gt in group_ground_truths
        ]

        # Check if all scores in this group are <= 0
        group_scores = scores[group_start:group_end]
        # if no correct answers exist in this group, encourage IDK
        if all(score <= 0 for score in group_scores):
            for j in range(group_start, group_end):
                if is_rejection_ls[j]:  # If this is a rejection case
                    if scores[j] != -1: # If the format is correct (reward correct "I don't know" format)
                        scores[j] += 0.5   # Give a small positive reward
                        reasons[j] = "i don't know should be encouraged"
        # if any correct answer exists in this group, penalize IDK
        if any(score > 0 for score in group_scores):
            max_score = max(group_scores)
            for j in range(group_start, group_end):
                if is_rejection_ls[j]:  # If this is a rejection case
                    if scores[j] != -1:
                        scores[j] = 0
                        reasons[j] = "i don't know should be penalized"
```

Furthermore, we implement an adaptive modulator that monitors validation scores to distinguish between exploration stage and plateau stage in the `ReSearchRewardManagerWithSaveFaithful` class:

```python
if step > 1 and self.early_stage: # check if the stage transition happens in train stage
    validation_file = os.path.join(os.path.dirname(curr_save_path), f"val_{step-1}.jsonl")
    curr_score = cal_validation_score(validation_file)
    self.max_val_score = max(curr_score, self.max_val_score)
    if curr_score < self.max_val_score:
        self.tolerance_step += 1
        if self.tolerance_step > 5:
            self.early_stage = False
```

In the early stage, we skip IDK rewards if IDK rate ≥ threshold (default 0.1) to prevent over-encouragement of admitting ignorance.

In the later stage, we maintain answer consistency to detect if the model is exploring this sample's solutions or has plateaued to a fixed solution.

The corresponding implementation is in `re_search_faith.compute_score`.

## Resampling Rollout Component

### Core Logic

The oversample function (`generate_sequences_oversample`) works with a simple principle:

**If a group of samples all get zero scores but contain no "I don't know" responses, resample that group.**

### Implementation Details

#### Main Flow

```python
def generate_sequences_oversample(self, prompts, ground_truth_ls, max_resample_attempts=2):
    # 1. Generate initial samples
    current_results = self.generate_sequences(prompts)

    # 2. Skip resampling for validation data
    if prompts.meta_info.get('validate', False):
        return current_results

    # 3. Iterative resampling loop
    for attempt in range(max_resample_attempts):
        # Score all samples
        scores = []
        is_rejection_ls = []
        for i in range(batch_size):
            score, _ = compute_score_single(tokenizer, solution_str, ground_truth)
            scores.append(score)
            is_rejection_ls.append(is_rejection(solution_str))

        # Check each group for resampling criteria
        groups_to_resample = []
        for group_idx in range(num_groups):
            group_scores = scores[group_start:group_end]
            group_rejections = is_rejection_ls[group_start:group_end]

            # Resampling condition: all zero scores + no IDK responses
            all_zero_scores = all(score <= 0 for score in group_scores)
            no_rejection = not any(group_rejections)

            if all_zero_scores and no_rejection:
                groups_to_resample.append(group_idx)

        # If no groups need resampling, stop
        if not groups_to_resample:
            break

        # Resample the problematic groups
        resample_results = self.generate_sequences(resample_prompts)
        # Replace old results with new ones

    return current_results
```

#### Resampling Criteria

A group gets resampled if **ALL** conditions are met:
1. All samples in the group have score ≤ 0
2. No samples contain "I don't know" responses
3. Haven't reached maximum resample attempts

#### Group Structure

- Samples are organized in groups (typically 8 samples per group in GRPO)
- Each group shares the same prompt but generates different responses
- Resampling happens at the group level, not individual samples

### Configuration Parameters

#### Key Parameters

- `max_resample_attempts`: Maximum resampling rounds (default: 2)
- `rollout_size`: Samples per group (typically 8)
- `validate`: If True, skip oversample (for validation data)

#### Training Configuration

```python
# In your training config
oversample_faith = True  # Enable oversample
oversample_attempts = 2  # resample attempts
```

#### Code Entry Point

The decision between normal generation and oversample happens in the FSDP worker (`fsdp_workers.py`):

```python
def generate_sequences(self, prompts):
    with self.rollout_sharding_manager:
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)

        if self.config.rollout.oversample_faith:  # Check if oversample is enabled
            # Extract ground truth for scoring
            ground_truth_ls = [data.non_tensor_batch['reward_model']['ground_truth'] for data in prompts]
            ground_truth_ls = self.rollout_sharding_manager.allgather_list(ground_truth_ls)

            # Call oversample function
            output = self.rollout.generate_sequences_oversample(
                prompts=prompts,
                ground_truth_ls=ground_truth_ls,
                max_resample_attempts=self.config.rollout.oversample_attempts
            )
        else:
            # Use normal generation
            output = self.rollout.generate_sequences(prompts=prompts)

        output = self.rollout_sharding_manager.postprocess_data(output)
        return output
```


These components enable BAPO to train reliable agentic search models that know when to admit ignorance while maintaining high accuracy on answerable questions.