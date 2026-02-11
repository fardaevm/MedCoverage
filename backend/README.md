## Confidence Level Formula (Policy Clarity Score)

> **Purpose**  
> Quantify how clearly insurance policy rules apply to a given case.  
> This is **not a probability of payment** and **not user correctness**.

---

### Definitions

- `s` = semantic match score between user query and procedure (FAISS / MiniLM), `s ∈ [0,1]`
- `Q` = total number of eligibility questions
- `A` = number of answered questions
- `Y` = number of questions answered **YES** (rule triggered)
- `N` = number of questions answered **NO**
- `none_apply` = boolean flag (user explicitly says no rules apply)
- `decision` = `{covered, not_covered, uncertain}`

---

### Step 1 - Normalize semantic match

text
match_strength = clamp((s − 0.45) / (0.85 − 0.45), 0, 1)


### Step 2 - Answer strength
if Q == 0:
    answer_strength = 0.10
elif none_apply == true:
    answer_strength = 0.80
else:
    answer_strength = 0.55 + 0.45 × (Y / Q)


### Step 3 - Completeness
completeness = 0.35 + 0.65 × (A / max(1, Q))


### Step 4 - Contradiction Penalty
contradiction =
    0.35 if (none_apply == true and Y > 0)
    else 0

### Step 5 - Weighted Confidence score
confidence =
    0.15
  + 0.45 × match_strength
  + 0.30 × answer_strength
  + 0.10 × completeness
  − contradiction

confidence = clamp(confidence, 0, 1)

### Step 6 - If LLM is uncertain
if decision == "uncertain":
    confidence = min(confidence, 0.55)