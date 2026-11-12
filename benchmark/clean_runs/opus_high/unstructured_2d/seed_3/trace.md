# unstructured_2d, seed 3

Run `opus_high`, model claude-opus-5-5. Final log10 regret **-8.00** after 60 evaluations; 4 calls, $0.88.

Regret is the noise-free gap to the optimum of the best point so far (1e-8 counts as solved).

| Block | Call | Method | Script | Evaluations | log10 regret after |
|---|---|---|---|---|---|
| 1 | 1 | gpbo | [work/gpbo.py](scripts/001_gpbo.py) | 1–30 | -8.00 |
| 2 | 2 | explore | [work/gpbo_explore.py](scripts/002_explore.py) | 31–50 | -8.00 |
| 3 | 3 | localtr | [work/localtr.py](scripts/003_localtr.py) | 51–56 | -8.00 |
| 4 | 4 | gapfill | points | 57–60 | -8.00 |

## Call 1: block 1, gpbo (evaluations 1–30)

*7 turns, 63 s, $0.19*

<details><summary>2 tool calls</summary>

- `Bash: cat skill/METHOD.md && ls -la skill .agentoptimizer`
- `Bash: cd .agentoptimizer; cat PROBLEM.md problem.yaml space.json data.csv decisions.md; ls agent work; cat agent/*`

</details>

**Reasoning:**

> I have no data yet, and the setup is 2D with a budget of 60 evaluations and a deterministic objective. My plan is a space-filling initial design followed by GP-based Bayesian optimization. I'm leaning toward writing a GP-EI optimizer that builds its own Sobol initial design, so the whole thing runs as one block, and running about 30 points before I reassess.
>
> I'm settling on botorch's SingleTaskGP with LogEI, optimized through optimize_acqf, and I also want the state saved: the points, their values, which ones are still pending, and the RNG seed and count.

<details><summary>1 tool call</summary>

- `Bash: cat ../skill/SKILL.md | head -80; ls ../skill/references ../skill/scripts; cat ../skill/check.py | head -40`

</details>

**Reasoning:**

> I'm putting together a GP-based Bayesian optimizer: an initial Sobol design of 10 points by default, then LogEI acquisition over a GP with a Matérn 5/2 ARD kernel on standardized outputs. Since the objective may be multimodal, the GP could struggle, but that's acceptable for a first block of 30 points before I reassess.

<details><summary>1 tool call</summary>

- `Write: /work/.agentoptimizer/work/gpbo.py`

</details>

**Reasoning:**

> I'm checking the `_bo_point` logic: the pending points get appended to the told ones and then cut to the length of y, which looks right. The NaN handling in the JSON save also seems fine. There's an unused variable `k` that I should remove.

<details><summary>1 tool call</summary>

- `Bash: cd /work && sed -i '/ k = self.n_asked$/d' .agentoptimizer/work/gpbo.py && skill/aopt python skill/check.py .a…`

</details>

**Message to the user:**

Optimizer passes the offline checks. Writing the proposal and notes.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer && cat > agent/proposal.json <<'EOF' {"script": "work/gpbo.py", "label": "gpbo", "arg…`

</details>

**Message to the user:**

`aopt validate` accepted the first block, so I'm stopping here. The harness will commit it and take the measurements.

**Block 1 (`gpbo`), 30 of the 60 measurements:** there's no data yet and the objective's identity is withheld, so this choice is based only on the problem definition: two continuous variables in [-5, 5], no noise, one measurement at a time.
- **Start (10 points):** evenly spread points covering the whole box (a scrambled Sobol design, seed 3).
- **Next 20 points:** chosen one at a time by Bayesian optimization. A Gaussian-process model (Matern-5/2) is fitted to the results so far and picks the next point where it expects the most improvement (LogEI).
- **Checks:** the optimizer is in `work/gpbo.py`. It passed the skill's offline checks, including saving and reloading mid-run and starting from earlier data.

I'm using only half the budget so the next call can see the measured landscape first. If there's one smooth basin, I'll refine locally around the best point. If there are several good regions, I'll keep exploring. The reasoning is in `agent/decision.md` and my notes for the next call are in `agent/context.md`.

**Proposed:** **gpbo**: optimizer `work/gpbo.py` (seed=3, n_init=10), 30 points, state restart, not warm-started

**Rationale:** No data and no domain knowledge (objective identity withheld), so the decision rests only on problem.yaml: 2 continuous variables in [-5, 5], deterministic, cheap, sequential, 60 evaluations. Ran GP Bayesian optimization (`work/gpbo.py`): 10-point scrambled Sobol design (seed 3) to cover the box, then sequential LogEI on a Matern-5/2 ARD GP with standardized y. The block is 30 points (10 initial + 20 BO), half the budget, so I can look at the landscape (multimodality, how the GP fits, where the best points sit) before spending the remaining 30.

**Review:** Is the best value improving steadily, or is BO clustering? Are there several separated good basins (multimodal), or one smooth basin? If one basin, continue BO or switch to local refinement (BOBYQA/trust region) around the best point; if rugged, keep more exploration.

**Result:** log10 regret -8.00

## Call 2: block 2, explore (evaluations 31–50)

*10 turns, 101 s, $0.26*

<details><summary>3 tool calls</summary>

- `Bash: cat /work/skill/METHOD.md; ls /work/skill /work/.agentoptimizer`
- `Bash: cd /work/.agentoptimizer; cat agent/context.md; echo ----; tail -40 decisions.md; echo ----; cat PROBLEM.md pr…`
- `Bash: cd /work/.agentoptimizer; aopt() { ../skill/aopt "$@"; }; cat > work/show.py <<'EOF' from aoptcore import obse…`

</details>

**Reasoning:**

> I'm noting that this looks like a rugged, multimodal function with several basins sitting around -30.5. The best value so far, -32.56, seems almost like a near-optimum, so it could be a local minimum, and the landscape may hold many minima of varying depth.
>
> I'm weighing whether to keep going with the GP-based search, which struggles on this rugged landscape, or to switch to a mixed approach that refines each basin locally. The basins near -30.5 look flat and converged, but the pair around (0.93, 1.94) and (1.02, 1.49) shows slightly different values, so there may be an unexplored region with deeper minima.
>
> I'm trying to read the flat values near -30.5 as a broad plateau with narrow dips. The point at -32.56 looks like a broader well, since it sits a fair distance from the -32.41 point. I'll fit a GP to the data to check this, and I'm thinking about warm-starting it and adding local exploitation near the best point.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer; cat work/gpbo.py; cat blocks/001_gpbo.json`

</details>

**Reasoning:**

> I'm weighing the 6 points spent refining a -30.55 basin as wasteful, since the landscape looks rugged with many local basins and pure LogEI got stuck. I'm leaning toward adding exploration with about 20 points under an exploration-favoring acquisition, considering Thompson sampling, UCB with a higher beta, or a TuRBO-style trust region.
>
> I'm considering a block of about 20 points with an exploration-focused acquisition, such as LogEI with a radius constraint around evaluated points or Thompson sampling, before spending the last 10 refining the best basin. Next I'll fit the GP and check its lengthscales and predicted surface to see whether that plan holds up.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer; cat > work/fit.py <<'EOF' from aoptcore import observations import numpy as np, torc…`

</details>

**Reasoning:**

> The GP is smoothing over the rugged structure because the noise term is large, so it treats much of the real variation as noise. Nearby points show sharp differences, like the drop from about -11 to -33 over a short distance, which points to narrow wells that the model isn't capturing at this length scale.
>
> I'm weighing whether the search should mix fresh exploration with refining the best well, since 30 evaluations are left and the wells look about 0.3–0.5 wide, which a coarse spacing would miss. The best value so far is essentially converged, but I can't tell from here whether it's the true minimum or just a shallow bottom of one basin.
>
> I'm going to keep the search simple: a modified Bayesian optimizer with near-zero noise and a capped lengthscale so it explores more, plus a minimum spacing from existing points, with the last block reserved for polishing the best basin. I want to check whether expected improvement behaves sensibly on a noise-free GP before committing to that.

**Reasoning:**

> The landscape is rugged, with narrow wells near −30 that the block 1 GP mostly treated as noise, and BO then spent 6 points on one −30.55 well. I'm next running an exploration-weighted BO variant with a noise-free GP, a capped lengthscale, and a minimum distance from measured points.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer; python3 - <<'EOF' s=open('work/gpbo.py').read() s=s.replace('''"""GP Bayesian optimi…`

</details>

**Reasoning:**

> I'll run the proposed next step against real data and simulate the first request after a warm start, to see what it actually suggests.

<details><summary>2 tool calls</summary>

- `Bash: cd /work/.agentoptimizer; cat > work/try.py <<'EOF' import json, importlib.util, numpy as np from aoptcore imp…`
- `Bash: cd /work/.agentoptimizer; sed -i 's#"work/gpbo_explore.py"#".agentoptimizer/work/gpbo_explore.py"#; s#open("sp…`

</details>

**Reasoning:**

> I'm picking an exploration point in an unexplored region, away from the already-sampled points near 3 and 13, and I'll go with 20 points total.

**Reasoning:**

> The first proposal on the real data, (−4.705, 2.66), is in an unexplored area and looks right after the offline check passed. I'm writing the proposal, decision and context files next.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer; cat > agent/proposal.json <<'EOF' {"script": "work/gpbo_explore.py", "label": "explo…`

</details>

**Message to the user:**

I've set up block 2 to hunt for new wells: 20 points from an exploration-weighted BO variant, and `aopt validate` accepted it.

**What block 1 showed:** The function is rugged. It has narrow wells about 0.3–0.5 wide:
- Four separate wells bottom out around −30.5, at roughly (−0.09, −0.62), (0.93, 1.94), (1.02, 1.49) and (−1.07, 0.95).
- One deeper well holds the current best, **−32.56 at (1.047, −3.254)**.
- Points 0.3–0.5 away from these wells read −9 to −13.

The original GP smoothed the wells away, treating most of the variation as noise. That's why points 24–29 were all spent polishing one −30.55 well.

**Block 2:** The optimizer is `work/gpbo_explore.py`, started with all 30 earlier measurements:
- It assumes the measurements are essentially exact (noise fixed at 1e-4).
- Its lengthscales are capped at 0.15 of the box width, so narrow wells aren't smoothed away.
- It won't place a point within 0.3 (user units) of one already measured, so it searches for new, deeper wells instead of polishing known ones.

It passed the offline check. On the real data its first proposal is (−4.705, 2.66), an unexplored area next to two good points.

**Last 10 points:** These are kept for local refinement around the best well or two after this block.

I updated `agent/context.md` with the landscape notes and this plan.

**Proposed:** **explore**: optimizer `work/gpbo_explore.py` (seed=7, n_init=0, ls_max=0.15, noise=0.0001, min_dist=0.03), 20 points, state restart, warm-started on all data

**Rationale:** Block 1 shows a rugged, multi-well landscape: narrow wells about 0.3-0.5 wide reach about -30.5 at several separated places ((-0.09,-0.62), (0.93,1.94), (1.02,1.49), (-1.07,0.95)), and a deeper one reaches -32.56 at (1.05,-3.25). Points 0.3-0.5 away from these wells read -9 to -13. The default GP fitted a large noise (0.54 standardized) and lengthscales of about 0.4 (unit cube), so it smoothed the wells away. LogEI then spent points 24-29 polishing one -30.55 well. No domain knowledge (identity withheld). New optimizer `work/gpbo_explore.py`: LogEI on a Matern-5/2 GP with noise fixed at 1e-4, lengthscales capped at 0.15 (unit cube), and candidates within 0.03 unit (0.3 in user units) of any measured point rejected, so it hunts for new and deeper wells instead of polishing known ones. It is warm-started with all 30 points (its first proposal offline: (-4.705, 2.66)). The block is 20 points, leaving 10 for local refinement of the best well(s) at the end.

**Review:** Did any new well beat -32.56? Identify the 1-3 best wells and spend the last 10 on local refinement (small listed pattern or BOBYQA from the best), allowing points within 0.3 of the best.

**Result:** log10 regret -8.00 → -8.00

## Call 3: block 3, localtr (evaluations 51–56)

*11 turns, 76 s, $0.25*

<details><summary>2 tool calls</summary>

- `Bash: cat /work/skill/METHOD.md; ls /work/skill /work/.agentoptimizer`
- `Bash: cd /work/.agentoptimizer; cat agent/context.md; tail -60 decisions.md; cat PROBLEM.md problem.yaml; ls work bl…`

</details>

**Reasoning:**

> I'm checking the measurement: Block 2 came in at 20 points, so I'll look at the data behind that number.

<details><summary>1 tool call</summary>

- `Bash: cd /work; cat .agentoptimizer/work/show.py; skill/aopt python .agentoptimizer/work/show.py | sort -t, -k1 -g |…`

</details>

**Reasoning:**

> I'm comparing the best optimum near (1.05, -3.25) at -32.56 against a second region around (-4.5, 3.4). Nearby points along that line of x0 near 1 all sit close in value, which suggests a ridge-like valley running along x1.
>
> I'm checking the minimum near (1.0, -3.4), where the values sit around -32.4 to -32.56 and the valley looks elongated along x1. The x0 direction is narrow, since moving 0.36 away drops it to about -11, so I'm weighing a local quadratic fit against spending a few probes along x0 at x1 = -3.4.
>
> I'm weighing a sequential local optimizer: a trust-region quadratic model built from the points within some radius of the best well, falling back to a coordinate pattern when there aren't enough points to fit one. I'm also wondering whether the second well at -31.65 might be deeper than it looks, but I'd rather not guess at its shape. With 10 evaluations left, I'm leaning toward spending most of them on the best well and only a few on the second.
>
> I'm settling on a plan: a trust-region local optimizer around the best well, running for about six evaluations before I check where things stand. It will fit a quadratic to nearby points, expand or shrink the region depending on whether each step improves, and keep new probes at least 0.01 away from existing ones.
>
> I'm checking which of the nearby points fall within 0.6 of the best well, and it looks like only four do, not enough to fit a full quadratic. So I'm planning to probe along x0 at ±0.1 around the best point first, to fill in the missing local data before building the model.
>
> I'm settling on a quadratic trust-region model over the nearby points, with a fit radius of 0.5 and weights applied to the fit.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer; cat work/gpbo_explore.py | head -60; cat space.json`

</details>

**Reasoning:**

> I'm going to create the file work/localtr.py.

<details><summary>1 tool call</summary>

- `Write: /work/.agentoptimizer/work/localtr.py`

</details>

**Reasoning:**

> I'm fixing the ask method so it returns a single proposal, and I'm noting that shrinking delta on failed axis probes is a questionable side effect, though tolerable. Next I'll apply the ask fix and preview the result with warm data.

<details><summary>2 tool calls</summary>

- `Bash: cd /work/.agentoptimizer; python3 - <<'E' p='work/localtr.py'; s=open(p).read() s=s.replace("return [self._pro…`
- `Bash: cd /work/.agentoptimizer; sed -i 's/model = lambda s: float(_feats(s) @ coef)/model = lambda s: float((_feats(…`

</details>

**Reasoning:**

> I'm weighing whether to simulate the sequence, but I can't evaluate it directly, so I'll plan around the probes instead. The idea is to spend fewer probes: two along the axis should bring the near count to six before fitting a quadratic. I'm also concerned that shrinking the step after a failed probe halves the delta to 0.05, which might be too aggressive.
>
> I'm considering whether the second probe's step of 0.05 is too small to be useful, though it should still resolve the curvature more finely. I also want to check that the width around x0 is reasonable before I write up the proposal and decision.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer; cat > agent/proposal.json <<'E' {"script": "work/localtr.py", "label": "localtr", "a…`

</details>

**Reasoning:**

> I'm noticing the docstring describes a `center` argument that the function never actually implements, which is misleading even though it's minor. The file-changed concern turns out to be fine, so I'll just correct the docstring since that edit is harmless.

<details><summary>2 tool calls</summary>

- `Bash: cd /work/.agentoptimizer; sed -i 's/^Centre = best point told so far (its coordinates in args 'center' if give…`
- `Bash: cd /work/.agentoptimizer; sed -i '3,4c Centre = best point told so far. Points within fit_radius of the centre…`

</details>

**Reasoning:**

> I'm checking the context where the `center` argument comes up, since the line about it not being implemented needs correcting. The statement itself is accurate, so there's nothing else to change there.

**Message to the user:**

The next block is set up and `aopt validate` accepted it: 6 points using a new local optimizer, which leaves 4 measurements for the final call.

**Where things stand (50 of 60 used):**
- **Best so far:** -32.56 at (1.047, -3.254). The 20 exploratory points in block 2 found nothing deeper.
- **Second-best region:** a separate well near (-4.5, 3.4), reading down to -31.65.
- **Shape of the best well:** it's long and thin. Points with x0 between 0.94 and 1.08 all read between -30.0 and -32.56 across x1 from -3.73 to -2.92, but moving x0 to 0.36 gives about -11. So the true bottom is probably close to the best point but not exactly on it.

**Next block (label `localtr`, `work/localtr.py`):**
- It's started from all 50 earlier measurements and centres a small search on the best point.
- It fits a quadratic surface to the measured points within 0.6 of the best one. Only 4 points are that close now, so it first measures nearby test points to get enough for the fit; the first is (1.147, -3.254).
- After that, each point goes to the lowest point the fitted surface predicts within a search radius of 0.1. The radius grows after an improvement and shrinks when a point is no better.
- The optimizer passed `check.py` and a preview run on the real data.

**For the last 4 measurements:** if this block is still improving, keep running the same optimizer. If it has stopped improving, I'll spend some points refining the second well instead. I wrote this plan down in `agent/context.md`.

**Proposed:** **localtr**: optimizer `work/localtr.py` (delta=0.1, fit_radius=0.6, min_sep=0.01), 6 points, state restart, warm-started on all data

**Rationale:** Block 2 (exploratory GP-BO) found no well deeper than the best, -32.56 at (1.047,-3.254); the runner-up is a second well near (-4.5,3.4) reading -31.65 (pts 32, 36, 40, 42 all -30.9..-31.7). The best well looks elongated along x1 (x0~0.94-1.08 reads -30.0..-32.56 for x1 from -3.73 to -2.92) and narrow in x0 (x0=0.36 reads -11), so its true minimum is probably near but not at the best point. With 10 left, exploration is over: refine. New optimizer `work/localtr.py`: a quadratic-model trust region centred on the best point, warm-started with all 50 points; it fits a weighted full quadratic to points within 0.6 of the best (only 4 now, so it first asks axis probes at +/-delta, first one (1.147,-3.254)), then asks the model minimum inside a trust region of radius delta=0.1 (x1.5 on improvement, x0.5 otherwise). No domain knowledge (identity withheld). Block of 6 so the last 4 can be re-decided (continue here, or spend some on the second well).

**Review:** Did the well's minimum move below -32.56, and is the quadratic consistent (is the step shrinking)? If converged, consider continuing `localtr` (state continue) for the last 4, or a few points of local refinement in the (-4.5,3.4) well if the first well is exhausted.

**Result:** log10 regret -8.00 → -8.00

## Call 4: block 4, gapfill (evaluations 57–60)

*9 turns, 49 s, $0.19*

<details><summary>2 tool calls</summary>

- `Bash: cat /work/skill/METHOD.md; ls /work/skill /work/.agentoptimizer`
- `Bash: cd /work/.agentoptimizer; cat agent/context.md; tail -40 decisions.md; cat PROBLEM.md problem.yaml; ls work; c…`

</details>

**Reasoning:**

> Localtr didn't turn up anything better, so I'm moving on to look at the block 3 data.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer; cat work/show.py | head -30; cd ..; skill/aopt python .agentoptimizer/work/show.py 2…`

</details>

**Reasoning:**

> I'm checking the best value's extra digits, which show it's really a flat plateau at about -32.56. Nearby points like (0.94, -3.59) and (0.945, -2.915) come out higher, so the region looks like a well, and I'm weighing whether the flat part is a true minimum.
>
> I'm asking whether the flat level around -32.56 really is the bottom, and I want to print the values at full precision to check.

<details><summary>1 tool call</summary>

- `Bash: cd /work; cat > .agentoptimizer/work/show2.py <<'EOF' from aoptcore import observations import numpy as np X,y…`

</details>

**Reasoning:**

> I'm treating the -32.56 well as a fixed floor, so any further improvement there is at most about 1e-8. The other wells look like they bottom out at their own fixed depths too. I'm wondering whether the second well near (-4.5, 3.4), sitting at -31.65, might dip lower than -32.56, and I'm starting to look at its points.
>
> I'm checking whether the second well's floor might sit below -32.56, using its four sampled points around (-4.6, 3.3). The first well's behavior suggests the second bottoms out near -31.7, so it probably won't beat the first by much. I'm weighing whether to spend the remaining evaluations on a quadratic fit of that well or on exploring elsewhere.
>
> I'm going to fit a quadratic to the six points in the second well, since that should pin down its minimum more precisely than the four points I have so far.

<details><summary>1 tool call</summary>

- `Bash: cd /work; cat > .agentoptimizer/work/well2.py <<'EOF' from aoptcore import observations import numpy as np X,y…`

</details>

**Reasoning:**

> I'm checking whether a deeper well could exist elsewhere, since a predicted value near -31.85 is unlikely to beat -32.56. I want to see which regions of the box are still unexplored, because with only four points left, probing a large gap is the only realistic path to improvement, while refining well 2 won't help.
>
> I'm weighing whether the remaining four points are better spent on exploring the largest uncovered gaps rather than refining well 2, since its fitted floor sits around -31.7 and can't beat -32.56. Exploration looks like the better use of the budget, so I'm moving toward computing the maximin gap points.

**Reasoning:**

> Block 3 brought the best well to its floor, with six points near (1.05, -3.25) all reading -32.56, so only a deeper unexplored well can beat it. I'm now searching for the largest unsampled gaps.

<details><summary>1 tool call</summary>

- `Bash: cd /work; cat > .agentoptimizer/work/gaps.py <<'EOF' from aoptcore import observations import numpy as np X,y,…`

</details>

**Reasoning:**

> I'm looking at the gaps, which are largest on the right half where x0 is above 2. The wells are narrow, so random points in those gaps rarely hit, but I don't have a better option. Corners look weak, so I'm considering moving the corner points inward toward the interior gap centers, and I'm checking whether the wells near the edges around (-4.5, 3.4) change the search range I should use.
>
> I'll settle on the box at [-4.5, 4.5] and pick four maximin points one at a time inside it.

<details><summary>1 tool call</summary>

- `Bash: cd /work; sed -i 's/linspace(-5,5,201)/linspace(-4.5,4.5,181)/; s/range(6)/range(4)/' .agentoptimizer/work/gap…`

</details>

**Reasoning:**

> I'm going with the four listed points as the basis, and next I'll write out the files.

<details><summary>1 tool call</summary>

- `Bash: cd /work/.agentoptimizer; cat > agent/proposal.json <<'EOF' {"label": "gapfill", "points": [{"x0": 4.5, "x1": …`

</details>

**Message to the user:**

I've set up the last block: 4 exploration points, labelled `gapfill`. `aopt validate` accepted it, so I'm stopping here for the harness to commit and measure it.

- **Best result so far:** -32.56 at (1.047, -3.254). The refinement block put six points near there and all read -32.5600, so that low area is fully refined. More points there could improve the result by at most about 1e-8.
- **Second-best area:** near (-4.6, 3.3). A curve fit to the points there predicts its lowest value is about -31.85, which would not beat -32.56, so I'm not spending points on it.
- **Final 4 points:** placed in the largest unsampled parts of the box: (4.5, -4.5), (-4.5, -4.15), (3.35, -2.5) and (4.5, 4.5). The low areas found so far are narrow, so the chance of finding a deeper one is small. Still, it's the only use of these points that could improve the final result.

This block uses the whole budget. The final result will be the lower of -32.56 and these four measurements.

**Proposed:** **gapfill**: a list of 4 points chosen by the agent

**Rationale:** Block 3 (localtr) converged the best well: six points near (1.05,-3.25) all read -32.5600 to within 1e-6 (best -32.559999992), so that well's floor is about -32.56 and further refinement can gain at most ~1e-8. A weighted quadratic fit to the second well near (-4.6,3.3) (points -31.65, -31.58, -31.41, -30.89) puts its minimum at about (-4.42,3.21) with predicted value about -31.85, which would not beat -32.56. So the only route to a better final value is a deeper, not yet found well. I listed 4 points: the sequential maximin gap centres of the unsampled area within [-4.5,4.5]^2 (work/gaps.py): (4.5,-4.5), (-4.5,-4.15), (3.35,-2.5), (4.5,4.5). Wells are narrow, so the chance of a hit is small, but it is the only option with positive expected gain. No domain knowledge was used (identity withheld).

**Review:** Campaign ends after this block; final best is the minimum of -32.56 and these 4 values.

**Result:** log10 regret -8.00 → -8.00
