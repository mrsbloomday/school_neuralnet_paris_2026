# %% [markdown]
# # 04 · Population activity and dimensionality reduction — **Python / pynapple**
#
# So far we looked at one neuron (notebooks 01 and 02) or one pair of neurons (notebook 03) at a
# time. But we record ~200 prefrontal units at once. At each moment, the population activity is a
# point in a space with **one axis per neuron**: ~200 dimensions, impossible to look at. Yet the
# neurons are not independent: many of them go up and down together. **Dimensionality reduction**
# finds a few directions (combinations of neurons) that capture most of this shared activity.
#
# This notebook is a short tour of four methods, with what each of them is good for:
#
# 1. **PCA** (principal component analysis) on the trial-averaged activity: the main patterns of
#    the population response, and how it changes from block to block;
# 2. **TCA** (tensor component analysis) on the single-trial activity: patterns across neurons,
#    **time within a trial**, and **trials** at once;
# 3. **UMAP** and a **self-organizing map (SOM)**: non-linear maps of the units themselves,
#    described by the firing features of notebooks 01 and 02 (rate, burstiness, memory, waveform).
#
# **Dataset.** Session `156131_20191112-probe0`, the same as in notebooks 01–03, with the trials of
# blocks 1–4 (see the trial structure in notebook 01, section 1) and the units of the four
# prefrontal areas (ACA, PL, ILA, ORB).
#
# ## 0. Get the data
#
# The session is on the [DANDI Archive](https://dandiarchive.org), in
# [Dandiset 000473](https://dandiarchive.org/dandiset/000473): we use the file
# `sub-156131/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb` (5.6 GB), the same as in
# notebook 01. If you already downloaded it, just write its path in `NWB` below.
#
# **Option 1 (recommended): download the file** to a folder of your choice on your computer
# (keep it out of the git repo: it is far too big for GitHub). Either:
#
# - in the browser: open the
#   [files of the Dandiset](https://dandiarchive.org/dandiset/000473/0.261102.1657/files?location=sub-156131),
#   and click the download icon next to the file;
# - or in a terminal, from the `python/` folder, with the DANDI command-line tool
#   (replace `/path/to/your/data/folder` by the folder where you want to save the file):
#
#   ```bash
#   uv run dandi download -o /path/to/your/data/folder "https://api.dandiarchive.org/api/assets/6b424b64-764a-4677-a828-49094420e930/download/"
#   ```
#
# Then write the full path to the file in `NWB` in the cell below.
#
# **Option 2 (advanced): stream the file from DANDI**, without downloading it: set `STREAM = True`
# below (see section 0 of notebook 01).

# %%
# numpy: arrays and maths; matplotlib: figures; pandas: tables; pynapple: spike trains;
# h5py: low-level access to the NWB file; scikit-learn: PCA; tensorly: TCA; umap: UMAP;
# minisom: self-organizing maps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pynapple as nap
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import umap
from minisom import MiniSom

# Option 1: full path to the NWB file you downloaded. Change it to where you saved it.
NWB = "/path/to/your/data/folder/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb"

# Option 2: set to True to stream the file from the DANDI Archive instead
STREAM = False

if not STREAM and not Path(NWB).is_file():
    raise FileNotFoundError(f"No NWB file at {NWB}: set NWB to the downloaded file, or STREAM = True")

if STREAM:
    import remfile
    from dandi.dandiapi import DandiAPIClient

    DANDISET, VERSION = "000473", "0.261102.1657"  # Dandiset id and published version
    ASSET = "sub-156131/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb"
    with DandiAPIClient() as client:
        asset = client.get_dandiset(DANDISET, VERSION).get_asset_by_path(ASSET)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
    print("streaming from", url)


def open_nwb():
    """Open the NWB file with h5py: the local file, or the remote one if STREAM is True."""
    return h5py.File(remfile.File(url) if STREAM else NWB, "r")

# %% [markdown]
# ## 1. Load the units and the trials
#
# Same steps as in notebook 01 (sections 1 and 2): the trials of blocks 1–4, and the units of
# ACA, PL, ILA and ORB that pass the quality criteria. We also keep the trough-to-peak duration
# of the waveform (`Peak_to_Trough`, read by pynapple) to classify the units into **narrow** and
# **wide** waveforms with the two thresholds of notebook 02 (section 3.1).

# %%
if STREAM:
    import pynwb
    data = nap.NWBFile(pynwb.NWBHDF5IO(file=open_nwb()).read())  # remote file
else:
    data = nap.load_file(NWB)  # open the file and list its content

units = data["units"]  # all sorted units (TsGroup)
trials = data["trials"]  # task trials (IntervalSet)
trials = trials[trials.Block <= 4]  # blocks 1-4

with open_nwb() as f:
    isi_fp = f["units/quality"][:, 1]  # ISI violation false-positive rate
    fs = f["units/FS"][:] == 1  # fast-spiking label of the file
    rs = f["units/RS"][:] == 1  # regular-spiking label of the file

presence_ratio = (units.count(units.time_support.tot_length() / 100).values > 0).mean(axis=0)
area = units.location.str.extract(r"^(ACA|PL|ILA|ORB)", expand=False)  # NaN outside the 4 areas
units.set_info(isi_fp=isi_fp, plausible=fs | rs, presence_ratio=presence_ratio, area=area)
pfc = units[units.area.notna() & (units.isi_fp < 0.1) & units.plausible & (units.presence_ratio > 0)]

low_th, high_th = 0.380, 0.430  # ms, the thresholds of notebook 02
pfc.set_info(wf_class=np.select([pfc.Peak_to_Trough < low_th, pfc.Peak_to_Trough > high_th],
                                ["narrow", "wide"], default="unclassified"))

block = trials.Block.astype(int)  # block of each trial: 1, 2, 3 or 4
block_names = {1: "sound (pre)", 2: "sound + opto", 3: "sound (post)", 4: "air puff"}
block_colors = {1: "tab:blue", 2: "tab:purple", 3: "tab:cyan", 4: "tab:orange"}
area_colors = {"ACA": "C0", "PL": "C1", "ILA": "C2", "ORB": "C4"}
print("trials:", len(trials), "| PFC units:", len(pfc))
print(pfc.area.value_counts().to_string())

# %% [markdown]
# ## 2. The population tensor
#
# For each unit and each trial, we count the spikes in 100 ms bins, from 1 s before to 2 s after
# the trial onset. pynapple does it in one call with `nap.build_tensor`: it returns a 3D array, a
# **tensor**, of shape (units × trials × time bins). We divide by the bin size to get rates (Hz),
# and smooth each trial in time with a Gaussian kernel (σ = 1 bin), to reduce the noise of single
# trials.

# %%
bin_size = 0.1  # s
window = nap.IntervalSet(start=trials.start - 1, end=trials.start + 2)  # -1 to +2 s around onset
rates = nap.build_tensor(pfc, window, bin_size=bin_size) / bin_size  # (units, trials, bins), in Hz
rates = gaussian_filter1d(rates, sigma=1, axis=2)  # smooth along time
t_bins = np.arange(rates.shape[2]) * bin_size - 1 + bin_size / 2  # center of each bin, in s
print("tensor (units x trials x bins):", rates.shape)

# %% [markdown]
# Averaging over the trials of each block gives one PSTH per unit and per block. We put the 4
# blocks side by side, and **z-score** each unit (subtract its mean, divide by its standard
# deviation): otherwise the fast units would dominate everything that follows. Below, one row per
# unit, sorted by its response to the pre-conditioning sound.

# %%
psth = np.concatenate([rates[:, block == b].mean(axis=1) for b in block_names], axis=1)  # units x (4 x bins)
z = (psth - psth.mean(axis=1, keepdims=True)) / psth.std(axis=1, keepdims=True)
n_bins = len(t_bins)
order = np.argsort(z[:, :n_bins][:, (t_bins > 0) & (t_bins < 0.5)].mean(axis=1))  # response 0-0.5 s, block 1

fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(z[order], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="none")
onset = np.searchsorted(t_bins, 0)  # first bin after the trial onset
for k, b in enumerate(block_names):
    ax.axvline(k * n_bins - 0.5, color="k", lw=3)  # start of each block
    ax.axvline(k * n_bins + onset - 0.5, color="k", ls=":", lw=1)  # trial onset in each block
    ax.text(k * n_bins + n_bins / 2, -2, block_names[b], ha="center")
ax.set(xticks=[], xlabel="Time: -1 to 2 s around the trial onset (dotted line), in each block",
       ylabel="Unit (sorted)")
plt.colorbar(im, label="z-scored rate")
plt.show()

# %% [markdown]
# Many units are excited just after the trial onset (red after the dotted line), others are
# inhibited (blue). The pattern is similar in the four blocks, but not identical: in blocks 2 and 4,
# a second wave of activity comes about 1 s after the onset.
#
# ## 3. PCA: the main patterns of the population
#
# ### 3.1 Principal components
#
# PCA looks for the direction in the 192-dimensional space (a weighted sum of the units) along
# which the data vary the most: the first **principal component** (PC1). Then the direction, at
# right angles to PC1, with the most remaining variance: PC2, and so on. Each PC has:
#
# - **loadings** (or weights): one value per unit, how much the unit contributes;
# - a **score** (or projection): one value per time point, the activity of the population
#   along this direction.
#
# Here the "samples" are the 4 × 30 time bins and the "features" are the units.

# %%
pca = PCA()
scores = pca.fit_transform(z.T)  # (4 x bins) x components: one row per time bin
explained = pca.explained_variance_ratio_  # fraction of the variance of each PC
n80 = np.searchsorted(np.cumsum(explained), 0.8) + 1  # number of PCs for 80 % of the variance

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(1, 21), 100 * explained[:20], "ko-", label="each PC")
ax.plot(np.arange(1, 21), 100 * np.cumsum(explained[:20]), "o-", color="gray", label="cumulative")
ax.axhline(80, color="gray", ls=":")
ax.set(xlabel="Principal component", ylabel="Variance explained (%)", title=f"{n80} PCs explain 80 %")
ax.legend()
plt.show()

# %% [markdown]
# A few PCs explain most of the variance of 192 units: the trial-averaged activity is
# **low-dimensional**. (Careful: averaging over trials already removed most of the noise, which
# makes the data look more low-dimensional than single trials.)
#
# ### 3.2 What do the PCs look like?
#
# We plot the scores of the first 3 PCs against time, one curve per block, and the population
# **trajectory** in the plane of PC1 and PC2: each point is the state of the population at one
# moment, and the trajectory goes from 1 s before to 2 s after the onset.

# %%
scores_blocks = scores[:, :3].reshape(4, n_bins, 3)  # block x time bin x PC

fig, axes = plt.subplots(1, 4, figsize=(19, 4))
for k, b in enumerate(block_names):
    for pc in range(3):
        axes[pc].plot(t_bins, scores_blocks[k, :, pc], color=block_colors[b], lw=2, label=block_names[b])
    axes[3].plot(scores_blocks[k, :, 0], scores_blocks[k, :, 1], color=block_colors[b], lw=2)
    axes[3].plot(*scores_blocks[k, 0, :2], "o", color=block_colors[b])  # start: 1 s before onset
    axes[3].plot(*scores_blocks[k, onset, :2], "k*", ms=10)  # trial onset
for pc in range(3):
    axes[pc].axvline(0, color="k", ls=":")
    axes[pc].set(xlabel="Time from trial onset (s)", ylabel="Score",
                 title=f"PC{pc + 1} ({100 * explained[pc]:.0f} %)")
axes[0].legend(fontsize=8)
axes[3].set(xlabel="PC1", ylabel="PC2", title="Trajectories (dot: -1 s, star: onset)")
plt.tight_layout()
plt.show()

# %% [markdown]
# - **PC1** is the response common to all the blocks: the population jumps right after the trial
#   onset. In blocks 2 (sound + opto) and 4 (air puff) it jumps a second time, about 1 s after the
#   onset: the response to the aversive stimulus that follows the sound.
# - **PC2** is different in each block **already before the onset**, when nothing has happened
#   yet: it changes steadily from block to block. This is not a response, but a slow change of the
#   activity over the session (the blocks follow each other in time): a change of the state of
#   the mouse, or a **drift** of the recording. We meet it again in notebook 05.
# - In the PC1–PC2 plane, each block makes a loop that starts and ends near its own point (the
#   activity before the trial): the loops are shifted from block to block along PC2.
#
# ### 3.3 Which units make the PCs?
#
# The loadings tell which units contribute to each PC. Do the 4 areas contribute in the same way?

# %%
loadings = pd.DataFrame(pca.components_[:3].T, index=pfc.index, columns=["PC1", "PC2", "PC3"])
loadings["area"] = pfc.area.values

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, pc in zip(axes, ["PC1", "PC2", "PC3"]):
    for a, color in area_colors.items():
        ax.hist(loadings[pc][loadings.area == a], bins=np.linspace(-0.3, 0.3, 31), histtype="step",
                lw=2, color=color, density=True, label=a)
    ax.set(xlabel=f"Loading on {pc}", ylabel="Density")
axes[0].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The loadings of PC1 mostly have the same sign: most units go up together after the onset. (The
# sign of a PC is arbitrary: flipping both its loadings and its scores gives the same PC.) The
# loadings of the other PCs are spread around 0 in all the areas: every PC mixes units of the four
# areas, with positive and negative weights. A PC is a pattern of the **population**, not of an
# area.
#
# **Questions.**
#
# - Run the PCA without z-scoring (`psth - psth.mean(...)` only). Which units dominate PC1 now?
# - Run the PCA on the units of one area only. Are the first PCs the same?
# - Fit the PCA on blocks 1 and 4 only, and project block 3 on these PCs (`pca.transform`).
#
# ## 4. TCA: neurons, time and trials at once
#
# PCA needed the average over trials: it cannot see that the response changes **from trial to
# trial** (with learning, habituation, drift...). Tensor component analysis
# ([Williams et al., 2018](https://doi.org/10.1016/j.neuron.2018.05.015)) works on the whole
# tensor (units × time × trials). It writes the tensor as a sum of $R$ components, each the
# product of three vectors:
#
# $$ x_{n,t,k} \approx \sum_{r=1}^{R} a_r(n)\, b_r(t)\, c_r(k) $$
#
# - $a_r$: the **neuron factor**, which units take part in component $r$;
# - $b_r$: the **time factor**, its time course within a trial;
# - $c_r$: the **trial factor**, its strength in each trial.
#
# We use the **non-negative** version (all the factors ≥ 0, like the rates): the components then
# add up, and are easier to read. As Williams et al., we divide each unit by its maximum, so that
# all the units count. `tensorly` fits the model, starting from random factors: we fit it a few
# times (`n_fits`) with different random starts (`seed`), and keep the best fit.

# %%
tensor = np.transpose(rates, (0, 2, 1))  # units x time x trials
tensor = tensor / tensor.max(axis=(1, 2), keepdims=True)  # each unit divided by its maximum


def fit_tca(tensor, rank, seed=0):
    """Non-negative TCA (CP decomposition). Returns the 3 factors and the relative error."""
    cp = non_negative_parafac(tl.tensor(tensor), rank=rank, n_iter_max=1000, tol=1e-8, init="random",
                              random_state=seed)
    error = np.linalg.norm(tensor - tl.cp_to_tensor(cp)) / np.linalg.norm(tensor)
    return cp.factors, error


def best_tca(tensor, rank, n_fits=5):
    """Fit TCA n_fits times from different random starts, and keep the fit with the lowest error."""
    return min((fit_tca(tensor, rank, seed) for seed in range(n_fits)), key=lambda fit: fit[1])


errors = {rank: best_tca(tensor, rank)[1] for rank in range(1, 7)}

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(list(errors), list(errors.values()), "ko-")
ax.set(xlabel="Number of components R", ylabel="Relative error", title="TCA: error vs number of components")
plt.show()

# %% [markdown]
# The error decreases slowly, and stays high (~0.6): single-trial spike counts are **noisy**, and
# a large part of the variance is noise that no model can explain. There is no clear "elbow": we
# take $R = 4$ components, enough to see the main patterns. (With 4 components, most random starts
# give the same solution; with 3, they do not.)

# %%
rank = 4
(neuron_f, time_f, trial_f), error = best_tca(tensor, rank)
order_c = np.argsort(np.argmax(time_f, axis=0))  # components sorted by the time of their peak
neuron_f, time_f, trial_f = neuron_f[:, order_c], time_f[:, order_c], trial_f[:, order_c]

fig, axes = plt.subplots(rank, 3, figsize=(15, 2.5 * rank))
unit_order = np.argsort(pfc.area.values)  # units grouped by area
for r in range(rank):
    axes[r, 0].bar(np.arange(len(pfc)), neuron_f[unit_order, r], width=1,
                   color=[area_colors[a] for a in pfc.area.values[unit_order]])
    axes[r, 1].plot(t_bins, time_f[:, r], "k", lw=2)
    axes[r, 1].axvline(0, color="gray", ls=":")
    axes[r, 2].scatter(np.arange(len(trials)), trial_f[:, r], s=8, c=[block_colors[b] for b in block])
    axes[r, 0].set_ylabel(f"component {r + 1}")
axes[0, 0].set_title("Neuron factor (color: area)")
axes[0, 1].set_title("Time factor")
axes[0, 2].set_title("Trial factor (color: block)")
axes[-1, 0].set_xlabel("Units, grouped by area: " + ", ".join(sorted(area_colors)))
axes[-1, 1].set_xlabel("Time from trial onset (s)")
axes[-1, 2].set_xlabel("Trial (in session order)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Each component tells a small story: *these* units, with *this* time course, in *these* trials.
# The components are sorted by the time of the peak of their time factor:
#
# 1. active **before** the onset, and lower after it: the units **inhibited** by the stimuli.
#    Its trial factor slowly increases over the session;
# 2. an **onset response**, ~0.15 s after the onset, in all the blocks, and largest in the air
#    puff block;
# 3. another **onset response**, with other units, present in blocks 1–3 but **absent in
#    block 4**: block 4 starts with a different sound (a blue noise), so this is a response to the
#    **tone**. It decreases over the blocks (**habituation**?);
# 4. a **late response**, ~1 s after the onset, absent in block 1 and strong in blocks 2
#    (sound + opto) and 4 (air puff): the response to the **aversive stimulus** that follows the
#    sound. Look at its trial factor in block 3: after the conditioning, with the sound alone, a
#    small late response is still there in some trials.
#
# TCA found by itself, without knowing the blocks, that some trials have a late response and
# others do not. PCA on trial averages could not tell us whether this response is there in every
# trial of a block, or changes within a block.
#
# **Careful.** TCA has no unique solution: the components can come out in a different order, and
# a different random start (`seed`) can give different factors. Always fit the model several
# times and keep the components that come back every time.
#
# **Questions.**
#
# - Fit TCA with `fit_tca(tensor, 4, seed)` for `seed = 1, 2, 3`. Which components are stable?
# - Try $R = 3$ and $R = 5$. Which components split or merge?
# - Which units have the largest neuron factor in the response component? Look at their PSTH
#   (notebook 01, section 3.5).
#
# ## 5. A map of the units: UMAP and self-organizing maps
#
# PCA and TCA reduce the activity over **time**. We can also reduce the **description of the
# units**: in notebooks 01 and 02, we described each unit by a few numbers. Do the units form
# groups (cell types?) in this feature space? We use four features:
#
# - $\log_{10}$ of the firing rate (notebook 01, section 3.1: the rates span orders of magnitude);
# - the **burstiness** $B$ and the **memory** $M$ in the inter-trial intervals (notebook 01,
#   sections 4.3 and 5), without the corrections of the advanced sections;
# - the **waveform class**: 1 for narrow, 0 for wide (notebook 02, section 3.1).
#
# ### 5.1 The features
#
# We compute $B$ and $M$ in the 3 s windows before each trial, as in notebook 01: windows with at
# least 4 ISIs, units with at least 12 such windows. We keep the units with a narrow or wide
# waveform.

# %%
iti = nap.IntervalSet(start=trials.start - 3, end=trials.start)  # 3 s before each trial
min_isi, min_windows = 4, 12


def isis_per_window(spikes, windows, min_isi=4):
    """List of the ISI arrays of the windows that have at least min_isi ISIs."""
    isis = [np.diff(spikes.restrict(windows[k]).t) for k in range(len(windows))]
    return [isi for isi in isis if len(isi) >= min_isi]


def burstiness(isi):
    """Burstiness B = (std - mean) / (std + mean) of an array of ISIs."""
    return (np.std(isi) - np.mean(isi)) / (np.std(isi) + np.mean(isi))


def memory(isi_list):
    """Memory coefficient: correlation between consecutive ISIs, pooled over windows."""
    current = np.concatenate([isi[:-1] for isi in isi_list])
    following = np.concatenate([isi[1:] for isi in isi_list])
    return np.corrcoef(current, following)[0, 1]


B, M = [], []
for i in pfc.keys():
    isi_list = isis_per_window(pfc[i], iti, min_isi)
    enough = len(isi_list) >= min_windows
    B.append(np.mean([burstiness(isi) for isi in isi_list]) if enough else np.nan)
    M.append(memory(isi_list) if enough else np.nan)
pfc.set_info(burstiness=B, memory=M, log_rate=np.log10(pfc.rate))

units_f = pfc[pfc.burstiness.notna() & (pfc.wf_class != "unclassified")]  # units with all features
features = pd.DataFrame({"log10 rate": units_f.log_rate, "burstiness": units_f.burstiness,
                         "memory": units_f.memory, "narrow": (units_f.wf_class == "narrow").astype(float)})
features_z = (features - features.mean()) / features.std()  # z-score: same weight for every feature
print("units with all the features:", len(features))
features.describe().round(2)

# %% [markdown]
# As in PCA, we **z-score** each feature: the distances between units should not depend on the
# units of measurement (Hz, no unit...).
#
# ### 5.2 UMAP
#
# [UMAP](https://umap-learn.readthedocs.io) (Uniform Manifold Approximation and Projection) places
# the units on a 2D map so that units that are **neighbors** in the feature space stay neighbors
# on the map. Unlike PCA, it is non-linear: it can unfold curved groups. Its main parameter,
# `n_neighbors`, sets how local the map is.

# %%
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(features_z.values)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, col in zip(axes[:3], ["log10 rate", "burstiness", "memory"]):
    sc = ax.scatter(*embedding.T, c=features[col], s=15, cmap="viridis")
    plt.colorbar(sc, ax=ax)
    ax.set(title=f"color: {col}")
for c, color in {"wide": "tab:blue", "narrow": "tab:red"}.items():
    sel = units_f.wf_class.values == c
    axes[3].scatter(*embedding[sel].T, s=15, color=color, label=c)
axes[3].set(title="color: waveform class")
axes[3].legend()
for ax in axes:
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
plt.tight_layout()
plt.show()

# %% [markdown]
# The narrow units form their own island: the waveform class is a binary feature, so it splits
# the map in two. Inside each island, the units are ordered by rate, burstiness and memory, which
# vary **continuously**: no clear sub-groups.
#
# **Careful with UMAP.** The map keeps the neighbors, but not the distances: the size of the
# islands and the distance between them mean little. Change `n_neighbors` (5, 50) and
# `random_state`: the map changes, the neighbors much less.
#
# ### 5.3 Self-organizing map and clustering
#
# A **self-organizing map** (SOM, [Kohonen, 1982](https://doi.org/10.1007/BF00337288)) is a grid
# of "nodes", each with a prototype (a point in the feature space). During training, each unit
# pulls the prototype of its closest node, and a little the prototypes of the neighbor nodes on the
# grid. In the end, close nodes have similar prototypes: the grid is a smooth 2D map of the feature
# space. Each unit belongs to the node with the closest prototype.
#
# We use a 6 × 6 grid (`minisom`), then group the 36 prototypes into a few **clusters** with
# k-means: a common way to cluster with a SOM.

# %%
grid = 6
som = MiniSom(grid, grid, features_z.shape[1], sigma=1.5, learning_rate=0.5, random_seed=0)
som.train_random(features_z.values, 5000)
prototypes = som.get_weights()  # grid x grid x features (z-scored)
node = np.array([som.winner(x) for x in features_z.values])  # (row, column) of the node of each unit

n_clusters = 3
kmeans = KMeans(n_clusters, n_init=10, random_state=0).fit(prototypes.reshape(-1, features_z.shape[1]))
node_cluster = kmeans.labels_.reshape(grid, grid)  # cluster of each node
cluster = node_cluster[node[:, 0], node[:, 1]]  # cluster of each unit

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for k, col in enumerate(features.columns):  # component planes: one feature per map
    im = axes[k].imshow(prototypes[:, :, k] * features[col].std() + features[col].mean(), cmap="viridis")
    plt.colorbar(im, ax=axes[k])
    axes[k].set(title=col, xticks=[], yticks=[])
axes[4].imshow(node_cluster, cmap="Set2", vmin=0, vmax=7)
hits = np.zeros((grid, grid), int)
np.add.at(hits, (node[:, 0], node[:, 1]), 1)  # number of units in each node
for r in range(grid):
    for c in range(grid):
        axes[4].text(c, r, hits[r, c], ha="center", va="center")
axes[4].set(title="clusters (number of units per node)", xticks=[], yticks=[])
plt.tight_layout()
plt.show()

units_f.set_info(cluster=cluster)
features.assign(cluster=cluster, area=units_f.area.values).groupby("cluster").agg(
    n_units=("narrow", "size"), rate_Hz=("log10 rate", lambda x: (10 ** x).median()),
    burstiness=("burstiness", "median"), memory=("memory", "median"), narrow=("narrow", "mean")).round(2)

# %% [markdown]
# The first four maps (the **component planes**) show the value of each feature across the grid:
# the SOM has organized the units along gradients of rate, burstiness and memory, and put the
# narrow units in one corner. The k-means clusters cut the grid into regions; the table gives the
# median features of each cluster.
#
# Here the clusters mostly follow the waveform class and the rate. Are they **cell types**? Not
# necessarily: k-means always returns the number of clusters we ask for, even when the data form a
# continuum. Check how stable the clusters are (other seeds, other `n_clusters`), and whether they
# have a meaning outside the features that made them (area, depth, responses to the stimuli).
#
# **Questions.**
#
# - Change `n_clusters`. Is there a number of clusters that comes back for several seeds?
# - Remove the waveform class from the features. Do the narrow units still group together?
# - Are the clusters spread over the four areas (`pd.crosstab(units_f.cluster, units_f.area)`)?
