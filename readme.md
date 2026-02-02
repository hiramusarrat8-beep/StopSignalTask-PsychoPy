# Stop-Signal Task (SST) – PsychoPy Builder

This repository contains a **stimulus-change Stop-Signal Task (SST)** implemented in **PsychoPy Builder**, designed according to the consensus recommendations by **Verbruggen et al. (2019)**.

The task supports both:
- ✅ **Offline (lab) testing**
- ✅ **Online deployment (Pavlovia / PsychoJS-compatible)**

It generates trial-level behavioural data suitable for **SSRT estimation**.

---

## 📌 Contents
- [Task Overview](#-task-overview)
- [Task Logic](#-task-logic)
- [Timing Parameters](#-timing-parameters-default)
- [Adaptive SSD (Staircase)](#-adaptive-ssd-staircase-procedure)
- [Experimental Structure](#-experimental-structure)
- [Stimuli](#-stimuli)
- [Output Data](#output-data)
- [Repository Structure](#repository-structure)
- [Running Online (Pavlovia)](#running-online-pavlovia)
- [References](#references)
- [Contact](#contact)

---

## 🧠 Task Overview

The Stop-Signal Task measures **response inhibition** by requiring participants to withhold an already initiated response when a stop signal occurs.

- Every trial begins as a **Go trial**
- On a minority of trials (~25%), a **Stop signal** occurs (stimulus colour changes)
- The stimulus **direction remains constant** (only colour changes)

---

## ✅ Task Logic

### Go Trials (~75%)
- White arrow (left or right) appears
- Participant must respond:
  - Left arrow → Left key
  - Right arrow → Right key
- Maximum response window: **1250 ms**

### Stop Trials (~25%)
- White arrow appears first
- After a variable **Stop-Signal Delay (SSD)** the arrow turns red
- Participants must **withhold the response**
  - Any keypress = failed stop
  - No response = successful stop

---

## ⏱ Timing Parameters (default)

| Parameter | Value | Description |
|----------|------:|-------------|
| FIX | 250 ms | Fixation duration |
| MAXRT | 1250 ms | Max response window |
| ITI | 500 ms | Inter-trial interval |
| SSD (start) | 200 ms | Initial stop-signal delay |
| SSD step | 50 ms | Staircase step size |
| Practice feedback | 750 ms | Immediate feedback |
| Block break | 15 s | Between-block break |

✅ All timings are in **milliseconds**.

---

## 🔁 Adaptive SSD (Staircase Procedure)

SSD is adjusted dynamically to target:

> **p(respond | stop) ≈ 0.50**

- Successful stop → SSD increases by **50 ms**
- Failed stop → SSD decreases by **50 ms**
- SSD is bounded to avoid invalid values

---

## 🧩 Experimental Structure

### Practice Phase
- 1 block × 32 trials
- Trial-by-trial feedback
- Block-level performance feedback

### Experimental Phase
- 4 blocks × 64 trials
- No trial-level feedback
- Block-level feedback only

Between blocks, participants see:
- Mean Go RT
- Proportion of missed Go trials (target ≈ 0)
- Proportion of successful stops (target ≈ 0.5)

---

## 🖼 Stimuli

The task uses four stimulus images:

```text
images/
 ├── go_left.png     (white left arrow)
 ├── go_right.png    (white right arrow)
 ├── stop_left.png   (red left arrow)
 └── stop_right.png  (red right arrow)

## Output Data

-The task logs trial-level behavioural measures including:
-Go/Stop trial type
-stimulus direction
-response key
-reaction time (RT)
-SSD per stop trial
-stop success / failure
The output format supports SSRT computation using the integration method.

## Repository Strucuture

├── SST2.psyexp
├── conditions.xlsx
├── images/
├── data/                          # local output (optional)
└── analysis_pipeline/
    ├── SST_data_cleaning.ipynb
    ├── requirements.txt
    └── data_sample/

## Running Online (Pavlovia)

This experiment is PsychoJS-compatible.
Steps:
Open the .psyexp file in PsychoPy Builder
Tools → Pavlovia → Log in
Tools → Pavlovia → Sync project
Pilot the study via Pavlovia before data collection

## References
Verbruggen, F., Aron, A. R., Band, G. P. H., et al. (2019).
A consensus guide to capturing the ability to inhibit actions and impulsive behaviors in the stop-signal task.
eLife, 8:e46323.
https://doi.org/10.7554/eLife.46323

## Contact
Questions or suggestions are welcome via GitHub Issues.


