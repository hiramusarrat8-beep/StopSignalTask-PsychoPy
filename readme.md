Stop-Signal Task (SST) – PsychoPy Builder

This repository contains a stimulus-change Stop-Signal Task (SST) implemented in PsychoPy Builder, designed according to the consensus guidelines by Verbruggen et al. (2019).
The task is suitable for offline (lab) and online (Pavlovia) data collection and produces data appropriate for SSRT estimation.

📌 Task Overview

The Stop-Signal Task measures response inhibition by requiring participants to withhold an already initiated response when a stop signal appears.

This implementation uses a stimulus-change design (not Go/No-Go):

Every trial starts as a Go trial

On a minority of trials (~25%), the stimulus changes color after a delay, signaling Stop

The direction never changes, only the color

**Task Logic
Go Trials (~75%)

White arrow (left or right) appears

Participant must respond:

Left arrow → Left key

Right arrow → Right key

Maximum response window: 1250 ms

Stop Trials (~25%)

White arrow appears first

After a variable Stop-Signal Delay (SSD), the arrow turns red

Participants must withhold the response

Any keypress = failed stop

No response = successful stop

⏱ Timing Parameters (default)
Parameter	Value	Description
FIX	250 ms	Fixation duration
MAXRT	1250 ms	Max response window
ITI	500 ms	Inter-trial interval
SSD (start)	200 ms	Initial stop-signal delay
SSD step	50 ms	Staircase step size
Practice feedback	750 ms	Immediate feedback
Block break	15 s	Between-block break

All timings are specified in milliseconds.

Adaptive SSD (Staircase Procedure)

SSD is adjusted dynamically to target p(respond | stop) ≈ 0.50:

Successful stop → SSD increases by 50 ms

Failed stop → SSD decreases by 50 ms

SSD is bounded to prevent invalid values.

Experimental Structure
Practice Phase

1 block × 32 trials

Immediate trial-by-trial feedback

Block-level performance feedback

Experimental Phase

4 blocks × 64 trials

No trial-level feedback

Block-level feedback only

Between blocks, participants see:

Mean Go RT

Proportion of missed Go trials (target ≈ 0)

Proportion of successful stops (target ≈ 0.5)

Stimuli

The task uses four PNG images:

images/
 ├── go_left.png    (white left arrow)
 ├── go_right.png   (white right arrow)
 ├── stop_left.png  (red left arrow)
 └── stop_right.png (red right arrow)


The background is white, and all text is black for high contrast.

📂 Repository Structure
.
├── SST2.psyexp              # PsychoPy Builder file
├── conditions.xlsx          # Trial condition file (Go/Stop)
├── README.md                # This file
├── images/                  # Stimulus images
└── data/                    # Output data (local runs)

📊 Output Data

The task logs:

Trial type (Go / Stop)

Stimulus direction

Response key

Reaction time (RT)

SSD per stop trial

Stop success / failure

The data format supports SSRT computation using the integration method.

🌐 Running Online (Pavlovia)

This experiment is PsychoJS-compatible.

Steps:

Open the .psyexp file in PsychoPy Builder

Log in via Tools → Pavlovia → Log in

Sync the project (Tools → Pavlovia → Sync project)

Pilot the study via Pavlovia before data collection

📖 References

Verbruggen, F., Aron, A. R., Band, G. P. H., et al. (2019).
A consensus guide to capturing the ability to inhibit actions and impulsive behaviors in the stop-signal task.
eLife, 8:e46323.
https://doi.org/10.7554/eLife.46323

Contact

If you use or adapt this task, feel free to cite or acknowledge the repository.
Questions or suggestions are welcome via GitHub issues.
