# TODO

Keep this file as the project task queue for user-requested follow-ups.

- Fix or resubmit the failed FiLM Skill Gate Stage 2 LIBERO-PRO evaluation from job `6941278_1`; it failed while starting seed `78`, horizon `16`, object perturbation after writing partial results.
- Fix or resubmit the failed FiLM Skill Gate Stage 2 `prob_threshold_zero` evaluations from job `6946357`; task `0` failed while starting seed `78`, horizon `10`, and task `1` failed while starting seed `42`, horizon `10`, object perturbation.
- After the resubmitted FiLM gate evaluations finish, rerun `/fs/nexus-scratch/yliang17/miniconda3/envs/gr00t/bin/python results/gather_eval_results.py --results-dir results --output results_csv/eval_summary.csv` and update `experiment.md` with final results.
- After training job `6941270` finishes, confirm whether `checkpoint-60000` changed from the pre-existing checkpoint and update `experiment.md` before comparing evaluations.
- Investigate the Hugging Face local-path validation error seen when LIBERO eval jobs start from `/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/liberoall_256_full_skill_router_film_gate_stage2/checkpoint-60000`.
