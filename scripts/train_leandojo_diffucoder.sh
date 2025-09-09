
# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

python -u -m main \
  loader.global_batch_size=2 \
  data=leandojo_diffucoder \
  wandb.name=diffucoder \
  wandb.offline=false \
  model=small \
  algo=duo_diffucoder \
  model.length=512 \
  algo.gumbel_tau_log10_start=-3.0 \
  algo.gumbel_tau_log10_end=-3.0 \
  algo.gamma_min=-3.5 \
  algo.gamma_max=-1.75 \
  algo.curriculum_start=0 \
  algo.curriculum_end=30000
