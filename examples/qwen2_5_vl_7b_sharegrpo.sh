set -x

MODEL_PATH=Qwen2.5-VL-3B-Instruct  # replace it with your local file path


FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Your_data_path@train \
    data.val_files=geometry3k@train \
    algorithm.kl_coef=0 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=6 \
    worker.rollout.question_num=2 \
    worker.rollout.gpu_memory_utilization=0.5 \
    trainer.experiment_name=qwen2_5_vl_7b_sharegrpo \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=20 \
    trainer.project_name=easy_r1_sharegrpo \
    trainer.save_checkpoint_path=./workdir/qwen2_5_vl_7b_sharegrpo \
