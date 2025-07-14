<div align="center">

<h1> R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.

<h5 align="center"> 

<a href='https://arxiv.org/abs/2505.16673'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<!--
<a href='https://huggingface.co/HuanjinYao/Mulberry_llava_8b'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
<a href='https://huggingface.co/datasets/HuanjinYao/Mulberry-SFT'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'>
-->
<!--<a href='https://huggingface.co/collections/HuanjinYao/denseconnector-66500e173fc8c9f05dc98dea'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/700000183)
<a href='https://huggingface.co/spaces/HuanjinYao/DenseConnector-v1.5-8B'><img src='https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg'></a>-->


[Huanjin Yao](https://scholar.google.com/citations?user=pDtsCBQAAAAJ&hl=zh-CN)<sup>2,3*</sup>,
[Qixiang Yin](https://jxhuang0508.github.io/)<sup>4*</sup>,
[Jingyi Zhang]()<sup>1</sup>,
[Min Yang]()<sup>2</sup>,
[Yibo Wang]()<sup>3</sup>,
[Wenhao Wu]()<sup>5</sup>,

[Fei Su]()<sup>4</sup>,
[Li Shen]()<sup>1</sup>,
[Minghui Qiu]()<sup>2</sup>,
[Dacheng Tao]()<sup>1</sup>
[Jiaxing Huang](https://jxhuang0508.github.io)<sup>1✉️</sup>


<sup>1</sup>[Nanyang Technological University](https://www.ntu.edu.sg/), <sup>2</sup>[ByteDance](), <sup>3</sup>[Tsinghua University](https://www.tsinghua.edu.cn/en/), <sup>4</sup>[BUPT](), <sup>5</sup>[USYD](https://www.sysu.edu.cn/sysuen/)

<sup>*</sup>Equal Contribution,       <sup>✉️</sup>Corresponding Author

</h5>
</div>


## News
- [x] **`May 23, 2025.`** We release our [ShareGRPO Training code](), [52K Training Data]() and [R1-ShareVL-7B Model]()!
- [x] **`May 23, 2025.`** We release our paper in [arxiv](https://arxiv.org/abs/2505.16673).


## Training

### Installation
```bash
git clone https://github.com/hiyouga/EasyR1.git
cd R1-ShareVL
pip install -e .
```
### GRPO Training

```bash
bash examples/qwen2_5_vl_7b_sharegrpo.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## Evaluation
We evaluate R1-ShareVL using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)! Please make sure to include a thinking prompt after the question [here](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/qwen2_vl/model.py#L342).

```bash
R1_PROMPT = r"""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

item = {'type': 'text', 'text': s['value'] + " " + R1_PROMPT}
```

## Citation
If you find this repository is useful, please star🌟 this repo and cite🖇️ our paper.
```bibtex
@misc{yao2025r1sharevl,
      title={R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO}, 
      author={Huanjin Yao and Qixiang Yin and Jingyi Zhang and Min Yang and Yibo Wang and Wenhao Wu and Fei Su and Li Shen and Minghui Qiu and Dacheng Tao and Jiaxing Huang},
      year={2025},
      eprint={2505.16673},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

## Acknowledgment
Our work is primarily based on the following codebases. We are sincerely grateful for their work.
- [EasyR1](https://github.com/hiyouga/EasyR1): We use EasyR1 to fine-tune R1-ShareVL Models.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): We use VLMEvalKit for evaluation.
