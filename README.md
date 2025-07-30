# MIRAGE-Bench: LLM Agent is Hallucinating and Where to Find Them

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org/abs/2507.21017)

**Authors:** Weichen Zhang, Yiyou Sun, Pohao Huang, Jiayue Pu, Heyue Lin, Dawn Song

University of California, Berkeley

## Abstract

Hallucinations pose critical risks in large language model (LLM)-based agents. When outputs are inconsistent with the contextual or environmental reality, they manifest incorrect or harmful actions. While recent study have exposed such failures, existing evaluations remain fragmented and lack a principled testbed. In this paper, we present **MIRAGE-Bench**—**M**easuring **I**llusions in **R**isky **AGE**nt settings—the first unified benchmark for eliciting and evaluating hallucinations in interactive LLM-agent scenarios. We begin by introducing a three-part taxonomy to address agentic hallucinations: actions that are unfaithful to (i) task instructions, (ii) execution history, or (iii) environment observations. To analyze, we first elicit such failures by performing a systematic audit of existing agent benchmarks, then synthesize test cases using a snapshot strategy that isolates decision points in deterministic and reproducible manners. To evaluate hallucination behaviors, we adopt a fine-grained-level LLM-as-a-Judge paradigm with tailored risk-aware prompts, enabling scalable, high-fidelity assessment of agent actions without enumerating full action spaces. **MIRAGE-Bench** provides actionable insights on failure modes of LLM agents and lays the groundwork for principled progress in mitigating hallucinations in interactive environments.

## 🚀 Quick Start

1. Clone the repository
```bash
git clone https://github.com/sunblaze-ucb/mirage-bench.git
cd mirage-bench
```

2. Create and activate the Conda environment

```bash
git clone
conda env create -f environment.yml
conda activate mirage
```

3. Run inference for all models

```bash
bash inference_all.sh
```

4. Verify all inference results using LLM-as-a-Judge
```bash
bash verify_all.sh
```

5. Compute metrics
```bash
python ./script/calculate_utility_score.py
python ./script/calculate_hallucination_rate.py
```

6. Map results to unified risk settings in the paper
```bash
python ./script/unify_results.py
```

## 📚 Citation

If you use MIRAGE in your research, please cite:

```bibtex
@misc{zhang2025miragebenchllmagenthallucinating,
      title={MIRAGE-Bench: LLM Agent is Hallucinating and Where to Find Them}, 
      author={Weichen Zhang and Yiyou Sun and Pohao Huang and Jiayue Pu and Heyue Lin and Dawn Song},
      year={2025},
      eprint={2507.21017},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.21017}, 
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


For questions or issues, please open a GitHub issue or contact the authors.