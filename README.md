Untested, just something on my todo. Might help with training plasticity, allowing for a model that can be better finetuned? Idk.

```python
model = SimpleModel(10, 64, 2)

spectral_reg = SpectralRegularizer(
    model,
    k=3,                    # exponent k
    n_power_iterations=1,   # 1 power-iter per step is usually enough
    store_vectors=False     # set True if you want more stable estimates across steps
)

# add to your loss
loss += spectral_reg()
```

```
@misc{algomancer2025,
  author = {@algomancer},
  title  = {Some Dumb Shit},
  year   = {2025}
}
```

Related (Idk if exactly follows there formulation, only skimmed it).
```
@misc{lewandowski2024learningcontinuallyspectralregularization,
      title={Learning Continually by Spectral Regularization}, 
      author={Alex Lewandowski and Michał Bortkiewicz and Saurabh Kumar and András György and Dale Schuurmans and Mateusz Ostaszewski and Marlos C. Machado},
      year={2024},
      eprint={2406.06811},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.06811}, 
}
```
