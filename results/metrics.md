Held-out test set: 60 comments written with phrasings never seen in training.

| system | rouge1 | rouge2 | rougeL | aspect_f1 | overall_acc | format_ok | compression | words |
|---|---|---|---|---|---|---|---|---|
| lead-1 baseline | 0.052 | 0.012 | 0.052 | 0.000 | 0.000 | 0.000 | 0.405 | 9.8 |
| base GPT-2 | 0.010 | 0.000 | 0.010 | 0.000 | 0.000 | 0.000 | 0.862 | 23.1 |
| LoRA (default) | 0.903 | 0.714 | 0.850 | 0.689 | 0.883 | 1.000 | 0.247 | 6.2 |
