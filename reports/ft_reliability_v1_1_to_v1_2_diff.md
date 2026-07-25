# FT-Reliability v1.1 to v1.2 frozen amendment

Made before real smoke, training, or development access:

- B/C/D/E now share a completed five-epoch Degradation-CE warm-up.
- E retention applies only from epoch 6 and only where the frozen teacher is
  correct on clean valid pixels.
- Core success is reliability non-inferiority plus a unique E-vs-C full-eAURC
  endpoint; accuracy gain is a separate strong-result condition.
- Checkpoints and outputs use the explicit v1.2 protocol identity.
