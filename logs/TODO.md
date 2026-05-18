# TODO

[2026-05-18T04:00:00Z] [planner] colab-remaining-failures: upstream PR to Meta sam3 to make PositionEmbeddingSine._encode_xy honor input dtype natively (Option D from spec §3.4). Until merged and a sam3 release lands, the monkey-patch in src/esam3/models/sam3.py::_patch_pos_enc_dtype remains the stop-gap. Re-evaluate on every sam3 version bump.
