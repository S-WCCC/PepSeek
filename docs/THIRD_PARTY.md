# Dependencies and architectural references

- PyTorch: https://pytorch.org/ — tensor/autograd/training backend. Preserve its license with redistributed binaries.
- RDKit: https://www.rdkit.org/ — chemical graphs and ETKDG conformer generation. Cite the actual version and generation recipe.
- ESM: https://github.com/facebookresearch/esm — frozen protein sequence embeddings. Pretrained weights are external assets with their own source/license provenance; do not redistribute implicitly.
- PSICHIC: https://github.com/huankoh/PSICHIC — pool/interact/unpool architectural inspiration. This project is an original simplified implementation/extension, not the official PSICHIC model or its checkpoints.
- NumPy/SciPy: numerical operations, metrics and Hungarian region assignment.

The repository initially contained only a README and no license declaration. This deployment does not choose a new repository license on behalf of the owner. Check ownership/license terms before public redistribution of code, curated datasets or model weights.

No credentials, native private datasets, pretrained weights or scientific outputs are placed in Git by deployment scripts.
