conda create -n jax-gigalens-multinode python=3.11.7
conda activate jax-gigalens-multinode
pip install jax[cuda12]==0.6.2 optax objax ipykernel lenstronomy tqdm scikit-image tensorflow-probability numpy==1.26.4