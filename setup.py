import setuptools

setuptools.setup(
    name="score_sde_pytorch",
    version="1.0",
    description="PyTorch implementation for Score-Based Generative Modeling through Stochastic Differential Equations",
    url="https://github.com/mariusarvinte/score_sde_pytorch",
    packages=["score_sde_pytorch", "score_sde_pytorch.models", "score_sde_pytorch.op", "score_sde_pytorch.configs", "score_sde_pytorch.configs.vp", "score_sde_pytorch.configs.ve", "score_sde_pytorch.configs.subvp"],
    install_requires = [
        "ml_collections==0.1.1",
	"torchdiffeq==0.2.3",
    ],
)
