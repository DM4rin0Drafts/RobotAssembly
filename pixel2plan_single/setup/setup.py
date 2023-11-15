from setuptools import setup

setup(
    name='pixel2plan',
    version='0.0.1',
    description='Solving the game Tangram'
                'Project at TU Darmstadt',
    author='M. Lutter, J. Moos, K. Hansel',
    author_email=' michael@robot-learning.de',
    packages=['simulation', 'environment', 'models', 'utilities'],
    zip_safe=False,
    install_requires=['numpy', 'torch', 'torchvision', 'matplotlib', 'mpi4py', 'pillow', 'tensorboard', 'pybullet']
)
