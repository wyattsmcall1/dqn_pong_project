# Instructions on setting up repo

## Setting up the files

1. Set up a conda environment with python=3.5

    `conda create -n newenv python=3.5`

3. Activate conda environment:

    `source activate newenv`


2. Conda install scipy, scikit-learn:

    `conda install scipy scikit-learn`

4. (Optional) Use pip to install optimized tensor flow wheel

	`pip install --ignore-installed /path/to/while/file.whl`

    or install tensor flow using conda:

    `conda install tensorflow`

5. Install OpenAI [`gym`](https://github.com/openai/gym#installing-everything) using `everything` method into the root/project folder.
* You can skip `MuJoCo` dependencies
* First run:
`apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig`
or for OSX:
`brew install cmake boost boost-python sdl2 swig wget`

* Next clone the directory and `cd` into it:

```
git clone https://github.com/openai/gym.git
cd gym
```
* Then run: `pip install -e '.[all]'`

6. Use pip to install `keras-rl` and `pygame` : `pip install keras-rl pygame`
or for osx
```
pip install keras-rl
pip install pygame==1.9.2
```
## Setting up the game environment

1. The root folder has a file `pong.py` that defines the game. We set this up as game in the `classic_control` environment provided by `gym`:

    `mv pong.py ./gym/gym/envs/classic_control/`

2. This new addition needs to be registered in two places.

    1. First in the `__init__.py` folder in `classic_control`. This makes the environment aware the game module exists. So append the line:

        `from gym.envs.classic_control.pong import PongEnv`

        to the file `../classic_control/__init__.py`
    2. Next we register the module with the gym. Append the lines:

        ```
        register(
            id='pong_new-v0',
            entry_point='gym.envs.classic_control:PongEnv',
            max_episode_steps=500,
        )
        ```
        to the file: `./gym/gym/envs/__init__.py`
