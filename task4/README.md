### Run example

1. Install requirements:
`pip install -r requirements.txt`

2. Clone submodules
`git submodule update --remote`

3. Run training
`python ppo_example.py`

4. Описание модели
В качестве награды было решено давать единицу награды за каждую новую найденную клетку.
Для исследования местности было решено использовать модуль Curiosity из статьи Intrinsic Curiosity Module, который позволяет лучше учиться агенту в системах с разреженными наградами, основываясь только на получаемых агентом наблюдениях.
