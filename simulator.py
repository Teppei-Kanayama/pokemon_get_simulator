from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Ball:
    def __init__(self, ball_type: str) -> None:
        self._random_variable_upper_bounds = dict(
            monster=256,
            super=201,
            hyper=151
        )
        self._ball_coefficient_mapper = dict(
            monster=12,
            super=8,
            hyper=12
        )
        self.ball_type = ball_type

    def get_random_variable(self) -> int:
        random_variable_upper_bound = self._random_variable_upper_bounds[self.ball_type]
        return np.random.randint(0, random_variable_upper_bound)

    def get_coefficient(self) -> int:
        return self._ball_coefficient_mapper[self.ball_type]


class Pokemon:
    def __init__(self, name: str, hp: int, hp_max: int, state: Optional[str] = None) -> None:
        self._state_mapper = dict(
            asleep=25,
            freeze=25,
            burn=12,
            paralysis=12,
            poison=12
        )
        self._rarity_mapper = dict(
            Iwark=45,
            Pikachu=190,
            Mu2=3
        )
        self._state = state
        self._name = name
        self.hp = hp
        self.hp_max = hp_max

    def get_state_adjustment(self) -> int:
        return self._state_mapper[self._state] if self._state else 0

    def get_rarity(self) -> int:
        return self._rarity_mapper[self._name]


def _get_f_value(ball: Ball, pokemon: Pokemon) -> int:
    numerator = int(pokemon.hp_max * 255 / ball.get_coefficient())
    denominator = int(pokemon.hp / 4)
    f_value = numerator / denominator if denominator > 0 else numerator
    return min(255, f_value)


def is_get(ball: Ball, pokemon: Pokemon) -> bool:
    if ball.ball_type == 'master':
        return True
    first_random_variable = ball.get_random_variable() - pokemon.get_state_adjustment()
    if first_random_variable < 0:
        return True
    if first_random_variable > pokemon.get_rarity():
        return False
    second_random_variable = np.random.randint(0, 256)
    if second_random_variable <= _get_f_value(ball, pokemon):
        return True
    return False


def count_thrown_balls(ball: Ball, pokemon: Pokemon) -> int:

    def f(n: int) -> int:
        if is_get(ball=ball, pokemon=pokemon):
            return n
        return f(n + 1)

    return f(1)


def decide_ball(hp: int) -> Ball:
    v = np.random.randint(hp, hp + 200)
    if v < 150:
        return Ball(ball_type='monster')
    if v >= 300:
        return Ball(ball_type='hyper')
    return Ball(ball_type='super')


def get_color(ball: Ball) -> str:
    if ball.ball_type == 'monster':
        return 'r'
    if ball.ball_type == 'super':
        return 'b'
    return 'g'


def main():
    hp_list = np.random.randint(1, 300, 100)
    balls = [decide_ball(hp) for hp in hp_list]
    thrown_balls_list = [count_thrown_balls(ball, Pokemon(name='Iwark', hp=hp, hp_max=300)) for hp, ball in zip(hp_list, balls)]

    ball_types = [ball.ball_type for ball in balls]

    df = pd.DataFrame(dict(hp=hp_list, ball_type=ball_types))
    df = pd.concat([df, pd.get_dummies(df['ball_type'])], axis=1)
    df['constant'] = 1

    if True:
        df = df[['constant', 'hp', 'monster', 'super', 'hyper']]
    else:
        df = df[['constant', 'hp']]

    X = df.values
    Y = np.array(thrown_balls_list)
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    print(beta)

    ball_colors = [get_color(ball) for ball in balls]
    plt.scatter(hp_list, thrown_balls_list, c=ball_colors, s=10)
    plt.xlabel('remaining HP')
    plt.ylabel('the number of balls to get the Iwark')
    plt.savefig('hyper.png')


if __name__ == '__main__':
    main()
