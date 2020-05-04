from typing import Optional, Tuple

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
    def __init__(self, rarity: int, hp: int, hp_max: int, state: Optional[str] = None) -> None:
        self._state_mapper = dict(
            asleep=25,
            freeze=25,
            burn=12,
            paralysis=12,
            poison=12
        )
        self._state = state
        self.rarity = rarity
        self.hp = hp
        self.hp_max = hp_max

    def get_state_adjustment(self) -> int:
        return self._state_mapper[self._state] if self._state else 0


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
    if first_random_variable > pokemon.rarity:
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


def decide_ball(rarity: int) -> Ball:
    p = - rarity / 245 + 255 / 245
    if np.random.binomial(1, p):
        return Ball(ball_type='super')
    return Ball(ball_type='monster')


def get_color(ball: Ball) -> str:
    if ball.ball_type == 'monster':
        return 'r'
    if ball.ball_type == 'super':
        return 'b'
    return 'g'


def generate_data() -> pd.DataFrame:
    rarity_list = np.random.randint(2, 52, 100) * 5
    balls = [decide_ball(rarity) for rarity in rarity_list]
    ball_types = [ball.ball_type for ball in balls]
    thrown_balls_list = [count_thrown_balls(ball, Pokemon(rarity=rarity, hp=100, hp_max=100)) for rarity, ball in
                         zip(rarity_list, balls)]
    df = pd.DataFrame(dict(rarity=rarity_list, ball_type=ball_types, thrown_balls=thrown_balls_list))
    return df


def get_beta(df: pd.DataFrame, x_columns: Tuple[str, ...], y_column: str) -> float:
    df = pd.concat([df, pd.get_dummies(df['ball_type'])], axis=1)
    df['constant'] = 1
    X = df[list(('constant',) + x_columns)].values
    Y = df[y_column].values
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return beta


def main():
    data = generate_data()
    data.to_csv('resources/data.csv')

    beta1 = get_beta(data, x_columns=('super',), y_column='thrown_balls')
    beta2 = get_beta(data, x_columns=('super', 'rarity'), y_column='thrown_balls')

    # import pdb; pdb.set_trace()
    #
    # ball_colors = [get_color(ball) for ball in balls]
    # plt.scatter(rarity_list, thrown_balls_list, c=ball_colors, s=10)
    # plt.xlabel('remaining HP')
    # plt.ylabel('the number of balls to get the Iwark')
    # plt.savefig('hyper.png')
    print(beta1)
    print(beta2)


if __name__ == '__main__':
    np.random.seed(seed=111)
    main()
