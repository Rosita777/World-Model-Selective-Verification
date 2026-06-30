import numpy as np

from wmsv.data.transitions import Transition
from wmsv.models.linear_dynamics import LinearDynamicsModel


def test_linear_dynamics_fits_simple_translation():
    transitions = []
    for x in range(5):
        state = np.array([float(x), 0.0])
        action = np.array([1.0, 0.0])
        transitions.append(
            Transition(state, action, state + np.array([0.5, 0.0]), -1.0, False, {})
        )
    model = LinearDynamicsModel.fit(transitions, ridge=1e-6)

    step = model.step(np.array([2.0, 0.0]), np.array([1.0, 0.0]))

    assert abs(step.state[0] - 2.5) < 1e-4
    assert step.reward < 0.0
