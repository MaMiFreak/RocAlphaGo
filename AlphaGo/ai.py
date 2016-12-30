"""Policy players"""
import numpy as np
from operator import itemgetter
from AlphaGo import go
from AlphaGo import mcts


class GreedyPolicyPlayer(object):
    """A player that uses a greedy policy (i.e. chooses the highest probability
       move each turn)
    """

    def __init__(self, policy_function, pass_when_offered=False, move_limit=None):
        self.policy = policy_function
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit

    def get_move(self, state):
        # check move limit
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return go.PASS_MOVE

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if len(state.history) > 100 and state.history[-1] == go.PASS_MOVE:
                return go.PASS_MOVE

        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]

        # check if there are sensible moves left to do
        if len(sensible_moves) > 0:
            move_probs = self.policy.eval_state(state, sensible_moves)
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]

        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE


class GreedyValuePlayer(object):
    """A player that uses a greedy value (i.e. chooses the highest probability
       move each turn)
    """

    def __init__(self, value_function, pass_when_offered=False, move_limit=None):
        self.value = value_function
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit

    def get_move(self, state):
        # check move limit
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return go.PASS_MOVE

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if len(state.history) > 100 and state.history[-1] == go.PASS_MOVE:
                return go.PASS_MOVE

        # list with 'sensible' moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]

        # check if there are 'sensible' moves left to do
        if len(sensible_moves) > 0:
            # list with legal moves
            legal_moves = [move for move in state.get_legal_moves()]

            # generate all possible next states
            state_list = [state.copy() for _ in legal_moves]
            for st, mv in zip(state_list, legal_moves):
                st.do_move(mv)

            # evaluate all possble states
            evaluate_list = [self.value.eval_state(next_state)[0] for next_state in state_list]
            # combine legal_moves and evaluate_list
            move_probs = zip(legal_moves, evaluate_list)

            # get move with highest win chance
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]

        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE


class ProbabilisticPolicyPlayer(object):
    """A player that samples a move in proportion to the probability given by the
    policy.

    By manipulating the 'temperature', moves can be pushed towards totally random
    (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, policy_function, temperature=1.0, pass_when_offered=False, move_limit=None):
        assert(temperature > 0.0)
        self.policy = policy_function
        self.move_limit = move_limit
        self.beta = 1.0 / temperature
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit

    def apply_temperature(self, distribution):
        log_probabilities = np.log(distribution)
        # apply beta exponent to probabilities (in log space)
        log_probabilities = log_probabilities * self.beta
        # scale probabilities to a more numerically stable range (in log space)
        log_probabilities = log_probabilities - log_probabilities.max()
        # convert back from log space
        probabilities = np.exp(log_probabilities)
        # re-normalize the distribution
        return probabilities / probabilities.sum()

    def get_move(self, state):
        # check move limit
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return go.PASS_MOVE

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if len(state.history) > 100 and state.history[-1] == go.PASS_MOVE:
                return go.PASS_MOVE

        # list with 'sensible' moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]

        # check if there are 'sensible' moves left to do
        if len(sensible_moves) > 0:
            move_probs = self.policy.eval_state(state, sensible_moves)
            # zip(*list) is like the 'transpose' of zip;
            # zip(*zip([1,2,3], [4,5,6])) is [(1,2,3), (4,5,6)]
            moves, probabilities = zip(*move_probs)
            # apply 'temperature' to the distribution
            probabilities = self.apply_temperature(probabilities)
            # numpy interprets a list of tuples as 2D, so we must choose an
            # _index_ of moves then apply it in 2 steps
            choice_idx = np.random.choice(len(moves), p=probabilities)
            return moves[choice_idx]

        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE

    def get_moves(self, states):
        """Batch version of get_move. A list of moves is returned (one per state)
        """
        sensible_move_lists = [[move for move in st.get_legal_moves(include_eyes=False)]
                               for st in states]
        all_moves_distributions = self.policy.batch_eval_state(states, sensible_move_lists)
        move_list = [None] * len(states)
        for i, move_probs in enumerate(all_moves_distributions):
            if len(move_probs) == 0 or len(states[i].history) > self.move_limit:
                move_list[i] = go.PASS_MOVE
            else:
                # this 'else' clause is identical to ProbabilisticPolicyPlayer.get_move
                moves, probabilities = zip(*move_probs)
                probabilities = np.array(probabilities)
                probabilities = probabilities ** self.beta
                probabilities = probabilities / probabilities.sum()
                choice_idx = np.random.choice(len(moves), p=probabilities)
                move_list[i] = moves[choice_idx]
        return move_list


class ProbabilisticValuePlayer(object):
    """A player that samples a move in proportion to the probability given by the
    value policy.

    By manipulating the 'temperature', moves can be pushed towards totally random
    (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, value_function, temperature=1.0, pass_when_offered=False, move_limit=None):
        assert(temperature > 0.0)
        self.value = value_function
        self.move_limit = move_limit
        self.beta = 1.0 / temperature
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit

    def apply_temperature(self, distribution):
        log_probabilities = np.log(distribution)
        # apply beta exponent to probabilities (in log space)
        log_probabilities = log_probabilities * self.beta
        # scale probabilities to a more numerically stable range (in log space)
        log_probabilities = log_probabilities - log_probabilities.max()
        # convert back from log space
        probabilities = np.exp(log_probabilities)
        # re-normalize the distribution
        return probabilities / probabilities.sum()

    def get_move(self, state):
        # check move limit
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return go.PASS_MOVE

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if len(state.history) > 100 and state.history[-1] == go.PASS_MOVE:
                return go.PASS_MOVE

        # list with 'sensible' moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]

        # check if there are 'sensible' moves left to do
        if len(sensible_moves) > 0:
            # list with legal moves
            legal_moves = [move for move in state.get_legal_moves()]

            # generate all possible next states
            state_list = [state.copy() for _ in legal_moves]
            for st, mv in zip(state_list, legal_moves):
                st.do_move(mv)

            # evaluate all possble states
            probabilities = [self.value.eval_state(next_state)[0] for next_state in state_list]

            # TODO right now we have to compensate for tanh output(-1/1)
            # TODO move to value.py?
            probabilities = [(prob + 1) / 2 for prob in probabilities]

            # apply 'temperature' to the distribution
            probabilities = self.apply_temperature(probabilities)

            # numpy interprets a list of tuples as 2D, so we must choose an
            # _index_ of moves then apply it in 2 steps
            choice_idx = np.random.choice(len(legal_moves), p=probabilities)
            return legal_moves[choice_idx]

        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE


class MCTSPlayer(object):
    def __init__(self, value_function, policy_function, rollout_function, lmbda=.5, c_puct=5,
                 rollout_limit=500, playout_depth=40, n_playout=100):
        self.mcts = mcts.MCTS(value_function, policy_function, rollout_function, lmbda, c_puct,
                              rollout_limit, playout_depth, n_playout)

    def get_move(self, state):
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(state)
            self.mcts.update_with_move(move)
            return move
        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE
