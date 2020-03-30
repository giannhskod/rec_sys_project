import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.solvers import plugins

from helpers import ProblemAPredictionsMatrix


def create_pyomo_model(
    ri_matrix: pd.DataFrame,
    cs_matrix: pd.DataFrame,
    dis_matrix: pd.DataFrame,
    Lmax: int,
    Kmax: int,
    Dmax: int,
    domain: Set = UnitInterval,
):
    """
    Function that defines and initializes the
    Args:
        ri_matrix:
        cs_matrix:
        dis_matrix:
        Lmax:
        Kmax:
        Dmax:
        domain:

    Returns:

    """
    model = AbstractModel()

    # Define Parameters of the Model
    model.i = Param(within=NonNegativeIntegers, initialize=ri_matrix.shape[1] - 1)
    model.u = Param(within=NonNegativeIntegers, initialize=ri_matrix.shape[0] - 1)
    model.c = Param(within=NonNegativeIntegers, initialize=cs_matrix.shape[0] - 1)

    model.K = Param(initialize=Kmax)
    model.D = Param(initialize=Dmax)
    model.Lmax = Param(initialize=Lmax)
    model.L = RangeSet(0, Lmax - 1)

    model.I = RangeSet(0, model.i)
    model.U = RangeSet(0, model.u)
    model.C = RangeSet(0, model.c)
    model.Cmax = RangeSet(0, cs_matrix.shape[1] - 1)

    model.ri = Param(
        model.U, model.I, initialize=lambda model_ins, i, j: ri_matrix.values[i][j]
    )
    model.ds = Param(
        model.U,
        model.U,
        initialize=lambda model_ins, i, j: 1 - abs(dis_matrix.values[i][j]),
    )

    model.cs = Param(
        model.C, model.Cmax, initialize=lambda model_ins, i, j: cs_matrix.values[i][j]
    )

    model.x = Var(model.U, model.I, domain=domain, bounds=(0, 1))

    # Define Constraints of the Model
    def coverage_constraint(model_ins: AbstractModel, c):
        cat_i = set(
            model_ins.cs[c, i] for i in model_ins.Cmax if model_ins.cs[c, i] != 0
        )
        return (
            sum(sum(model_ins.x[u, i] for u in model_ins.U) for i in cat_i) / len(cat_i)
            == model_ins.K
        )

    def diversity_constraint(model_ins, c):
        cat_i = set(
            model_ins.cs[c, i] for i in model_ins.Cmax if model_ins.cs[c, i] != 0
        )
        constant = model_ins.K * len(cat_i)
        summations = sum(
            sum(
                sum(
                    model_ins.x[u, i] * model_ins.ds[u, v] * model_ins.x[v, i]
                    for v in model_ins.U
                    if v != u
                )
                for u in model_ins.U
            )
            for i in cat_i
        )

        return (2 * summations / (constant * (constant - 1))) >= model_ins.D

    def maximum_recommendation_constraint(model_ins, u):
        return (
            sum(
                sum(model_ins.x[u, model_ins.cs[c, i]] for i in model_ins.Cmax)
                for c in model_ins.C
            )
            == model_ins.Lmax
        )

    model.CoverageConstraint = Constraint(model.C, rule=coverage_constraint)
    model.DiversityConstraint = Constraint(model.C, rule=diversity_constraint)
    model.MaximumRecommendationConstraint = Constraint(
        model.U, rule=maximum_recommendation_constraint
    )

    # Define Objective of the Model
    def maximizer_obj(model_ins):
        return sum(
            model_ins.ri[i] * model_ins.x[i] for i in model_ins.U * model_ins.I
        ) / (model_ins.Lmax * (model_ins.u + 1))

    model.OBJ = Objective(rule=maximizer_obj, sense=maximize)

    return model.create_instance()


def test_constraints(
    model_ins: AbstractModel,
    rs_matrix: pd.DataFrame,
    cs_matrix: pd.DataFrame,
    Lmax: (int, float),
    Kmax: (float),
    Dmax: (float),
    verbose: Boolean = False,
):
    prediction_val = model_ins.x.get_values().items()
    max_rec_const = False
    coverage_const = False
    diversity_const = False

    # Test Maximum Recommendation Limit
    if verbose:
        print("---Test Maximum Recommendation Limit-----")
    cur_row = 0
    sum_row = 0
    for user_idx, pred in prediction_val:
        user_id = rs_matrix.iloc[user_idx[0]].name

        if cur_row != user_idx[0]:
            if sum_row > Lmax:
                if verbose:
                    print(f"Maximum Limit Constraint Violation for user_id {user_id}")
                break
            sum_row = 0
        sum_row += pred
        cur_row = user_idx[0]
    else:
        max_rec_const = True
        if verbose:
            print("Maximum Recommendation Limit Constraint is satisfied")

    # Test Coverage
    if verbose:
        print("---Test Coverage-----")
    coverage_test_mat = np.zeros((cs_matrix.shape[0]))
    for user_idx, pred in prediction_val:
        for cat_id, movies in cs_matrix.iterrows():
            category_movies = [mov for mov in movies.values if mov != 0]
            # check if the movie belong to the category
            if user_idx[1] in category_movies:
                coverage_test_mat[cat_id] += pred

    for cat, cov_sum in enumerate(coverage_test_mat):
        cat_len = len([mov for mov in cs_matrix.loc[cat] if mov != 0])
        cov_val = cov_sum / cat_len
        if not (Kmax - 0.005 < cov_val < Kmax + 0.005):
            if verbose:
                print(f"Coverage Constraint Violation for category {cat}")
            break
        if verbose:
            print(f"Category {cat} - Coverage={cov_val}")

    else:
        coverage_const = True
        if verbose:
            print("Coverage Constraint is satisfied")

    # Test Diversity
    if verbose:
        print("---Test Diversity-----")
    diversity_mat = np.zeros((cs_matrix.shape[0]))
    users_size = rs_matrix.shape[0]
    for cat_id, movies in cs_matrix.iterrows():
        for movie in [mov for mov in movies.values if mov != 0]:
            for user_u in range(0, users_size):
                for user_v in range(1, users_size):
                    if user_v != user_u:
                        diversity_mat[cat_id] += (
                            model_ins.x[(user_u, movie)].value
                            * model_ins.x[(user_v, movie)].value
                            * model_ins.ds[(user_u, user_v)]
                        )

    for cat, div_sum in enumerate(diversity_mat):
        cat_len = len([mov for mov in cs_matrix.loc[cat] if mov != 0])
        div_val = 2 * div_sum / ((Kmax * cat_len) * (Kmax * cat_len - 1))
        if verbose:
            print(f"Category {cat} - Diversity={div_val}")
        if not (div_val > Dmax - 0.005):
            if verbose:
                print(f"Diversity Constraint Violation for category {cat}")
            break
    else:
        diversity_const = True
        if verbose:
            print("Diversity Constraint is satisfied")

    return max_rec_const, coverage_const, diversity_const
