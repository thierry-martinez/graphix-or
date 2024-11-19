import itertools
from graphix import Pattern
from graphix.command import CommandKind
from ortools.sat.python import cp_model

def minimize_space(pattern: Pattern) -> Pattern:
    model = cp_model.CpModel()
    nodes, edges = pattern.get_graph()

    # Discrete time line with two events by node, prepare and measure
    end_of_time = len(nodes) * 2

    # Map each node to preparation time (0 for input nodes)
    preparation_vars = {}

    # Map each node to measure time (`end_of_time` for output nodes)
    measure_vars = {}

    # The objective value to minimize
    max_space = model.new_int_var(0, len(nodes), "max_space")

    # Map each node to the slot index for its interval in the interval graph
    interval_vars = {}

    # Below, intervals that have the same slot index will be constrained to be disjoint
    # and max_space will be constrainted to be an upper bound for slot indexes.

    for node in nodes:
        if node in pattern.input_nodes:
            preparation_var = 0
        else:
            preparation_var = model.new_int_var(0, end_of_time, f"N{node}")
        if node in pattern.output_nodes:
            measure_var = end_of_time
        else:
            measure_var = model.new_int_var(0, end_of_time, f"M{node}")

        # Each node is prepared before being measured.
        model.add(preparation_var < measure_var)

        interval_var = model.new_int_var(0, len(nodes) - 1, f"I{node}")

        # max_space is an upper bound for slot indexes
        model.add(interval_var < max_space)

        preparation_vars[node] = preparation_var
        measure_vars[node] = measure_var
        interval_vars[node] = interval_var

    # For each edge {i, j}, the intervals for i and j overlap.
    for i, j in edges:
        model.add(preparation_vars[i] < measure_vars[j])
        model.add(preparation_vars[j] < measure_vars[i])

    # Runnability constraints: if i is in the domain of j, i is measured first.
    measures = {}
    for cmd in pattern:
        if cmd.kind == CommandKind.M:
            measures[cmd.node] = cmd
            for node in cmd.s_domain:
                if node not in pattern.results:
                    measure_vars[node] < measure_vars[cmd.node]
            for node in cmd.t_domain:
                if node not in pattern.results:
                    measure_vars[node] < measure_vars[cmd.node]

    # Constraints for interval overlap: for each pair {i, j}, one of
    # the three following property should be true:
    # - i and j are not on the same slot index,
    # - i is measured before j is prepared,
    # - j is measured before i is prepared.
    for i, j in itertools.combinations(nodes, 2):
        may_overlap = model.new_bool_var(f"may_overlap_{i}_{j}")
        model.add(interval_vars[i] != interval_vars[j]).only_enforce_if(may_overlap)
        before = model.new_bool_var(f"{i}_before_{j}")
        model.add(measure_vars[i] < preparation_vars[j]).only_enforce_if(before)
        model.add(measure_vars[i] >= preparation_vars[j]).only_enforce_if(~ before)
        after = model.new_bool_var(f"{i}_after_{j}")
        model.add(measure_vars[j] < preparation_vars[i]).only_enforce_if(after)
        model.add(measure_vars[j] >= preparation_vars[i]).only_enforce_if(~ after)
        model.add(may_overlap + before + after >= 1)
    model.minimize(max_space)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL

    measure_order = list(measures.values())

    # Measure commands are sorted according to the order of the measures on the time line.
    measure_order.sort(key=lambda measure: solver.value(measure_vars[measure.node]))

    # We use _reorder_pattern (this method should be public!) to build
    # the optimal pattern from the measure order.
    pattern._reorder_pattern(measure_order)

