"""
Example demonstrating csv imports (re: https://github.com/deepmind/open_spiel/issues/106)

(Be sure to add any missing imports once you paste this in your module :) )

alpha_rank_all_matches_simple.csv is an example with full payoff matrices specification for a 2x2x2 game.

alpha_rank_all_matches_simple_incomplete.csv is an example where the csv is incomplete in the sense that not all payoff entries are specified (due to Agentc appearing for first player, final row). Thus, this will throw an error.
"""

import pandas as pd
import numpy as np
import ast
import open_spiel.python.egt.alpharank as alpharank
import open_spiel.python.egt.utils as utils
import open_spiel.python.egt.alpharank_visualizer as alpharank_visualizer
from open_spiel.python.egt import heuristic_payoff_table

def payoff_tables_from_df(df):
    # Comments following each line show the corresponding outputs for the example csv above

    df_agents = df['agents'].apply(lambda loc: pd.Series(ast.literal_eval(loc)))

    df_scores = df['scores'].apply(lambda loc: pd.Series(ast.literal_eval(loc)))

    num_players = len(df_agents.columns)

    num_strategies = [len(df_agents[k].unique()) for k in range(num_players)]

    strat_labels = {k: df_agents[k].unique().tolist() for k in range(num_players)}

    labels_to_strat_ids = [
        {label: i for i, label in enumerate(strat_labels[k])}
        for k in range(num_players)
    ]

    # Replace agent names with integer IDs from mappings above, makes it easier to
    # fill payoff tables later
    for k in range(num_players):
        df_agents.replace({k: labels_to_strat_ids[k]}, inplace=True)

    # Init a list of payoff_tables, one per player (in case of asymmetric games)
    payoff_tables = [np.full(num_strategies, np.nan) for k in range(num_players)]

    # Fill the payoffs with the scores
    for i, row in df_scores.iterrows():
        for k in range(num_players):
            payoff_tables[k][tuple(df_agents.iloc[i])] = row[k]

    return payoff_tables, strat_labels


if __name__ == '__main__':
    csv_path = "alpha_rank_all_matches_simple.csv"

    # Import the csv
    df = pd.read_csv(csv_path, index_col=False)

    # Convert to payoff_tables
    payoff_tables, strat_labels = payoff_tables_from_df(df)
    # payoff_tables = heuristic_payoff_table.from_elo_scores([1286, 1322, 1401, 1440, 1457, 1466, 1470])

    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)

    # strat_labels = utils.get_strat_profile_labels(payoff_tables(), payoffs_are_hpt_format)

    # Run AlphaRank
    rhos, rho_m, pi, _, _ = alpharank.compute(payoff_tables, alpha=1e-1)

    # Report & plot results
    alpharank.print_results(
        payoff_tables, payoffs_are_hpt_format, rhos=rhos, rho_m=rho_m, pi=pi)

    utils.print_rankings_table(payoff_tables, pi, strat_labels)

    m_network_plotter = alpharank_visualizer.NetworkPlot(
        payoff_tables, rhos, rho_m, pi, strat_labels, num_top_profiles=7)
    m_network_plotter.compute_and_draw_network()
    # alpharank.sweep_pi_vs_alpha(payoff_tables, visualize=True)