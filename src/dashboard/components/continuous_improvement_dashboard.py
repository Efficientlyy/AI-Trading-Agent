"""Dashboard components for the continuous improvement system."""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.analysis_agents.sentiment.ab_testing import (
    ab_testing_framework, ExperimentType, ExperimentStatus
)
from src.analysis_agents.sentiment.continuous_improvement.improvement_manager import (
    continuous_improvement_manager
)
from src.analysis_agents.sentiment.continuous_improvement.stopping_criteria import (
    stopping_criteria_manager
)
from src.analysis_agents.sentiment.continuous_improvement.bayesian_analysis import (
    bayesian_analyzer
)
from src.dashboard.bayesian_visualizations import (
    create_posterior_distribution_plot,
    create_winning_probability_chart,
    create_lift_estimation_chart,
    create_experiment_progress_chart,
    create_expected_loss_chart,
    create_credible_interval_chart,
    create_multi_variant_comparison_chart,
    generate_experiment_visualizations
)


def create_continuous_improvement_layout():
    """Create the layout for the continuous improvement dashboard."""
    return html.Div([
        html.H2("Continuous Improvement System"),
        
        html.Div([
            html.H4("System Status"),
            dbc.Card(
                dbc.CardBody([
                    html.Div(id="improvement-system-status"),
                    html.Div([
                        dbc.Button(
                            "Refresh", id="refresh-improvement-status-btn", 
                            color="primary", className="mr-2"
                        ),
                    ], className="d-flex justify-content-end mt-3")
                ]),
                className="mb-4"
            ),
        ]),
        
        html.Div([
            html.H4("Active Experiments"),
            dbc.Card(
                dbc.CardBody([
                    html.Div(id="active-experiments-table"),
                    html.Div([
                        dbc.Button(
                            "Refresh", id="refresh-active-experiments-btn", 
                            color="primary", className="mr-2"
                        ),
                    ], className="d-flex justify-content-end mt-3")
                ]),
                className="mb-4"
            ),
        ]),
        
        html.Div([
            html.H4("Stopping Criteria Status"),
            dbc.Card(
                dbc.CardBody([
                    html.Div(id="stopping-criteria-status"),
                    html.Div([
                        dbc.Button(
                            "Refresh", id="refresh-stopping-criteria-btn", 
                            color="primary", className="mr-2"
                        ),
                    ], className="d-flex justify-content-end mt-3")
                ]),
                className="mb-4"
            ),
        ]),
        
        html.Div([
            html.H4("Experiment Results"),
            dbc.Card(
                dbc.CardBody([
                    html.Div(id="experiment-results"),
                    html.Div([
                        dbc.Button(
                            "Refresh", id="refresh-experiment-results-btn", 
                            color="primary", className="mr-2"
                        ),
                    ], className="d-flex justify-content-end mt-3")
                ]),
                className="mb-4"
            ),
        ]),
        
        html.Div([
            html.H4("Bayesian Analysis"),
            dbc.Card(
                dbc.CardBody([
                    dcc.Dropdown(
                        id="experiment-selector",
                        placeholder="Select an experiment for Bayesian analysis",
                    ),
                    html.Div(id="bayesian-analysis-results", className="mt-3"),
                ]),
                className="mb-4"
            ),
        ]),
        
        # Interval for regular updates
        dcc.Interval(
            id="improvement-dashboard-interval",
            interval=60000,  # 60 seconds
            n_intervals=0
        ),
    ])


@callback(
    Output("improvement-system-status", "children"),
    [Input("refresh-improvement-status-btn", "n_clicks"),
     Input("improvement-dashboard-interval", "n_intervals")]
)
def update_system_status(n_clicks, n_intervals):
    """Update the system status display."""
    try:
        # Get status from the improvement manager
        status = continuous_improvement_manager.get_status()
        
        # Format the status for display
        status_items = []
        for key, value in status.items():
            if key in ["last_check", "last_experiment_generation"]:
                # Format datetimes
                value_str = value
                try:
                    dt = datetime.fromisoformat(value)
                    value_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
                status_items.append(html.P(f"{key.replace('_', ' ').title()}: {value_str}"))
            else:
                status_items.append(html.P(f"{key.replace('_', ' ').title()}: {value}"))
        
        # Add system enabled status with appropriate color
        enabled_color = "success" if status.get("enabled", False) else "danger"
        enabled_text = "Enabled" if status.get("enabled", False) else "Disabled"
        status_items.insert(
            0, 
            html.Div([
                html.Strong("System Status: "),
                dbc.Badge(enabled_text, color=enabled_color, className="ml-2")
            ], className="mb-3")
        )
        
        return status_items
    
    except Exception as e:
        return html.Div([
            html.P(f"Error loading system status: {str(e)}"),
            html.P("Check if the continuous improvement system is properly initialized.")
        ], className="text-danger")


@callback(
    Output("active-experiments-table", "children"),
    [Input("refresh-active-experiments-btn", "n_clicks"),
     Input("improvement-dashboard-interval", "n_intervals")]
)
def update_active_experiments(n_clicks, n_intervals):
    """Update the active experiments table."""
    try:
        # Get active experiments
        experiments = ab_testing_framework.list_experiments(
            status=[ExperimentStatus.ACTIVE]
        )
        
        if not experiments:
            return html.P("No active experiments at the moment.")
        
        # Create a table of active experiments
        rows = []
        for exp in experiments:
            # Create status badge
            status_color = "primary"
            
            # Create button for details
            details_btn = dbc.Button(
                "Details", 
                id={"type": "experiment-details-btn", "index": exp["id"]},
                color="secondary", size="sm"
            )
            
            # Create row
            row = html.Tr([
                html.Td(exp["name"]),
                html.Td(exp["type"]),
                html.Td(f"{exp.get('total_traffic', 0)} requests"),
                html.Td(exp.get("start_time", "N/A")),
                html.Td(details_btn)
            ])
            rows.append(row)
        
        # Create the table
        table = dbc.Table(
            [
                # Header
                html.Thead(
                    html.Tr([
                        html.Th("Name"),
                        html.Th("Type"),
                        html.Th("Traffic"),
                        html.Th("Start Time"),
                        html.Th("Actions")
                    ])
                ),
                # Body
                html.Tbody(rows)
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
        )
        
        return table
    
    except Exception as e:
        return html.Div([
            html.P(f"Error loading active experiments: {str(e)}"),
            html.P("Check if the AB testing framework is properly initialized.")
        ], className="text-danger")


@callback(
    Output("stopping-criteria-status", "children"),
    [Input("refresh-stopping-criteria-btn", "n_clicks"),
     Input("improvement-dashboard-interval", "n_intervals")]
)
def update_stopping_criteria_status(n_clicks, n_intervals):
    """Update the stopping criteria status display."""
    try:
        # Get active experiments
        experiments = ab_testing_framework.list_experiments(
            status=[ExperimentStatus.ACTIVE]
        )
        
        if not experiments:
            return html.P("No active experiments to evaluate stopping criteria for.")
        
        # Evaluate stopping criteria for each active experiment
        criteria_evaluations = []
        
        for exp_data in experiments:
            experiment_id = exp_data["id"]
            experiment = ab_testing_framework.get_experiment(experiment_id)
            
            if not experiment:
                continue
            
            # Evaluate stopping criteria
            evaluation = stopping_criteria_manager.evaluate_experiment(experiment)
            
            # Create card for this experiment
            criteria_rows = []
            for criterion_name, result in evaluation["criteria_results"].items():
                # Status badge
                status_color = "success" if result["should_stop"] else "warning"
                status_text = "Stop" if result["should_stop"] else "Continue"
                
                row = html.Tr([
                    html.Td(criterion_name.replace("_", " ").title()),
                    html.Td(dbc.Badge(status_text, color=status_color)),
                    html.Td(result["reason"])
                ])
                criteria_rows.append(row)
            
            # Create stopping criteria table
            criteria_table = dbc.Table(
                [
                    # Header
                    html.Thead(
                        html.Tr([
                            html.Th("Criterion"),
                            html.Th("Status"),
                            html.Th("Reason")
                        ])
                    ),
                    # Body
                    html.Tbody(criteria_rows)
                ],
                bordered=True,
                hover=True,
                responsive=True,
                size="sm",
                striped=True,
            )
            
            # Add result header
            should_stop = evaluation["should_stop"]
            overall_color = "success" if should_stop else "warning"
            overall_text = "Should Stop" if should_stop else "Continue"
            
            # Create card for this experiment
            experiment_card = dbc.Card(
                dbc.CardBody([
                    html.H5(exp_data["name"], className="card-title"),
                    html.Div([
                        html.Strong("Overall Status: "),
                        dbc.Badge(overall_text, color=overall_color, className="ml-2 mb-3")
                    ]),
                    criteria_table,
                ]),
                className="mb-3"
            )
            
            criteria_evaluations.append(experiment_card)
        
        return html.Div(criteria_evaluations)
    
    except Exception as e:
        return html.Div([
            html.P(f"Error evaluating stopping criteria: {str(e)}"),
            html.P("Check if the stopping criteria system is properly initialized.")
        ], className="text-danger")


@callback(
    Output("experiment-results", "children"),
    [Input("refresh-experiment-results-btn", "n_clicks"),
     Input("improvement-dashboard-interval", "n_intervals")]
)
def update_experiment_results(n_clicks, n_intervals):
    """Update the experiment results display."""
    try:
        # Get analyzed experiments
        experiments = ab_testing_framework.list_experiments(
            status=[ExperimentStatus.ANALYZED, ExperimentStatus.IMPLEMENTED]
        )
        
        if not experiments:
            return html.P("No analyzed experiments available.")
        
        # Create results for each experiment
        results_cards = []
        
        for exp_data in experiments[:5]:  # Limit to most recent 5
            experiment_id = exp_data["id"]
            
            # Create card for this experiment
            has_winner = exp_data.get("has_winner", False)
            winning_variant = exp_data.get("winning_variant", "No clear winner")
            
            # Status badge
            status_color = "success" if has_winner else "warning"
            status_text = "Has Winner" if has_winner else "Inconclusive"
            
            experiment_card = dbc.Card(
                dbc.CardBody([
                    html.H5(exp_data["name"], className="card-title"),
                    html.Div([
                        html.Strong("Status: "),
                        dbc.Badge(status_text, color=status_color, className="ml-2 mb-3")
                    ]),
                    html.P(f"Type: {exp_data.get('type', 'Unknown')}"),
                    html.P(f"Completed: {exp_data.get('end_time', 'Unknown')}"),
                    html.P(f"Total Traffic: {exp_data.get('total_traffic', 0)} requests"),
                    html.Div([
                        html.Strong("Winner: "),
                        html.Span(winning_variant)
                    ]) if has_winner else html.Div()
                ]),
                className="mb-3"
            )
            
            results_cards.append(experiment_card)
        
        return html.Div(results_cards)
    
    except Exception as e:
        return html.Div([
            html.P(f"Error loading experiment results: {str(e)}"),
            html.P("Check if the AB testing framework is properly initialized.")
        ], className="text-danger")


@callback(
    Output("experiment-selector", "options"),
    [Input("improvement-dashboard-interval", "n_intervals")]
)
def update_experiment_dropdown(n_intervals):
    """Update the experiment selector dropdown."""
    try:
        # Get analyzed and completed experiments
        experiments = ab_testing_framework.list_experiments(
            status=[ExperimentStatus.ANALYZED, ExperimentStatus.COMPLETED, ExperimentStatus.IMPLEMENTED]
        )
        
        if not experiments:
            return []
        
        # Create options for dropdown
        options = []
        for exp in experiments:
            options.append({
                "label": exp["name"],
                "value": exp["id"]
            })
        
        return options
    
    except Exception as e:
        print(f"Error updating experiment dropdown: {e}")
        return []


@callback(
    Output("bayesian-analysis-results", "children"),
    [Input("experiment-selector", "value")]
)
def update_bayesian_analysis(experiment_id):
    """Update the Bayesian analysis results for the selected experiment."""
    if not experiment_id:
        return html.P("Select an experiment to view Bayesian analysis results.")
    
    try:
        # Get the experiment
        experiment = ab_testing_framework.get_experiment(experiment_id)
        
        if not experiment:
            return html.P(f"Experiment with ID {experiment_id} not found.")
        
        # Get stopping criteria evaluation
        stopping_criteria_eval = stopping_criteria_manager.evaluate_experiment(experiment)
        
        # Run Bayesian analysis
        analysis_results = bayesian_analyzer.analyze_experiment(experiment)
        
        # Get the summary
        summary = analysis_results.get_summary()
        
        # Generate visualizations
        visualizations = generate_experiment_visualizations(experiment, stopping_criteria_eval)
        
        # Create tabs for different visualizations
        tabs = dbc.Tabs([
            # Summary tab
            dbc.Tab([
                html.Div([
                    html.H5("Bayesian Analysis Summary"),
                    html.Pre(summary, style={"whiteSpace": "pre-wrap"}),
                ], className="p-3")
            ], label="Summary", tab_id="summary-tab"),
            
            # Winning probability tab
            dbc.Tab([
                html.Div([
                    html.H5("Winning Probability by Variant"),
                    dcc.Graph(
                        figure=visualizations.get("winning_probability", go.Figure()),
                        config={"responsive": True}
                    )
                ], className="p-3")
            ], label="Winning Probability", tab_id="probability-tab"),
            
            # Posterior distributions tab
            dbc.Tab([
                html.Div([
                    html.H5("Posterior Distributions"),
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader(metric.replace("_", " ").title()),
                            dbc.CardBody([
                                dcc.Graph(
                                    figure=visualizations.get(f"posterior_{metric}", go.Figure()),
                                    config={"responsive": True}
                                )
                            ])
                        ], className="mb-3")
                        for metric in ["sentiment_accuracy", "direction_accuracy", "calibration_error", "confidence_score"]
                        if f"posterior_{metric}" in visualizations
                    ])
                ], className="p-3")
            ], label="Posterior Distributions", tab_id="posterior-tab"),
            
            # Lift estimation tab
            dbc.Tab([
                html.Div([
                    html.H5("Lift Estimation"),
                    dcc.Graph(
                        figure=visualizations.get("lift_estimation", go.Figure()),
                        config={"responsive": True}
                    )
                ], className="p-3")
            ], label="Lift Estimation", tab_id="lift-tab"),
            
            # Expected loss tab
            dbc.Tab([
                html.Div([
                    html.H5("Expected Loss (Regret)"),
                    dcc.Graph(
                        figure=visualizations.get("expected_loss", go.Figure()),
                        config={"responsive": True}
                    )
                ], className="p-3")
            ], label="Expected Loss", tab_id="loss-tab"),
            
            # Credible intervals tab
            dbc.Tab([
                html.Div([
                    html.H5("Credible Intervals"),
                    dcc.Graph(
                        figure=visualizations.get("credible_intervals", go.Figure()),
                        config={"responsive": True}
                    )
                ], className="p-3")
            ], label="Credible Intervals", tab_id="credible-tab"),
            
            # Experiment progress tab
            dbc.Tab([
                html.Div([
                    html.H5("Experiment Progress"),
                    dcc.Graph(
                        figure=visualizations.get("experiment_progress", go.Figure()),
                        config={"responsive": True}
                    ),
                    
                    html.H6("Stopping Criteria Status", className="mt-4"),
                    html.Div(id="experiment-stopping-criteria")
                ], className="p-3")
            ], label="Experiment Progress", tab_id="progress-tab"),
            
            # Multi-variant comparison tab
            dbc.Tab([
                html.Div([
                    html.H5("Multi-Variant Comparison"),
                    dcc.Graph(
                        figure=visualizations.get("multi_variant_comparison", go.Figure()),
                        config={"responsive": True}
                    )
                ], className="p-3")
            ], label="Multi-Variant Analysis", tab_id="multi-variant-tab"),
            
        ], id="analysis-tabs", active_tab="summary-tab")
        
        # Create summary card at the top
        winning_variant = analysis_results.get_winning_variant()
        has_winner = analysis_results.has_clear_winner()
        
        status_color = "success" if has_winner else "warning"
        status_text = "Has Clear Winner" if has_winner else "No Clear Winner"
        
        summary_card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H4(experiment.name, className="card-title"),
                    dbc.Badge(status_text, color=status_color, className="ml-2")
                ], className="d-flex align-items-center mb-2"),
                
                html.P(f"Type: {experiment.experiment_type.value}"),
                
                html.Div([
                    html.Strong("Sample Sizes: "),
                    html.Span(", ".join([
                        f"{v.name}: {experiment.variant_metrics[v.id].requests}"
                        for v in experiment.variants
                    ]))
                ]),
                
                html.Div([
                    html.Strong("Winning Variant: "),
                    html.Span(winning_variant if winning_variant else "No clear winner")
                ]) if has_winner else html.Div(),
                
                html.Div([
                    html.Strong("Should Stop: "),
                    dbc.Badge(
                        "Yes" if stopping_criteria_eval["should_stop"] else "No", 
                        color="success" if stopping_criteria_eval["should_stop"] else "warning",
                        className="ml-2"
                    )
                ], className="mt-2")
            ])
        ], className="mb-4")
        
        # Combine everything
        results_div = html.Div([
            summary_card,
            tabs
        ])
        
        # Add callback to update stopping criteria details
        @callback(
            Output("experiment-stopping-criteria", "children"),
            [Input("analysis-tabs", "active_tab")]
        )
        def update_stopping_criteria_details(active_tab):
            if active_tab != "progress-tab":
                return html.Div()
            
            criteria_rows = []
            for criterion_name, result in stopping_criteria_eval["criteria_results"].items():
                status_color = "success" if result["should_stop"] else "warning"
                status_text = "Stop" if result["should_stop"] else "Continue"
                
                row = html.Tr([
                    html.Td(criterion_name.replace("_", " ").title()),
                    html.Td(dbc.Badge(status_text, color=status_color)),
                    html.Td(result["reason"])
                ])
                criteria_rows.append(row)
            
            # Create stopping criteria table
            criteria_table = dbc.Table(
                [
                    # Header
                    html.Thead(
                        html.Tr([
                            html.Th("Criterion"),
                            html.Th("Status"),
                            html.Th("Reason")
                        ])
                    ),
                    # Body
                    html.Tbody(criteria_rows)
                ],
                bordered=True,
                hover=True,
                responsive=True,
                size="sm",
                striped=True,
            )
            
            return criteria_table
        
        return results_div
    
    except Exception as e:
        return html.Div([
            html.P(f"Error running Bayesian analysis: {str(e)}"),
            html.P("Check if the Bayesian analyzer is properly configured.")
        ], className="text-danger")