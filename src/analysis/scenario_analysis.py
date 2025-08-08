"""
Scenario Analysis & Simulation for B2B Price Elasticity Modeling
Implements REQUIREMENT 7: Interactive scenario analysis and pricing simulation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import sys
import warnings
import joblib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import optimize
from sklearn.preprocessing import StandardScaler
from itertools import product
import concurrent.futures

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import config_loader, logger

warnings.filterwarnings('ignore')


@dataclass
class ScenarioConfig:
    """Configuration for scenario analysis"""
    scenario_name: str
    price_changes: Dict[str, float]  # Product_ID -> price change percentage
    market_conditions: Dict[str, Any]
    time_horizon: int  # days
    confidence_level: float = 0.95
    monte_carlo_samples: int = 1000


@dataclass
class ScenarioResult:
    """Results from scenario analysis"""
    scenario_name: str
    baseline_metrics: Dict[str, float]
    scenario_metrics: Dict[str, float]
    metric_changes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    risk_metrics: Dict[str, float]


class ScenarioAnalysisSystem:
    """
    Comprehensive Scenario Analysis and Simulation System
    Implements pricing strategy simulation and optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Scenario Analysis System
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config_loader if config is None else config
        self.logger = logger
        self.models = {}
        self.baseline_data = None
        self.scenario_results = {}
        self.optimization_results = {}
        
        # Get scenario analysis configuration
        self.scenario_config = self.config.get('scenario_analysis', {
            'price_range': [-0.3, 0.3],  # -30% to +30%
            'price_steps': 21,
            'monte_carlo_samples': 1000,
            'confidence_level': 0.95,
            'parallel_execution': True,
            'max_workers': 4
        })
        
        self.logger.info("Scenario Analysis System initialized")
    
    def load_trained_models(self, model_path: str) -> None:
        """
        Load trained models for scenario analysis
        
        Args:
            model_path: Path to trained models directory
        """
        self.logger.info(f"Loading models from {model_path}")
        
        model_files = {
            'ensemble': 'ensemble_model.pkl',
            'hierarchical_bayesian': 'hierarchical_bayesian_model.pkl',
            'x_learner': 'x_learner_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            try:
                model_file_path = Path(model_path) / filename
                if model_file_path.exists():
                    self.models[model_name] = joblib.load(model_file_path)
                    self.logger.info(f"Loaded {model_name} model")
                else:
                    self.logger.warning(f"Model file not found: {filename}")
            except Exception as e:
                self.logger.error(f"Error loading {model_name}: {str(e)}")
    
    def set_baseline_data(self, df: pd.DataFrame) -> None:
        """
        Set baseline data for scenario comparisons
        
        Args:
            df: Baseline DataFrame with features and outcomes
        """
        self.baseline_data = df.copy()
        self.logger.info(f"Baseline data set with {len(df)} records")
    
    def simulate_price_change_impacts(self, 
                                    price_changes: Dict[str, float],
                                    segments: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simulate demand impact across pricing scenarios
        
        Args:
            price_changes: Dictionary of Product_ID -> price change percentage
            segments: List of customer segments to analyze (optional)
            
        Returns:
            Dictionary with simulation results
        """
        self.logger.info("Simulating price change impacts...")
        
        if self.baseline_data is None:
            raise ValueError("Baseline data not set. Call set_baseline_data() first.")
        
        # Create scenario data
        scenario_data = self.baseline_data.copy()
        
        # Apply price changes
        for product_id, price_change in price_changes.items():
            mask = scenario_data['Product_ID'] == product_id
            if mask.any():
                scenario_data.loc[mask, 'Net_Price'] *= (1 + price_change)
                scenario_data.loc[mask, 'discount_depth'] = (
                    (scenario_data.loc[mask, 'List_Price'] - scenario_data.loc[mask, 'Net_Price']) / 
                    scenario_data.loc[mask, 'List_Price']
                ).clip(0, 1)
        
        # Filter by segments if provided
        if segments:
            scenario_data = scenario_data[scenario_data['Customer_Segment'].isin(segments)]
        
        # Calculate predictions using available models
        predictions = {}
        win_probabilities = {}
        
        for model_name, model in self.models.items():
            try:
                # Prepare features for prediction
                feature_cols = [col for col in scenario_data.columns if col not in [
                    'Quote_ID', 'Customer_ID', 'Product_ID', 'Status', 'Quote_Date'
                ]]
                X_scenario = scenario_data[feature_cols].fillna(0)
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scenario)
                    if proba.shape[1] > 1:
                        win_prob = proba[:, 1]
                    else:
                        win_prob = proba[:, 0]
                elif hasattr(model, 'predict'):
                    win_prob = model.predict(X_scenario)
                else:
                    continue
                
                predictions[model_name] = win_prob
                win_probabilities[model_name] = np.mean(win_prob)
                
            except Exception as e:
                self.logger.warning(f"Error with model {model_name}: {str(e)}")
        
        # Calculate baseline metrics for comparison
        baseline_metrics = self._calculate_baseline_metrics(segments)
        
        # Calculate scenario metrics
        scenario_metrics = self._calculate_scenario_metrics(scenario_data, predictions)
        
        # Calculate impacts
        impact_analysis = {}
        for metric in baseline_metrics:
            if metric in scenario_metrics:
                baseline_val = baseline_metrics[metric]
                scenario_val = scenario_metrics[metric]
                
                if baseline_val != 0:
                    pct_change = (scenario_val - baseline_val) / baseline_val * 100
                else:
                    pct_change = 0
                
                impact_analysis[metric] = {
                    'baseline': baseline_val,
                    'scenario': scenario_val,
                    'absolute_change': scenario_val - baseline_val,
                    'percent_change': pct_change
                }
        
        # Calculate elasticity estimates
        elasticity_estimates = self._calculate_price_elasticity(
            price_changes, impact_analysis
        )
        
        return {
            'price_changes': price_changes,
            'segments_analyzed': segments,
            'baseline_metrics': baseline_metrics,
            'scenario_metrics': scenario_metrics,
            'impact_analysis': impact_analysis,
            'model_predictions': predictions,
            'win_probabilities': win_probabilities,
            'elasticity_estimates': elasticity_estimates,
            'num_records_analyzed': len(scenario_data)
        }
    
    def optimize_pricing_strategies(self, 
                                  products: List[str],
                                  objective: str = 'revenue',
                                  constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize pricing strategies for revenue/profit maximization
        
        Args:
            products: List of Product_IDs to optimize
            objective: Optimization objective ('revenue', 'profit', 'win_rate')
            constraints: Dictionary of constraints
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Optimizing pricing strategy for {len(products)} products...")
        
        if not constraints:
            constraints = {
                'min_price_change': -0.3,  # -30%
                'max_price_change': 0.3,   # +30%
                'min_win_probability': 0.1,
                'max_risk_tolerance': 0.2
            }
        
        # Define optimization bounds
        bounds = [(constraints['min_price_change'], constraints['max_price_change']) 
                 for _ in products]
        
        # Define objective function
        def objective_function(price_changes):
            price_change_dict = dict(zip(products, price_changes))
            
            # Simulate scenario
            simulation_results = self.simulate_price_change_impacts(price_change_dict)
            
            # Calculate objective value
            if objective == 'revenue':
                return -simulation_results['scenario_metrics'].get('total_revenue', 0)
            elif objective == 'profit':
                revenue = simulation_results['scenario_metrics'].get('total_revenue', 0)
                # Estimate profit (simplified - assume 20% base margin)
                profit = revenue * 0.2
                return -profit
            elif objective == 'win_rate':
                return -simulation_results['scenario_metrics'].get('win_rate', 0)
            else:
                return 0
        
        # Constraint functions
        constraints_list = []
        
        # Win probability constraint
        if 'min_win_probability' in constraints:
            def win_prob_constraint(price_changes):
                price_change_dict = dict(zip(products, price_changes))
                simulation_results = self.simulate_price_change_impacts(price_change_dict)
                win_rate = simulation_results['scenario_metrics'].get('win_rate', 0)
                return win_rate - constraints['min_win_probability']
            
            constraints_list.append({
                'type': 'ineq',
                'fun': win_prob_constraint
            })
        
        try:
            # Perform optimization
            initial_guess = [0.0] * len(products)  # Start with no price changes
            
            result = optimize.minimize(
                objective_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 100, 'disp': True}
            )
            
            optimal_prices = dict(zip(products, result.x))
            
            # Simulate optimal scenario
            optimal_simulation = self.simulate_price_change_impacts(optimal_prices)
            
            optimization_results = {
                'optimization_successful': result.success,
                'optimal_price_changes': optimal_prices,
                'objective_value': -result.fun,
                'optimization_message': result.message,
                'iterations': result.nit,
                'simulation_results': optimal_simulation,
                'constraints_satisfied': self._check_constraints(
                    optimal_simulation, constraints
                )
            }
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            optimization_results = {
                'optimization_successful': False,
                'error': str(e),
                'optimal_price_changes': {},
                'objective_value': None
            }
        
        self.optimization_results[objective] = optimization_results
        return optimization_results
    
    def model_competitive_responses(self, 
                                  own_price_changes: Dict[str, float],
                                  competitor_scenarios: List[Dict]) -> Dict[str, Any]:
        """
        Model competitive responses and market share implications
        
        Args:
            own_price_changes: Our price changes
            competitor_scenarios: List of competitor response scenarios
            
        Returns:
            Dictionary with competitive analysis
        """
        self.logger.info("Modeling competitive responses...")
        
        competitive_analysis = {}
        
        # Base case - no competitive response
        base_case = self.simulate_price_change_impacts(own_price_changes)
        competitive_analysis['base_case'] = base_case
        
        # Analyze each competitor scenario
        for i, comp_scenario in enumerate(competitor_scenarios):
            scenario_name = comp_scenario.get('name', f'competitor_scenario_{i}')
            
            # Simulate market share impact
            market_share_impact = self._simulate_market_share_impact(
                own_price_changes, 
                comp_scenario
            )
            
            # Calculate adjusted win probabilities
            adjusted_scenario = self._adjust_for_competitive_response(
                base_case, 
                comp_scenario
            )
            
            competitive_analysis[scenario_name] = {
                'competitor_actions': comp_scenario,
                'market_share_impact': market_share_impact,
                'adjusted_metrics': adjusted_scenario,
                'competitive_advantage': self._calculate_competitive_advantage(
                    base_case, adjusted_scenario
                )
            }
        
        # Calculate overall competitive risk
        competitive_risk = self._assess_competitive_risk(competitive_analysis)
        competitive_analysis['competitive_risk_assessment'] = competitive_risk
        
        return competitive_analysis
    
    def create_interactive_dashboards(self, 
                                    scenario_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create interactive dashboards for scenario exploration
        
        Args:
            scenario_results: Results from scenario analysis
            
        Returns:
            Dictionary of Plotly figures
        """
        self.logger.info("Creating interactive dashboards...")
        
        dashboards = {}
        
        # 1. Price Elasticity Curve Dashboard
        elasticity_fig = self._create_elasticity_curve_dashboard(scenario_results)
        dashboards['price_elasticity'] = elasticity_fig
        
        # 2. Win Probability Heatmap
        win_prob_fig = self._create_win_probability_heatmap(scenario_results)
        dashboards['win_probability_heatmap'] = win_prob_fig
        
        # 3. Revenue Impact Visualization
        revenue_fig = self._create_revenue_impact_dashboard(scenario_results)
        dashboards['revenue_impact'] = revenue_fig
        
        # 4. Risk-Return Analysis
        risk_return_fig = self._create_risk_return_dashboard(scenario_results)
        dashboards['risk_return'] = risk_return_fig
        
        # 5. Sensitivity Analysis
        sensitivity_fig = self._create_sensitivity_analysis_dashboard(scenario_results)
        dashboards['sensitivity_analysis'] = sensitivity_fig
        
        return dashboards
    
    def run_sensitivity_analysis(self, 
                                key_parameters: List[str],
                                parameter_ranges: Dict[str, Tuple[float, float]],
                                base_scenario: Dict[str, float]) -> Dict[str, Any]:
        """
        Run sensitivity analysis on key parameters
        
        Args:
            key_parameters: List of parameters to analyze
            parameter_ranges: Dictionary of parameter -> (min, max) ranges
            base_scenario: Base scenario for comparison
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        self.logger.info("Running sensitivity analysis...")
        
        sensitivity_results = {}
        
        # One-at-a-time sensitivity analysis
        for param in key_parameters:
            param_results = []
            
            if param not in parameter_ranges:
                continue
            
            min_val, max_val = parameter_ranges[param]
            param_values = np.linspace(min_val, max_val, 11)
            
            for param_val in param_values:
                # Create scenario with parameter change
                scenario = base_scenario.copy()
                
                if param.startswith('price_change_'):
                    product_id = param.replace('price_change_', '')
                    scenario[product_id] = param_val
                
                # Simulate scenario
                try:
                    sim_result = self.simulate_price_change_impacts(scenario)
                    
                    param_results.append({
                        'parameter_value': param_val,
                        'win_rate': sim_result['scenario_metrics'].get('win_rate', 0),
                        'total_revenue': sim_result['scenario_metrics'].get('total_revenue', 0),
                        'risk_score': sim_result.get('risk_metrics', {}).get('overall_risk', 0)
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Sensitivity analysis failed for {param}={param_val}: {str(e)}")
            
            sensitivity_results[param] = param_results
        
        # Calculate sensitivity indices
        sensitivity_indices = self._calculate_sensitivity_indices(sensitivity_results)
        
        # Two-way interactions (simplified)
        interaction_effects = self._analyze_parameter_interactions(
            key_parameters[:3], parameter_ranges, base_scenario
        )
        
        return {
            'one_way_sensitivity': sensitivity_results,
            'sensitivity_indices': sensitivity_indices,
            'interaction_effects': interaction_effects,
            'most_sensitive_parameters': sorted(
                sensitivity_indices.items(), 
                key=lambda x: x[1]['total_sensitivity'], 
                reverse=True
            )[:5]
        }
    
    def run_monte_carlo_simulation(self, 
                                 scenario_config: ScenarioConfig,
                                 uncertainty_distributions: Dict[str, Dict]) -> ScenarioResult:
        """
        Run Monte Carlo simulation for scenario under uncertainty
        
        Args:
            scenario_config: Configuration for the scenario
            uncertainty_distributions: Distributions for uncertain parameters
            
        Returns:
            ScenarioResult with confidence intervals
        """
        self.logger.info(f"Running Monte Carlo simulation for {scenario_config.scenario_name}...")
        
        n_samples = scenario_config.monte_carlo_samples
        results_distribution = []
        
        for i in range(n_samples):
            # Sample uncertain parameters
            sampled_params = self._sample_uncertain_parameters(uncertainty_distributions)
            
            # Create scenario with sampled parameters
            scenario_prices = scenario_config.price_changes.copy()
            
            # Add uncertainty to price changes
            for product_id in scenario_prices:
                uncertainty_key = f'price_uncertainty_{product_id}'
                if uncertainty_key in sampled_params:
                    scenario_prices[product_id] += sampled_params[uncertainty_key]
            
            # Simulate scenario
            try:
                sim_result = self.simulate_price_change_impacts(scenario_prices)
                results_distribution.append(sim_result['scenario_metrics'])
            except Exception as e:
                self.logger.warning(f"Monte Carlo sample {i} failed: {str(e)}")
        
        # Calculate statistics
        if results_distribution:
            scenario_metrics = self._calculate_monte_carlo_statistics(
                results_distribution, scenario_config.confidence_level
            )
            
            # Calculate baseline for comparison
            baseline_metrics = self._calculate_baseline_metrics()
            
            # Calculate metric changes
            metric_changes = {}
            confidence_intervals = {}
            
            for metric in baseline_metrics:
                if metric in scenario_metrics['mean']:
                    baseline_val = baseline_metrics[metric]
                    scenario_val = scenario_metrics['mean'][metric]
                    
                    metric_changes[metric] = (scenario_val - baseline_val) / baseline_val * 100
                    confidence_intervals[metric] = (
                        scenario_metrics['confidence_intervals'][metric][0],
                        scenario_metrics['confidence_intervals'][metric][1]
                    )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(results_distribution)
            
            return ScenarioResult(
                scenario_name=scenario_config.scenario_name,
                baseline_metrics=baseline_metrics,
                scenario_metrics=scenario_metrics['mean'],
                metric_changes=metric_changes,
                confidence_intervals=confidence_intervals,
                risk_metrics=risk_metrics
            )
        else:
            raise ValueError("Monte Carlo simulation failed - no valid samples")
    
    # Helper methods for internal calculations
    
    def _calculate_baseline_metrics(self, segments: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate baseline metrics from historical data"""
        
        data = self.baseline_data.copy()
        
        if segments:
            data = data[data['Customer_Segment'].isin(segments)]
        
        metrics = {
            'win_rate': (data['Status'] == 'Won').mean(),
            'total_revenue': (data[data['Status'] == 'Won']['Net_Price']).sum(),
            'avg_deal_size': data[data['Status'] == 'Won']['Net_Price'].mean(),
            'total_opportunities': len(data),
            'conversion_count': (data['Status'] == 'Won').sum()
        }
        
        return metrics
    
    def _calculate_scenario_metrics(self, 
                                  scenario_data: pd.DataFrame, 
                                  predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate metrics for scenario data"""
        
        # Use ensemble average if multiple models available
        if predictions:
            avg_win_prob = np.mean(list(predictions.values()), axis=0)
        else:
            # Fallback - use historical win rate
            avg_win_prob = np.full(len(scenario_data), 0.5)
        
        # Calculate expected metrics
        expected_wins = avg_win_prob.sum()
        expected_revenue = (scenario_data['Net_Price'] * avg_win_prob).sum()
        
        metrics = {
            'win_rate': avg_win_prob.mean(),
            'total_revenue': expected_revenue,
            'avg_deal_size': expected_revenue / max(expected_wins, 1),
            'total_opportunities': len(scenario_data),
            'conversion_count': expected_wins
        }
        
        return metrics
    
    def _calculate_price_elasticity(self, 
                                  price_changes: Dict[str, float],
                                  impact_analysis: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate price elasticity estimates"""
        
        elasticity_estimates = {}
        
        for product_id, price_change in price_changes.items():
            if abs(price_change) > 0.001:  # Avoid division by zero
                # Calculate demand elasticity
                if 'win_rate' in impact_analysis:
                    demand_change = impact_analysis['win_rate']['percent_change'] / 100
                    elasticity = demand_change / price_change
                    elasticity_estimates[f'{product_id}_demand_elasticity'] = elasticity
                
                # Calculate revenue elasticity
                if 'total_revenue' in impact_analysis:
                    revenue_change = impact_analysis['total_revenue']['percent_change'] / 100
                    revenue_elasticity = revenue_change / price_change
                    elasticity_estimates[f'{product_id}_revenue_elasticity'] = revenue_elasticity
        
        return elasticity_estimates
    
    def _simulate_market_share_impact(self, 
                                    own_changes: Dict[str, float], 
                                    competitor_scenario: Dict) -> Dict[str, float]:
        """Simulate market share impact from competitive responses"""
        
        # Simplified market share model
        # In practice, this would use more sophisticated competitive dynamics
        
        market_share_impact = {}
        
        for product_id, our_change in own_changes.items():
            # Assume competitor follows with some response
            competitor_response = competitor_scenario.get('price_response_factor', 0.5)
            competitor_change = our_change * competitor_response
            
            # Calculate relative price advantage
            relative_advantage = our_change - competitor_change
            
            # Translate to market share (simplified logit model)
            market_share_change = -2.0 * relative_advantage  # Elasticity assumption
            market_share_impact[product_id] = market_share_change
        
        return market_share_impact
    
    def _adjust_for_competitive_response(self, 
                                       base_case: Dict[str, Any], 
                                       competitor_scenario: Dict) -> Dict[str, Any]:
        """Adjust scenario results for competitive response"""
        
        adjusted_scenario = base_case.copy()
        
        # Adjust win probabilities based on competitive intensity
        competitive_intensity = competitor_scenario.get('intensity_factor', 1.0)
        adjustment_factor = 1.0 / competitive_intensity
        
        # Apply adjustment to key metrics
        if 'scenario_metrics' in adjusted_scenario:
            metrics = adjusted_scenario['scenario_metrics']
            metrics['win_rate'] *= adjustment_factor
            metrics['total_revenue'] *= adjustment_factor
            metrics['conversion_count'] *= adjustment_factor
        
        return adjusted_scenario
    
    def _calculate_competitive_advantage(self, 
                                       base_case: Dict[str, Any], 
                                       adjusted_case: Dict[str, Any]) -> Dict[str, float]:
        """Calculate competitive advantage metrics"""
        
        base_metrics = base_case.get('scenario_metrics', {})
        adj_metrics = adjusted_case.get('scenario_metrics', {})
        
        advantage = {}
        
        for metric in base_metrics:
            if metric in adj_metrics and base_metrics[metric] != 0:
                advantage_pct = ((base_metrics[metric] - adj_metrics[metric]) / 
                               base_metrics[metric] * 100)
                advantage[f'{metric}_advantage_loss'] = advantage_pct
        
        return advantage
    
    def _assess_competitive_risk(self, competitive_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Assess overall competitive risk"""
        
        risk_assessment = {
            'low_risk_probability': 0.3,
            'medium_risk_probability': 0.5,
            'high_risk_probability': 0.2,
            'expected_market_share_loss': 5.0,  # percentage
            'revenue_at_risk': 10.0  # percentage
        }
        
        # Calculate based on scenario results
        scenario_count = len([k for k in competitive_analysis.keys() 
                            if k not in ['base_case', 'competitive_risk_assessment']])
        
        if scenario_count > 0:
            avg_advantage_loss = np.mean([
                sum(comp_analysis.get('competitive_advantage', {}).values()) 
                for comp_analysis in competitive_analysis.values()
                if isinstance(comp_analysis, dict) and 'competitive_advantage' in comp_analysis
            ])
            
            # Adjust risk based on calculated losses
            risk_assessment['revenue_at_risk'] = max(abs(avg_advantage_loss), 5.0)
        
        return risk_assessment
    
    def _check_constraints(self, simulation_results: Dict[str, Any], constraints: Dict) -> bool:
        """Check if simulation results satisfy constraints"""
        
        metrics = simulation_results.get('scenario_metrics', {})
        
        # Check minimum win probability
        if 'min_win_probability' in constraints:
            if metrics.get('win_rate', 0) < constraints['min_win_probability']:
                return False
        
        # Additional constraint checks can be added here
        
        return True
    
    def _sample_uncertain_parameters(self, uncertainty_distributions: Dict[str, Dict]) -> Dict[str, float]:
        """Sample parameters from uncertainty distributions"""
        
        sampled_params = {}
        
        for param, distribution in uncertainty_distributions.items():
            dist_type = distribution.get('type', 'normal')
            
            if dist_type == 'normal':
                mean = distribution.get('mean', 0)
                std = distribution.get('std', 0.1)
                sampled_params[param] = np.random.normal(mean, std)
                
            elif dist_type == 'uniform':
                low = distribution.get('low', -0.1)
                high = distribution.get('high', 0.1)
                sampled_params[param] = np.random.uniform(low, high)
                
            elif dist_type == 'beta':
                alpha = distribution.get('alpha', 2)
                beta = distribution.get('beta', 2)
                sampled_params[param] = np.random.beta(alpha, beta)
        
        return sampled_params
    
    def _calculate_monte_carlo_statistics(self, 
                                        results_distribution: List[Dict], 
                                        confidence_level: float) -> Dict[str, Any]:
        """Calculate statistics from Monte Carlo simulation results"""
        
        # Extract metrics across all samples
        all_metrics = {}
        for sample in results_distribution:
            for metric, value in sample.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        statistics = {
            'mean': {},
            'std': {},
            'confidence_intervals': {}
        }
        
        alpha = 1 - confidence_level
        
        for metric, values in all_metrics.items():
            values = np.array(values)
            
            statistics['mean'][metric] = np.mean(values)
            statistics['std'][metric] = np.std(values)
            statistics['confidence_intervals'][metric] = (
                np.percentile(values, 100 * alpha/2),
                np.percentile(values, 100 * (1 - alpha/2))
            )
        
        return statistics
    
    def _calculate_risk_metrics(self, results_distribution: List[Dict]) -> Dict[str, float]:
        """Calculate risk metrics from simulation results"""
        
        # Extract revenue values for risk calculation
        revenues = [sample.get('total_revenue', 0) for sample in results_distribution]
        revenues = np.array(revenues)
        
        if len(revenues) > 0:
            mean_revenue = np.mean(revenues)
            
            risk_metrics = {
                'value_at_risk_5pct': np.percentile(revenues, 5),
                'value_at_risk_1pct': np.percentile(revenues, 1),
                'expected_shortfall_5pct': np.mean(revenues[revenues <= np.percentile(revenues, 5)]),
                'volatility': np.std(revenues),
                'downside_risk': np.std(revenues[revenues < mean_revenue]),
                'probability_of_loss': np.mean(revenues < mean_revenue * 0.95)
            }
        else:
            risk_metrics = {
                'value_at_risk_5pct': 0,
                'value_at_risk_1pct': 0,
                'expected_shortfall_5pct': 0,
                'volatility': 0,
                'downside_risk': 0,
                'probability_of_loss': 0
            }
        
        return risk_metrics
    
    def _calculate_sensitivity_indices(self, sensitivity_results: Dict) -> Dict[str, Dict]:
        """Calculate sensitivity indices for parameters"""
        
        indices = {}
        
        for param, param_results in sensitivity_results.items():
            if not param_results:
                continue
            
            # Extract values for analysis
            param_values = [r['parameter_value'] for r in param_results]
            win_rates = [r['win_rate'] for r in param_results]
            revenues = [r['total_revenue'] for r in param_results]
            
            # Calculate sensitivity indices (simplified)
            win_rate_sensitivity = np.std(win_rates) / (np.std(param_values) + 1e-6)
            revenue_sensitivity = np.std(revenues) / (np.std(param_values) + 1e-6)
            
            indices[param] = {
                'win_rate_sensitivity': win_rate_sensitivity,
                'revenue_sensitivity': revenue_sensitivity,
                'total_sensitivity': win_rate_sensitivity + revenue_sensitivity
            }
        
        return indices
    
    def _analyze_parameter_interactions(self, 
                                      parameters: List[str], 
                                      parameter_ranges: Dict, 
                                      base_scenario: Dict) -> Dict[str, float]:
        """Analyze interactions between parameters (simplified)"""
        
        interactions = {}
        
        # Only analyze pairs to keep computational cost manageable
        for i in range(len(parameters)):
            for j in range(i+1, len(parameters)):
                param1, param2 = parameters[i], parameters[j]
                
                if param1 in parameter_ranges and param2 in parameter_ranges:
                    # Sample interaction effect
                    interaction_effect = self._measure_interaction_effect(
                        param1, param2, parameter_ranges, base_scenario
                    )
                    interactions[f'{param1}_x_{param2}'] = interaction_effect
        
        return interactions
    
    def _measure_interaction_effect(self, 
                                  param1: str, 
                                  param2: str, 
                                  parameter_ranges: Dict, 
                                  base_scenario: Dict) -> float:
        """Measure interaction effect between two parameters"""
        
        # Simple 2x2 factorial design
        try:
            low1, high1 = parameter_ranges[param1]
            low2, high2 = parameter_ranges[param2]
            
            scenarios = [
                {param1: low1, param2: low2},
                {param1: low1, param2: high2},
                {param1: high1, param2: low2},
                {param1: high1, param2: high2}
            ]
            
            results = []
            for scenario_params in scenarios:
                scenario = base_scenario.copy()
                scenario.update(scenario_params)
                
                sim_result = self.simulate_price_change_impacts(scenario)
                results.append(sim_result['scenario_metrics'].get('total_revenue', 0))
            
            # Calculate interaction effect (simplified)
            main_effect1 = (results[2] + results[3]) / 2 - (results[0] + results[1]) / 2
            main_effect2 = (results[1] + results[3]) / 2 - (results[0] + results[2]) / 2
            total_effect = results[3] - results[0]
            
            interaction = total_effect - main_effect1 - main_effect2
            
            return interaction
            
        except Exception as e:
            self.logger.warning(f"Interaction analysis failed for {param1} x {param2}: {str(e)}")
            return 0.0
    
    # Dashboard creation methods
    
    def _create_elasticity_curve_dashboard(self, scenario_results: Dict[str, Any]) -> go.Figure:
        """Create price elasticity curve dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Demand Elasticity', 'Revenue Elasticity', 
                           'Win Rate vs Price', 'Revenue vs Price'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add elasticity curves (placeholder data for demo)
        price_range = np.linspace(-0.3, 0.3, 21)
        
        # Demand curve
        demand_curve = 1 / (1 + np.exp(5 * price_range))  # Logistic curve
        fig.add_trace(
            go.Scatter(x=price_range*100, y=demand_curve, 
                      name='Demand Response', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Revenue curve
        revenue_curve = price_range * demand_curve
        fig.add_trace(
            go.Scatter(x=price_range*100, y=revenue_curve, 
                      name='Revenue Response', line=dict(color='green')),
            row=1, col=2
        )
        
        fig.update_layout(height=600, title_text="Price Elasticity Analysis Dashboard")
        
        return fig
    
    def _create_win_probability_heatmap(self, scenario_results: Dict[str, Any]) -> go.Figure:
        """Create win probability heatmap"""
        
        # Create sample heatmap data
        products = [f'Product_{i}' for i in range(1, 6)]
        price_changes = np.linspace(-0.3, 0.3, 11)
        
        # Generate sample win probability data
        z_data = []
        for i, product in enumerate(products):
            row_data = []
            for j, price_change in enumerate(price_changes):
                # Simulate win probability (decreases with price increases)
                base_prob = 0.6 + 0.1 * np.random.normal()
                win_prob = base_prob * (1 - price_change * 2)  # Simple elasticity
                win_prob = max(0.1, min(0.9, win_prob))  # Bound between 0.1 and 0.9
                row_data.append(win_prob)
            z_data.append(row_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f'{pc*100:.0f}%' for pc in price_changes],
            y=products,
            colorscale='RdYlGn',
            hovertemplate='Product: %{y}<br>Price Change: %{x}<br>Win Probability: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Win Probability by Product and Price Change',
            xaxis_title='Price Change (%)',
            yaxis_title='Product',
            height=500
        )
        
        return fig
    
    def _create_revenue_impact_dashboard(self, scenario_results: Dict[str, Any]) -> go.Figure:
        """Create revenue impact visualization"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Revenue Impact by Scenario', 'Risk-Return Profile'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sample scenario data
        scenarios = ['Conservative', 'Moderate', 'Aggressive']
        revenue_impact = [5, 12, 20]  # percentage increase
        risk_level = [2, 8, 15]  # risk score
        
        # Revenue impact bar chart
        fig.add_trace(
            go.Bar(x=scenarios, y=revenue_impact, name='Revenue Impact (%)',
                  marker_color=['green', 'orange', 'red']),
            row=1, col=1
        )
        
        # Risk-return scatter
        fig.add_trace(
            go.Scatter(x=risk_level, y=revenue_impact, mode='markers+text',
                      text=scenarios, textposition='top center',
                      marker=dict(size=15, color=['green', 'orange', 'red']),
                      name='Scenarios'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Revenue Impact Analysis")
        fig.update_xaxes(title_text="Risk Level", row=1, col=2)
        fig.update_yaxes(title_text="Revenue Impact (%)", row=1, col=2)
        
        return fig
    
    def _create_risk_return_dashboard(self, scenario_results: Dict[str, Any]) -> go.Figure:
        """Create risk-return analysis dashboard"""
        
        # Sample efficient frontier data
        risk_levels = np.linspace(0.05, 0.25, 20)
        expected_returns = 0.1 + 0.3 * risk_levels + np.random.normal(0, 0.01, len(risk_levels))
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(
            go.Scatter(x=risk_levels*100, y=expected_returns*100,
                      mode='lines+markers', name='Efficient Frontier',
                      line=dict(color='blue', width=2))
        )
        
        # Current position (example)
        fig.add_trace(
            go.Scatter(x=[12], y=[15], mode='markers',
                      marker=dict(size=15, color='red', symbol='star'),
                      name='Current Position')
        )
        
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Risk Level (% Volatility)',
            yaxis_title='Expected Return (%)',
            height=500
        )
        
        return fig
    
    def _create_sensitivity_analysis_dashboard(self, scenario_results: Dict[str, Any]) -> go.Figure:
        """Create sensitivity analysis dashboard"""
        
        # Sample sensitivity data
        parameters = ['Price_Product_A', 'Price_Product_B', 'Market_Conditions', 
                     'Competitive_Response', 'Economic_Factors']
        sensitivity_scores = [0.8, 0.6, 0.4, 0.7, 0.3]
        
        fig = go.Figure(data=go.Bar(
            y=parameters,
            x=sensitivity_scores,
            orientation='h',
            marker_color=['red' if x > 0.6 else 'orange' if x > 0.4 else 'green' 
                         for x in sensitivity_scores]
        ))
        
        fig.update_layout(
            title='Parameter Sensitivity Analysis',
            xaxis_title='Sensitivity Score',
            yaxis_title='Parameters',
            height=400
        )
        
        return fig
