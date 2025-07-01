#!/usr/bin/env python3
"""
Hybrid Ensemble Analysis Script
- Analyze ensemble effects of multiple era-based discomfort indices
- Apply various ensemble techniques: stacking, voting, blending
- Test various ensemble strategies: era-based, complexity-based, performance-based combinations
- Search for optimal ensemble combinations and weights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Font settings for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class HybridEnsembleAnalyzer:
    def __init__(self, data_path, location_name):
        self.data_path = data_path
        self.location_name = location_name
        self.df = None
        self.discomfort_indices = []
        self.index_metadata = {}
        self.ensemble_results = {}
        
    def load_data(self):
        """Load data and set metadata"""
        print(f"Loading data: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        print(f"Loaded data: {len(self.df)} rows")
        
        # Find discomfort index columns
        self.discomfort_indices = [col for col in self.df.columns 
                                 if any(idx in col for idx in [
                                     'THI', 'heat_index', 'traditional_DI', 'apparent_temp',
                                     'temp_humidity_composite', 'humidex', 'WBGT_simple',
                                     'UTCI_simplified', 'effective_temperature', 'feels_like_temp'
                                 ])]
        
        print(f"Found discomfort indices: {len(self.discomfort_indices)}")
        
        # Set metadata (based on development year)
        self.index_metadata = {
            'effective_temperature': {'era': 'Early (1920s)', 'year': 1923, 'complexity': 1},
            'THI': {'era': 'Classical (1950s)', 'year': 1950, 'complexity': 1},
            'traditional_DI': {'era': 'Classical (1950s)', 'year': 1950, 'complexity': 1},
            'WBGT_simple': {'era': 'Classical (1950s)', 'year': 1956, 'complexity': 2},
            'humidex': {'era': 'Modern (1960s-1980s)', 'year': 1965, 'complexity': 2},
            'heat_index': {'era': 'Modern (1960s-1980s)', 'year': 1979, 'complexity': 4},
            'apparent_temp': {'era': 'Modern (1960s-1980s)', 'year': 1984, 'complexity': 3},
            'feels_like_temp': {'era': 'Contemporary (1990s-2010s)', 'year': 1990, 'complexity': 3},
            'UTCI_simplified': {'era': 'Contemporary (1990s-2010s)', 'year': 2012, 'complexity': 5},
            'temp_humidity_composite': {'era': 'Latest (2020s)', 'year': 2024, 'complexity': 1}
        }
        
    def prepare_data(self):
        """Prepare data for ensemble analysis"""
        # Remove rows with missing values
        feature_columns = self.discomfort_indices + ['water_level']
        self.df_clean = self.df[feature_columns].dropna()
        
        print(f"Data after removing missing values: {len(self.df_clean)} rows")
        
        if len(self.df_clean) < 1000:
            print("Warning: Insufficient data for ensemble analysis.")
            
    def analyze_individual_performance(self):
        """Analyze individual discomfort index performance"""
        print("\n=== Individual Discomfort Index Performance Analysis ===")
        
        individual_scores = {}
        
        X_all = self.df_clean[self.discomfort_indices]
        y = self.df_clean['water_level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for i, index_name in enumerate(self.discomfort_indices):
            try:
                # Single feature model
                X_single_train = X_train_scaled[:, i:i+1]
                X_single_test = X_test_scaled[:, i:i+1]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_single_train, y_train)
                
                y_pred = model.predict(X_single_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                individual_scores[index_name] = {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                }
                
                print(f"  {index_name}: R²={r2:.3f}, RMSE={rmse:.3f}")
                
            except Exception as e:
                print(f"  {index_name}: Analysis failed - {str(e)}")
                
        self.individual_scores = individual_scores
        return individual_scores
    
    def create_era_based_ensembles(self):
        """Create era-based ensembles"""
        print("\n=== Era-based Ensemble Analysis ===")
        
        # Group by era
        era_groups = {}
        for index_name in self.discomfort_indices:
            if index_name in self.index_metadata:
                era = self.index_metadata[index_name]['era']
                if era not in era_groups:
                    era_groups[era] = []
                era_groups[era].append(index_name)
        
        print("Era-based groups:")
        for era, indices in era_groups.items():
            print(f"  {era}: {len(indices)} indices")
        
        era_results = {}
        
        X = self.df_clean[self.discomfort_indices]
        y = self.df_clean['water_level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for era, indices in era_groups.items():
            if len(indices) < 2:  # Need at least 2 indices for ensemble
                continue
                
            try:
                # Select indices for this era
                era_indices_positions = [self.discomfort_indices.index(idx) for idx in indices]
                X_era_train = X_train_scaled[:, era_indices_positions]
                X_era_test = X_test_scaled[:, era_indices_positions]
                
                # Test multiple ensemble methods
                ensemble_methods = {
                    'voting': VotingRegressor([
                        (f'rf_{i}', RandomForestRegressor(n_estimators=50, random_state=42+i))
                        for i in range(len(indices))
                    ]),
                    'stacking': StackingRegressor([
                        (f'rf_{i}', RandomForestRegressor(n_estimators=50, random_state=42+i))
                        for i in range(len(indices))
                    ], final_estimator=LinearRegression()),
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
                }
                
                era_method_results = {}
                
                for method_name, model in ensemble_methods.items():
                    try:
                        model.fit(X_era_train, y_train)
                        y_pred = model.predict(X_era_test)
                        
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        era_method_results[method_name] = {
                            'r2': r2,
                            'rmse': rmse,
                            'mae': mae,
                            'n_features': len(indices)
                        }
                        
                    except Exception as e:
                        print(f"    {method_name} failed: {str(e)}")
                        continue
                
                era_results[era] = era_method_results
                
                # Output best performance
                if era_method_results:
                    best_method = max(era_method_results.items(), key=lambda x: x[1]['r2'])
                    print(f"  {era}: Best performance = {best_method[0]} (R²={best_method[1]['r2']:.3f})")
                
            except Exception as e:
                print(f"  {era} analysis failed: {str(e)}")
                continue
        
        self.era_results = era_results
        return era_results
    
    def create_complexity_based_ensembles(self):
        """Create complexity-based ensembles"""
        print("\n=== Complexity-based Ensemble Analysis ===")
        
        # Group by complexity
        complexity_groups = {}
        for index_name in self.discomfort_indices:
            if index_name in self.index_metadata:
                complexity = self.index_metadata[index_name]['complexity']
                if complexity not in complexity_groups:
                    complexity_groups[complexity] = []
                complexity_groups[complexity].append(index_name)
        
        print("Complexity-based groups:")
        for complexity, indices in sorted(complexity_groups.items()):
            print(f"  Complexity {complexity}: {len(indices)} indices")
        
        complexity_results = {}
        
        X = self.df_clean[self.discomfort_indices]
        y = self.df_clean['water_level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for complexity, indices in complexity_groups.items():
            if len(indices) < 2:
                continue
                
            try:
                complexity_indices_positions = [self.discomfort_indices.index(idx) for idx in indices]
                X_comp_train = X_train_scaled[:, complexity_indices_positions]
                X_comp_test = X_test_scaled[:, complexity_indices_positions]
                
                # Ensemble model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_comp_train, y_train)
                y_pred = model.predict(X_comp_test)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                complexity_results[complexity] = {
                    'r2': r2,
                    'rmse': rmse,
                    'n_features': len(indices),
                    'indices': indices
                }
                
                print(f"  Complexity {complexity}: R²={r2:.3f}, RMSE={rmse:.3f}")
                
            except Exception as e:
                print(f"  Complexity {complexity} analysis failed: {str(e)}")
                continue
        
        self.complexity_results = complexity_results
        return complexity_results
    
    def create_top_performer_ensembles(self):
        """Create ensembles of top performing indices"""
        print("\n=== Top Performer Ensemble Analysis ===")
        
        if not hasattr(self, 'individual_scores'):
            print("Individual performance analysis required.")
            return
        
        # Sort by performance
        sorted_indices = sorted(self.individual_scores.items(), 
                              key=lambda x: x[1]['r2'], reverse=True)
        
        top_performer_results = {}
        
        X = self.df_clean[self.discomfort_indices]
        y = self.df_clean['water_level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test ensembles with top N indices
        for n in [2, 3, 5, 7, len(self.discomfort_indices)]:
            if n > len(sorted_indices):
                continue
                
            top_n_indices = [item[0] for item in sorted_indices[:n]]
            
            try:
                indices_positions = [self.discomfort_indices.index(idx) for idx in top_n_indices]
                X_top_train = X_train_scaled[:, indices_positions]
                X_top_test = X_test_scaled[:, indices_positions]
                
                # Test multiple ensemble methods
                ensemble_methods = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'linear_blend': LinearRegression(),
                    'ridge_blend': Ridge(alpha=1.0),
                    'voting': VotingRegressor([
                        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                        ('svr', SVR(kernel='rbf')),
                        ('mlp', MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42))
                    ])
                }
                
                top_n_results = {}
                
                for method_name, model in ensemble_methods.items():
                    try:
                        model.fit(X_top_train, y_train)
                        y_pred = model.predict(X_top_test)
                        
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        top_n_results[method_name] = {
                            'r2': r2,
                            'rmse': rmse
                        }
                        
                    except Exception as e:
                        continue
                
                if top_n_results:
                    best_method = max(top_n_results.items(), key=lambda x: x[1]['r2'])
                    top_performer_results[f'top_{n}'] = {
                        'best_method': best_method[0],
                        'best_r2': best_method[1]['r2'],
                        'best_rmse': best_method[1]['rmse'],
                        'all_methods': top_n_results,
                        'indices': top_n_indices
                    }
                    
                    print(f"  Top {n}: {best_method[0]} (R²={best_method[1]['r2']:.3f})")
                
            except Exception as e:
                print(f"  Top {n} analysis failed: {str(e)}")
                continue
        
        self.top_performer_results = top_performer_results
        return top_performer_results
    
    def optimize_ensemble_weights(self):
        """Optimize ensemble weights"""
        print("\n=== Ensemble Weight Optimization ===")
        
        from scipy.optimize import minimize
        
        # Select top 5 indices
        if not hasattr(self, 'individual_scores'):
            return
        
        sorted_indices = sorted(self.individual_scores.items(), 
                              key=lambda x: x[1]['r2'], reverse=True)
        top_5_indices = [item[0] for item in sorted_indices[:5]]
        
        X = self.df_clean[top_5_indices]
        y = self.df_clean['water_level']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train individual models
        models = []
        predictions_train = []
        predictions_test = []
        
        for i in range(len(top_5_indices)):
            model = RandomForestRegressor(n_estimators=100, random_state=42+i)
            model.fit(X_train_scaled[:, i:i+1], y_train)
            models.append(model)
            
            pred_train = model.predict(X_train_scaled[:, i:i+1])
            pred_test = model.predict(X_test_scaled[:, i:i+1])
            
            predictions_train.append(pred_train)
            predictions_test.append(pred_test)
        
        predictions_train = np.array(predictions_train).T
        predictions_test = np.array(predictions_test).T
        
        # Weight optimization function
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.dot(predictions_train, weights)
            mse = mean_squared_error(y_train, ensemble_pred)
            return mse
        
        # Constraints: sum of weights = 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(top_5_indices))]
        
        # Initial weights (equal)
        initial_weights = np.ones(len(top_5_indices)) / len(top_5_indices)
        
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            optimal_weights = result.x / np.sum(result.x)
            
            # Test prediction with optimal weights
            ensemble_pred_test = np.dot(predictions_test, optimal_weights)
            
            r2_optimized = r2_score(y_test, ensemble_pred_test)
            rmse_optimized = np.sqrt(mean_squared_error(y_test, ensemble_pred_test))
            
            print("Optimal weights:")
            for idx, weight in zip(top_5_indices, optimal_weights):
                print(f"  {idx}: {weight:.3f}")
            
            print(f"Optimized ensemble performance: R²={r2_optimized:.3f}, RMSE={rmse_optimized:.3f}")
            
            # Compare with equal weights
            equal_weights = np.ones(len(top_5_indices)) / len(top_5_indices)
            ensemble_pred_equal = np.dot(predictions_test, equal_weights)
            r2_equal = r2_score(y_test, ensemble_pred_equal)
            
            print(f"Equal weights performance: R²={r2_equal:.3f}")
            print(f"Performance improvement: {r2_optimized - r2_equal:.4f}")
            
            self.optimal_weights = {
                'indices': top_5_indices,
                'weights': optimal_weights,
                'r2': r2_optimized,
                'rmse': rmse_optimized,
                'improvement': r2_optimized - r2_equal
            }
            
        except Exception as e:
            print(f"Weight optimization failed: {str(e)}")
    
    def plot_ensemble_performance_comparison(self):
        """Visualize ensemble performance comparison"""
        
        if not all(hasattr(self, attr) for attr in ['individual_scores', 'era_results', 'top_performer_results']):
            print("Required analysis results are missing.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'{self.location_name.title()} Ensemble Performance Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Individual vs Best Ensemble Performance
        ax1 = axes[0, 0]
        
        # Individual performance
        individual_r2 = [score['r2'] for score in self.individual_scores.values()]
        individual_names = [name.replace('_', ' ').title() for name in self.individual_scores.keys()]
        
        # Best ensemble performance
        if hasattr(self, 'top_performer_results'):
            ensemble_r2 = []
            ensemble_names = []
            for key, result in self.top_performer_results.items():
                ensemble_r2.append(result['best_r2'])
                ensemble_names.append(f"{key.replace('_', ' ').title()} Ensemble")
        
        y_pos_ind = np.arange(len(individual_names))
        y_pos_ens = np.arange(len(individual_names), len(individual_names) + len(ensemble_names))
        
        bars1 = ax1.barh(y_pos_ind, individual_r2, alpha=0.7, label='Individual', color='lightblue')
        if ensemble_r2:
            bars2 = ax1.barh(y_pos_ens, ensemble_r2, alpha=0.7, label='Ensemble', color='orange')
        
        ax1.set_yticks(list(y_pos_ind) + list(y_pos_ens))
        ax1.set_yticklabels(individual_names + ensemble_names)
        ax1.set_xlabel('R² Score')
        ax1.set_title('Individual vs Ensemble Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Era-based ensemble performance
        ax2 = axes[0, 1]
        
        if hasattr(self, 'era_results'):
            era_names = []
            era_best_r2 = []
            
            for era, methods in self.era_results.items():
                if methods:
                    best_method = max(methods.items(), key=lambda x: x[1]['r2'])
                    era_names.append(era.replace('(', '\n('))
                    era_best_r2.append(best_method[1]['r2'])
            
            if era_names:
                bars = ax2.bar(range(len(era_names)), era_best_r2, color=plt.cm.viridis(np.linspace(0, 1, len(era_names))))
                ax2.set_xticks(range(len(era_names)))
                ax2.set_xticklabels(era_names, rotation=45, ha='right')
                ax2.set_ylabel('R² Score')
                ax2.set_title('Era-based Ensemble Performance')
                ax2.grid(True, alpha=0.3)
                
                # Display values
                for bar, value in zip(bars, era_best_r2):
                    ax2.text(bar.get_x() + bar.get_width()/2, value + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Complexity-based performance
        ax3 = axes[1, 0]
        
        if hasattr(self, 'complexity_results'):
            complexity_levels = sorted(self.complexity_results.keys())
            complexity_r2 = [self.complexity_results[level]['r2'] for level in complexity_levels]
            
            bars = ax3.bar(complexity_levels, complexity_r2, color=plt.cm.plasma(np.linspace(0, 1, len(complexity_levels))))
            ax3.set_xlabel('Complexity Level')
            ax3.set_ylabel('R² Score')
            ax3.set_title('Complexity-based Ensemble Performance')
            ax3.grid(True, alpha=0.3)
            
            # Display values
            for bar, value in zip(bars, complexity_r2):
                ax3.text(bar.get_x() + bar.get_width()/2, value + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Ensemble size vs performance
        ax4 = axes[1, 1]
        
        if hasattr(self, 'top_performer_results'):
            ensemble_sizes = []
            ensemble_performances = []
            
            for key, result in self.top_performer_results.items():
                if 'top_' in key:
                    size = int(key.split('_')[1])
                    ensemble_sizes.append(size)
                    ensemble_performances.append(result['best_r2'])
            
            if ensemble_sizes:
                ax4.plot(ensemble_sizes, ensemble_performances, 'o-', linewidth=2, markersize=8)
                ax4.set_xlabel('Ensemble Size (Number of Indices)')
                ax4.set_ylabel('R² Score')
                ax4.set_title('Ensemble Size vs Performance')
                ax4.grid(True, alpha=0.3)
                
                # Mark best performance
                best_idx = np.argmax(ensemble_performances)
                ax4.scatter(ensemble_sizes[best_idx], ensemble_performances[best_idx], 
                          color='red', s=100, zorder=5)
                ax4.annotate(f'Best: {ensemble_performances[best_idx]:.3f}',
                           xy=(ensemble_sizes[best_idx], ensemble_performances[best_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{self.location_name}_ensemble_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_ensemble_report(self):
        """Generate ensemble analysis report"""
        
        # Find best overall ensemble
        best_overall = None
        best_r2 = 0
        
        # Extract best performance from each analysis
        all_results = []
        
        if hasattr(self, 'era_results'):
            for era, methods in self.era_results.items():
                for method, result in methods.items():
                    all_results.append({
                        'type': 'era_based',
                        'name': f"{era}_{method}",
                        'r2': result['r2'],
                        'rmse': result['rmse']
                    })
        
        if hasattr(self, 'top_performer_results'):
            for key, result in self.top_performer_results.items():
                all_results.append({
                    'type': 'top_performer',
                    'name': f"{key}_{result['best_method']}",
                    'r2': result['best_r2'],
                    'rmse': result['best_rmse']
                })
        
        if all_results:
            best_overall = max(all_results, key=lambda x: x['r2'])
        
        report = {
            'location': self.location_name,
            'analysis_type': 'hybrid_ensemble_analysis',
            'total_indices_tested': len(self.discomfort_indices),
            'individual_performance': self.individual_scores if hasattr(self, 'individual_scores') else {},
            'era_based_results': self.era_results if hasattr(self, 'era_results') else {},
            'complexity_based_results': self.complexity_results if hasattr(self, 'complexity_results') else {},
            'top_performer_results': self.top_performer_results if hasattr(self, 'top_performer_results') else {},
            'optimal_weights': self.optimal_weights if hasattr(self, 'optimal_weights') else {},
            'best_overall_ensemble': best_overall,
            'summary': {
                'best_individual_r2': max([score['r2'] for score in self.individual_scores.values()]) if hasattr(self, 'individual_scores') else 0,
                'best_ensemble_r2': best_overall['r2'] if best_overall else 0,
                'ensemble_improvement': (best_overall['r2'] - max([score['r2'] for score in self.individual_scores.values()])) if (best_overall and hasattr(self, 'individual_scores')) else 0
            }
        }
        
        # Save report
        import json
        with open(f'results/reports/ensemble_analysis_{self.location_name}.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        return report
    
    def run_analysis(self):
        """Run complete analysis"""
        print(f"=== {self.location_name.upper()} HYBRID ENSEMBLE ANALYSIS START ===")
        
        self.load_data()
        self.prepare_data()
        
        # Analysis sequence
        self.analyze_individual_performance()
        self.create_era_based_ensembles()
        self.create_complexity_based_ensembles()
        self.create_top_performer_ensembles()
        self.optimize_ensemble_weights()
        
        # Visualization and report
        self.plot_ensemble_performance_comparison()
        report = self.generate_ensemble_report()
        
        print(f"\n=== {self.location_name.upper()} HYBRID ENSEMBLE ANALYSIS COMPLETE ===")
        
        # Summary output
        if report['best_overall_ensemble']:
            best = report['best_overall_ensemble']
            print(f"\nBest Overall Ensemble:")
            print(f"  Type: {best['type']}")
            print(f"  Name: {best['name']}")
            print(f"  R²: {best['r2']:.4f}")
            print(f"  RMSE: {best['rmse']:.4f}")
            
        if 'ensemble_improvement' in report['summary']:
            improvement = report['summary']['ensemble_improvement']
            print(f"  Improvement over individual: {improvement:.4f}")
            
        return report

def main():
    """Main execution function"""
    locations = [
        ('data/processed/cheongju_processed.csv', 'cheongju'),
        ('data/processed/gadeok_processed.csv', 'gadeok')
    ]
    
    all_reports = {}
    
    for data_path, location_name in locations:
        try:
            analyzer = HybridEnsembleAnalyzer(data_path, location_name)
            report = analyzer.run_analysis()
            all_reports[location_name] = report
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"Error during {location_name} analysis: {str(e)}")
            continue
    
    # Output comprehensive comparison results
    print("=== COMPREHENSIVE ENSEMBLE ANALYSIS RESULTS ===")
    for location, report in all_reports.items():
        print(f"\n{location.upper()}:")
        if report['best_overall_ensemble']:
            best = report['best_overall_ensemble']
            print(f"  Best Ensemble: {best['name']} (R²={best['r2']:.4f})")
            
        summary = report['summary']
        print(f"  Best Individual: R²={summary['best_individual_r2']:.4f}")
        print(f"  Best Ensemble: R²={summary['best_ensemble_r2']:.4f}")
        print(f"  Improvement: +{summary['ensemble_improvement']:.4f}")

if __name__ == "__main__":
    main() 