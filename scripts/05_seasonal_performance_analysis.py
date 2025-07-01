#!/usr/bin/env python3
"""
Seasonal Discomfort Index Performance Analysis Script
- Analyze seasonal (spring/summer/autumn/winter) water level prediction performance of each discomfort index
- Analyze seasonal correlation trends
- Identify optimal seasonal discomfort indices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Font settings for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SeasonalPerformanceAnalyzer:
    def __init__(self, data_path, location_name):
        self.data_path = data_path
        self.location_name = location_name
        self.df = None
        self.seasonal_results = {}
        
    def load_data(self):
        """Load data and add seasonal information"""
        print(f"Loading data: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        # Add seasonal information
        self.df['month'] = self.df['datetime'].dt.month
        self.df['season'] = self.df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        print(f"Loaded data: {len(self.df)} rows")
        
        # Check seasonal data distribution
        season_counts = self.df['season'].value_counts()
        print("\nSeasonal data distribution:")
        for season, count in season_counts.items():
            percentage = count / len(self.df) * 100
            print(f"  {season}: {count} times ({percentage:.1f}%)")
    
    def get_discomfort_indices(self):
        """Return list of discomfort index columns"""
        return [col for col in self.df.columns 
                if any(idx in col for idx in [
                    'THI', 'heat_index', 'traditional_DI', 'apparent_temp',
                    'temp_humidity_composite', 'humidex', 'WBGT_simple',
                    'UTCI_simplified', 'effective_temperature', 'feels_like_temp'
                ])]
    
    def analyze_seasonal_correlations(self):
        """Analyze seasonal correlations"""
        discomfort_indices = self.get_discomfort_indices()
        seasons = ['spring', 'summer', 'autumn', 'winter']
        
        self.seasonal_results['correlations'] = {}
        
        print("\n=== SEASONAL CORRELATION ANALYSIS ===")
        
        for season in seasons:
            season_data = self.df[self.df['season'] == season].copy()
            if len(season_data) < 100:  # Check for sufficient data
                print(f"{season}: Insufficient data (n={len(season_data)})")
                continue
                
            season_correlations = {}
            
            for index_name in discomfort_indices:
                try:
                    # Remove missing values
                    mask = ~(season_data[index_name].isna() | season_data['water_level'].isna())
                    if mask.sum() < 50:  # Need at least 50 data points
                        continue
                        
                    correlation, p_value = pearsonr(
                        season_data.loc[mask, index_name], 
                        season_data.loc[mask, 'water_level']
                    )
                    
                    season_correlations[index_name] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'n_samples': mask.sum()
                    }
                    
                except Exception as e:
                    continue
            
            self.seasonal_results['correlations'][season] = season_correlations
            
            # Print top 3 indices
            if season_correlations:
                top_indices = sorted(season_correlations.items(), 
                                   key=lambda x: abs(x[1]['correlation']), reverse=True)[:3]
                print(f"\n{season.upper()} Top 3 Indices:")
                for idx_name, result in top_indices:
                    print(f"  {idx_name}: r={result['correlation']:.3f} (p={result['p_value']:.4f})")
    
    def analyze_seasonal_prediction_performance(self):
        """Analyze seasonal prediction performance"""
        discomfort_indices = self.get_discomfort_indices()
        seasons = ['spring', 'summer', 'autumn', 'winter']
        
        self.seasonal_results['prediction_performance'] = {}
        
        print("\n=== SEASONAL PREDICTION PERFORMANCE ANALYSIS ===")
        
        for season in seasons:
            season_data = self.df[self.df['season'] == season].copy()
            if len(season_data) < 200:  # Check for sufficient data
                print(f"{season}: Insufficient data (n={len(season_data)})")
                continue
            
            season_performance = {}
            
            for index_name in discomfort_indices:
                try:
                    # Remove missing values
                    mask = ~(season_data[index_name].isna() | season_data['water_level'].isna())
                    if mask.sum() < 100:  # Need at least 100 data points
                        continue
                    
                    X = season_data.loc[mask, [index_name]].values
                    y = season_data.loc[mask, 'water_level'].values
                    
                    # Split data (70% train, 30% test)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42)
                    
                    # Train Random Forest model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Predict and evaluate
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    season_performance[index_name] = {
                        'r2_score': r2,
                        'rmse': rmse,
                        'n_samples': mask.sum(),
                        'feature_importance': model.feature_importances_[0]
                    }
                    
                except Exception as e:
                    continue
            
            self.seasonal_results['prediction_performance'][season] = season_performance
            
            # Print top 3 indices
            if season_performance:
                top_indices = sorted(season_performance.items(), 
                                   key=lambda x: x[1]['r2_score'], reverse=True)[:3]
                print(f"\n{season.upper()} Top 3 Prediction Performance:")
                for idx_name, result in top_indices:
                    print(f"  {idx_name}: R²={result['r2_score']:.3f}, RMSE={result['rmse']:.3f}")
    
    def plot_seasonal_correlation_heatmap(self):
        """Plot seasonal correlation heatmap"""
        discomfort_indices = self.get_discomfort_indices()
        seasons = ['spring', 'summer', 'autumn', 'winter']
        
        # Build correlation matrix
        correlation_matrix = []
        index_labels = []
        
        for index_name in discomfort_indices:
            row = []
            index_labels.append(index_name.replace('_', ' ').title())
            for season in seasons:
                if (season in self.seasonal_results['correlations'] and 
                    index_name in self.seasonal_results['correlations'][season]):
                    corr = self.seasonal_results['correlations'][season][index_name]['correlation']
                    row.append(corr)
                else:
                    row.append(np.nan)
            correlation_matrix.append(row)
        
        correlation_df = pd.DataFrame(correlation_matrix, 
                                    index=index_labels, 
                                    columns=[s.title() for s in seasons])
        
        # Create heatmap
        plt.figure(figsize=(10, 12))
        mask = correlation_df.isna()
        
        sns.heatmap(correlation_df, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   mask=mask,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   fmt='.3f')
        
        plt.title(f'{self.location_name.title()} Seasonal Correlation with Water Level', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Discomfort Index', fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(f'results/plots/{self.location_name}_seasonal_correlation_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seasonal_performance_comparison(self):
        """Plot seasonal performance comparison"""
        seasons = ['spring', 'summer', 'autumn', 'winter']
        
        # Prepare data for plotting
        performance_data = []
        
        for season in seasons:
            if season not in self.seasonal_results['prediction_performance']:
                continue
                
            for index_name, perf in self.seasonal_results['prediction_performance'][season].items():
                performance_data.append({
                    'season': season.title(),
                    'index': index_name.replace('_', ' ').title(),
                    'r2_score': perf['r2_score'],
                    'rmse': perf['rmse']
                })
        
        if not performance_data:
            print("No performance data available for plotting")
            return
            
        performance_df = pd.DataFrame(performance_data)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # R² score comparison
        r2_pivot = performance_df.pivot(index='index', columns='season', values='r2_score')
        
        sns.heatmap(r2_pivot, 
                   annot=True, 
                   cmap='YlOrRd',
                   ax=ax1,
                   square=True,
                   linewidths=0.5,
                   fmt='.3f')
        
        ax1.set_title('R² Score by Season and Index', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Season', fontsize=12)
        ax1.set_ylabel('Discomfort Index', fontsize=12)
        ax1.tick_params(axis='x', rotation=0)
        ax1.tick_params(axis='y', rotation=0)
        
        # RMSE comparison
        rmse_pivot = performance_df.pivot(index='index', columns='season', values='rmse')
        
        sns.heatmap(rmse_pivot, 
                   annot=True, 
                   cmap='YlOrRd_r',
                   ax=ax2,
                   square=True,
                   linewidths=0.5,
                   fmt='.3f')
        
        ax2.set_title('RMSE by Season and Index', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('Discomfort Index', fontsize=12)
        ax2.tick_params(axis='x', rotation=0)
        ax2.tick_params(axis='y', rotation=0)
        
        plt.suptitle(f'{self.location_name.title()} Seasonal Performance Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(f'results/plots/{self.location_name}_seasonal_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seasonal_trends(self):
        """Plot seasonal trends of top performing indices"""
        
        # Find top 5 indices by average correlation across seasons
        all_correlations = {}
        
        for season in self.seasonal_results['correlations']:
            for index_name, result in self.seasonal_results['correlations'][season].items():
                if index_name not in all_correlations:
                    all_correlations[index_name] = []
                all_correlations[index_name].append(abs(result['correlation']))
        
        # Calculate average correlation for each index
        avg_correlations = {idx: np.mean(corrs) for idx, corrs in all_correlations.items()}
        top_indices = sorted(avg_correlations.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Plot trends
        seasons = ['spring', 'summer', 'autumn', 'winter']
        
        plt.figure(figsize=(12, 8))
        
        for idx_name, _ in top_indices:
            correlations = []
            for season in seasons:
                if (season in self.seasonal_results['correlations'] and 
                    idx_name in self.seasonal_results['correlations'][season]):
                    corr = self.seasonal_results['correlations'][season][idx_name]['correlation']
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)
            
            plt.plot(seasons, correlations, 
                    marker='o', linewidth=2, markersize=8,
                    label=idx_name.replace('_', ' ').title())
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Correlation with Water Level', fontsize=12)
        plt.title(f'{self.location_name.title()} Seasonal Correlation Trends (Top 5 Indices)', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'results/plots/{self.location_name}_seasonal_trends.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def identify_best_seasonal_indices(self):
        """Identify best discomfort indices for each season"""
        seasons = ['spring', 'summer', 'autumn', 'winter']
        best_indices = {}
        
        print("\n=== OPTIMAL SEASONAL DISCOMFORT INDICES ===")
        
        for season in seasons:
            if season in self.seasonal_results['correlations']:
                # Find index with highest absolute correlation
                best_index = max(self.seasonal_results['correlations'][season].items(),
                               key=lambda x: abs(x[1]['correlation']))
                
                best_indices[season] = {
                    'index': best_index[0],
                    'correlation': best_index[1]['correlation'],
                    'p_value': best_index[1]['p_value']
                }
                
                print(f"{season.title()}: {best_index[0]} (r={best_index[1]['correlation']:.3f})")
        
        return best_indices
    
    def generate_seasonal_report(self):
        """Generate seasonal analysis report"""
        best_indices = self.identify_best_seasonal_indices()
        
        report = {
            'location': self.location_name,
            'analysis_type': 'seasonal_performance',
            'best_seasonal_indices': best_indices,
            'seasonal_correlations': self.seasonal_results.get('correlations', {}),
            'seasonal_performance': self.seasonal_results.get('prediction_performance', {})
        }
        
        # Save report
        report_path = f'results/reports/seasonal_analysis_{self.location_name}.json'
        pd.Series(report).to_json(report_path, indent=2, force_ascii=False)
        
        return report
    
    def run_analysis(self):
        """Run complete seasonal analysis"""
        print(f"=== {self.location_name.upper()} SEASONAL ANALYSIS START ===")
        
        self.load_data()
        self.analyze_seasonal_correlations()
        self.analyze_seasonal_prediction_performance()
        self.plot_seasonal_correlation_heatmap()
        self.plot_seasonal_performance_comparison()
        self.plot_seasonal_trends()
        self.identify_best_seasonal_indices()
        report = self.generate_seasonal_report()
        
        print(f"=== {self.location_name.upper()} SEASONAL ANALYSIS COMPLETE ===")
        return report

def main():
    locations = ['cheongju', 'gadeok']
    
    for location in locations:
        data_path = f"data/processed/{location}_processed.csv"
        analyzer = SeasonalPerformanceAnalyzer(data_path, location)
        
        try:
            analyzer.run_analysis()
            print("=" * 60)
        except Exception as e:
            print(f"{location} analysis error: {e}")

if __name__ == "__main__":
    main() 