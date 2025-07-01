#!/usr/bin/env python3
"""
Extreme Event Analysis Script
- Analyze discomfort index prediction performance during extreme water level situations such as floods
- Apply Extreme Value Theory (EVT) analysis
- Detect anomalies and set extreme value thresholds
- Evaluate extreme event prediction model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Font settings for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ExtremeEventAnalyzer:
    def __init__(self, data_path, location_name):
        self.data_path = data_path
        self.location_name = location_name
        self.df = None
        self.extreme_thresholds = {}
        self.extreme_events = None
        self.results = {}
        
    def load_data(self):
        """Load data"""
        print(f"Loading data: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        print(f"Loaded data: {len(self.df)} rows")
        
    def identify_extreme_events(self):
        """Identify extreme events"""
        water_level = self.df['water_level'].dropna()
        
        # 1. Statistical approach - 3 sigma rule
        mean_level = water_level.mean()
        std_level = water_level.std()
        sigma_3_threshold = mean_level + 3 * std_level
        
        # 2. Percentile approach
        p95_threshold = water_level.quantile(0.95)
        p99_threshold = water_level.quantile(0.99)
        p99_9_threshold = water_level.quantile(0.999)
        
        # 3. Extreme Value Theory approach - POT (Peaks Over Threshold)
        # Set POT threshold based on 95th percentile
        pot_threshold = water_level.quantile(0.95)
        exceedances = water_level[water_level > pot_threshold] - pot_threshold
        
        # GPD (Generalized Pareto Distribution) fitting
        try:
            gpd_params = stats.genpareto.fit(exceedances, floc=0)
            # Estimate 50-year return level (assuming daily data)
            return_level_50yr = pot_threshold + stats.genpareto.ppf(
                1 - 1/(50*365), *gpd_params)
        except:
            return_level_50yr = p99_9_threshold
            
        self.extreme_thresholds = {
            '3_sigma': sigma_3_threshold,
            'p95': p95_threshold,
            'p99': p99_threshold,
            'p99.9': p99_9_threshold,
            'pot_95': pot_threshold,
            '50yr_return': return_level_50yr
        }
        
        print("Extreme value thresholds:")
        for method, threshold in self.extreme_thresholds.items():
            count = (water_level >= threshold).sum()
            percentage = count / len(water_level) * 100
            print(f"  {method}: {threshold:.2f}m ({count} times, {percentage:.2f}%)")
            
        # Label extreme events (using 99th percentile)
        self.df['is_extreme'] = (self.df['water_level'] >= self.extreme_thresholds['p99']).astype(int)
        
        extreme_count = self.df['is_extreme'].sum()
        print(f"\nExtreme event count: {extreme_count} times ({extreme_count/len(self.df)*100:.2f}%)")
        
    def analyze_extreme_event_patterns(self):
        """Analyze extreme event patterns"""
        extreme_events = self.df[self.df['is_extreme'] == 1].copy()
        
        if len(extreme_events) < 10:
            print("Too few extreme events (minimum 10 required)")
            return
            
        # Seasonal distribution
        extreme_events['month'] = extreme_events['datetime'].dt.month
        extreme_events['season'] = extreme_events['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        print("\n=== EXTREME EVENT PATTERN ANALYSIS ===")
        
        # Seasonal distribution
        seasonal_dist = extreme_events['season'].value_counts()
        print("\nSeasonal extreme event distribution:")
        for season, count in seasonal_dist.items():
            percentage = count / len(extreme_events) * 100
            print(f"  {season}: {count} times ({percentage:.1f}%)")
            
        # Monthly distribution
        monthly_dist = extreme_events['month'].value_counts().sort_index()
        print("\nMonthly extreme event distribution:")
        for month, count in monthly_dist.items():
            percentage = count / len(extreme_events) * 100
            print(f"  {month}월: {count} times ({percentage:.1f}%)")
            
        self.extreme_events = extreme_events
        
    def analyze_extreme_prediction_performance(self):
        """Analyze extreme event prediction performance"""
        
        # Find discomfort index columns
        discomfort_indices = [col for col in self.df.columns 
                            if any(idx in col for idx in [
                                'THI', 'heat_index', 'traditional_DI', 'apparent_temp',
                                'temp_humidity_composite', 'humidex', 'WBGT_simple',
                                'UTCI_simplified', 'effective_temperature', 'feels_like_temp'
                            ])]
        
        print(f"\n=== EXTREME EVENT PREDICTION PERFORMANCE ANALYSIS ===")
        print(f"Discomfort indices to analyze: {len(discomfort_indices)}")
        
        self.results = {}
        
        for index_name in discomfort_indices:
            try:
                # Remove missing values
                mask = ~(self.df[index_name].isna() | self.df['is_extreme'].isna())
                X = self.df.loc[mask, [index_name]].values
                y = self.df.loc[mask, 'is_extreme'].values
                
                if len(X) < 100 or sum(y) < 5:  # Check for sufficient data and extreme cases
                    continue
                    
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y)
                
                # Data scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model (use class_weight for imbalanced extreme event data)
                clf = RandomForestClassifier(
                    n_estimators=100, 
                    class_weight='balanced',
                    random_state=42
                )
                clf.fit(X_train_scaled, y_train)
                
                # Predict and evaluate
                y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
                y_pred = clf.predict(X_test_scaled)
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                # Classification report
                if sum(y_test) > 0:  # Only if extreme values exist in test set
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    
                    self.results[index_name] = {
                        'roc_auc': roc_auc,
                        'pr_auc': pr_auc,
                        'precision': class_report.get('1', {}).get('precision', 0),
                        'recall': class_report.get('1', {}).get('recall', 0),
                        'f1_score': class_report.get('1', {}).get('f1-score', 0),
                        'fpr': fpr,
                        'tpr': tpr,
                        'precision_curve': precision,
                        'recall_curve': recall,
                        'feature_importance': clf.feature_importances_[0],
                        'n_samples': len(X),
                        'n_extreme_events': sum(y)
                    }
                    
                    print(f"  {index_name}: ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, "
                          f"F1={class_report.get('1', {}).get('f1-score', 0):.3f}")
                    
            except Exception as e:
                print(f"  {index_name}: Analysis failed - {str(e)}")
                continue
                
    def analyze_extreme_with_isolation_forest(self):
        """Extreme detection using Isolation Forest"""
        
        # Use all discomfort indices for anomaly detection
        discomfort_indices = [col for col in self.df.columns 
                            if any(idx in col for idx in [
                                'THI', 'heat_index', 'traditional_DI', 'apparent_temp',
                                'temp_humidity_composite', 'humidex', 'WBGT_simple',
                                'UTCI_simplified', 'effective_temperature', 'feels_like_temp'
                            ])]
        
        # Prepare data
        feature_data = self.df[discomfort_indices + ['water_level']].dropna()
        
        if len(feature_data) < 1000:
            print("Insufficient data for Isolation Forest analysis")
            return
            
        # Apply Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.01,  # Expect 1% anomalies
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = iso_forest.fit_predict(feature_data)
        
        # -1: anomaly, 1: normal
        anomaly_count = (anomaly_labels == -1).sum()
        
        print(f"\n=== ISOLATION FOREST EXTREME DETECTION ===")
        print(f"Isolation Forest detected anomalies: {anomaly_count} ({anomaly_count/len(feature_data)*100:.2f}%)")
        
        # Compare with actual extreme events
        actual_extreme_mask = feature_data.index.isin(
            self.df[self.df['is_extreme'] == 1].index
        )
        
        detected_anomalies = anomaly_labels == -1
        
        # Calculate overlap
        overlap = (actual_extreme_mask & detected_anomalies).sum()
        print(f"Overlap with actual extreme events: {overlap}/{anomaly_count} "
              f"({overlap/anomaly_count*100:.1f}%)")
    
    def plot_extreme_analysis_results(self):
        """Visualize extreme analysis results"""
        
        if not self.results:
            print("No results available for plotting")
            return
            
        # Performance comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.location_name.title()} Extreme Event Prediction Performance', 
                     fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        indices = list(self.results.keys())
        index_labels = [idx.replace('_', ' ').title() for idx in indices]
        
        roc_aucs = [self.results[idx]['roc_auc'] for idx in indices]
        pr_aucs = [self.results[idx]['pr_auc'] for idx in indices]
        precisions = [self.results[idx]['precision'] for idx in indices]
        f1_scores = [self.results[idx]['f1_score'] for idx in indices]
        
        # ROC AUC
        bars1 = ax1.bar(range(len(indices)), roc_aucs, alpha=0.8)
        ax1.set_title('ROC AUC Scores', fontweight='bold')
        ax1.set_xlabel('Discomfort Index')
        ax1.set_ylabel('ROC AUC')
        ax1.set_xticks(range(len(indices)))
        ax1.set_xticklabels(index_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, roc_aucs):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # PR AUC
        bars2 = ax2.bar(range(len(indices)), pr_aucs, alpha=0.8, color='orange')
        ax2.set_title('PR AUC Scores', fontweight='bold')
        ax2.set_xlabel('Discomfort Index')
        ax2.set_ylabel('PR AUC')
        ax2.set_xticks(range(len(indices)))
        ax2.set_xticklabels(index_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, pr_aucs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Precision
        bars3 = ax3.bar(range(len(indices)), precisions, alpha=0.8, color='green')
        ax3.set_title('Precision Scores', fontweight='bold')
        ax3.set_xlabel('Discomfort Index')
        ax3.set_ylabel('Precision')
        ax3.set_xticks(range(len(indices)))
        ax3.set_xticklabels(index_labels, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, precisions):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # F1 Score
        bars4 = ax4.bar(range(len(indices)), f1_scores, alpha=0.8, color='red')
        ax4.set_title('F1 Scores', fontweight='bold')
        ax4.set_xlabel('Discomfort Index')
        ax4.set_ylabel('F1 Score')
        ax4.set_xticks(range(len(indices)))
        ax4.set_xticklabels(index_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, f1_scores):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{self.location_name}_extreme_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_extreme_events_distribution(self):
        """Plot extreme events distribution"""
        
        if self.extreme_events is None or len(self.extreme_events) == 0:
            print("No extreme events data available for plotting")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.location_name.title()} Extreme Events Distribution Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Seasonal distribution
        seasonal_counts = self.extreme_events['season'].value_counts()
        colors_seasonal = plt.cm.Set3(np.linspace(0, 1, len(seasonal_counts)))
        
        wedges, texts, autotexts = ax1.pie(seasonal_counts.values, 
                                          labels=seasonal_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors_seasonal,
                                          startangle=90)
        ax1.set_title('Seasonal Distribution', fontweight='bold')
        
        # Monthly distribution
        monthly_counts = self.extreme_events['month'].value_counts().sort_index()
        ax2.bar(monthly_counts.index, monthly_counts.values, alpha=0.8)
        ax2.set_title('Monthly Distribution', fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Events')
        ax2.grid(True, alpha=0.3)
        
        # Water level distribution
        ax3.hist(self.df['water_level'].dropna(), bins=50, alpha=0.6, 
                label='All Data', density=True)
        ax3.hist(self.extreme_events['water_level'], bins=20, alpha=0.8, 
                label='Extreme Events', density=True)
        ax3.axvline(self.extreme_thresholds['p99'], color='red', linestyle='--', 
                   label=f"99th Percentile ({self.extreme_thresholds['p99']:.2f}m)")
        ax3.set_title('Water Level Distribution', fontweight='bold')
        ax3.set_xlabel('Water Level (m)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Time series of extreme events
        extreme_dates = pd.to_datetime(self.extreme_events['datetime'])
        extreme_years = extreme_dates.dt.year.value_counts().sort_index()
        
        ax4.bar(extreme_years.index, extreme_years.values, alpha=0.8)
        ax4.set_title('Annual Extreme Events', fontweight='bold')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Events')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{self.location_name}_extreme_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves_extreme(self):
        """Plot ROC curves for extreme event prediction"""
        
        if not self.results:
            print("No results available for ROC curve plotting")
            return
            
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))
        
        for i, (index_name, result) in enumerate(self.results.items()):
            plt.plot(result['fpr'], result['tpr'], 
                    color=colors[i], linewidth=2,
                    label=f"{index_name.replace('_', ' ').title()} (AUC={result['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{self.location_name.title()} Extreme Event Prediction - ROC Curves', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{self.location_name}_extreme_roc_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_extreme_report(self):
        """Generate extreme event analysis report"""
        
        # Find best performing indices
        best_performers = {}
        
        if self.results:
            for metric in ['roc_auc', 'pr_auc', 'precision', 'f1_score']:
                best_index = max(self.results.items(), key=lambda x: x[1][metric])
                best_performers[f'best_{metric}'] = {
                    'index': best_index[0],
                    'score': best_index[1][metric]
                }
        
        report = {
            'location': self.location_name,
            'analysis_type': 'extreme_event_analysis',
            'extreme_thresholds': self.extreme_thresholds,
            'extreme_event_count': int(self.df['is_extreme'].sum()),
            'extreme_event_percentage': float(self.df['is_extreme'].mean() * 100),
            'best_performers': best_performers,
            'detailed_results': self.results
        }
        
        # Add extreme event patterns if available
        if self.extreme_events is not None and len(self.extreme_events) > 0:
            seasonal_dist = self.extreme_events['season'].value_counts().to_dict()
            monthly_dist = self.extreme_events['month'].value_counts().to_dict()
            
            report['extreme_patterns'] = {
                'seasonal_distribution': seasonal_dist,
                'monthly_distribution': monthly_dist
            }
        
        # Save report
        report_path = f'results/reports/extreme_analysis_{self.location_name}.json'
        pd.Series(report).to_json(report_path, indent=2, force_ascii=False)
        
        # Print summary
        print(f"\n{self.location_name.upper()} Extreme Analysis Summary:")
        print(f"  Extreme events: {report['extreme_event_count']} ({report['extreme_event_percentage']:.2f}%)")
        
        if best_performers:
            print("  Best performing indices:")
            for metric_key, result in best_performers.items():
                metric_name = metric_key.replace('best_', '').upper()
                print(f"    {metric_name}: {result['index']} ({result['score']:.3f})")
        
        return report
        
    def run_analysis(self):
        """Run complete extreme event analysis"""
        print(f"=== {self.location_name.upper()} EXTREME ANALYSIS START ===")
        
        self.load_data()
        self.identify_extreme_events()
        self.analyze_extreme_event_patterns()
        self.analyze_extreme_prediction_performance()
        self.analyze_extreme_with_isolation_forest()
        self.plot_extreme_analysis_results()
        self.plot_extreme_events_distribution()
        self.plot_roc_curves_extreme()
        report = self.generate_extreme_report()
        
        print(f"=== {self.location_name.upper()} EXTREME ANALYSIS COMPLETE ===")
        return report

def main():
    locations = ['cheongju', 'gadeok']
    all_reports = {}
    
    for location in locations:
        data_path = f"data/processed/{location}_processed.csv"
        analyzer = ExtremeEventAnalyzer(data_path, location)
        
        try:
            report = analyzer.run_analysis()
            all_reports[location] = report
            
            # Print top 3 performers for this location
            if 'detailed_results' in report and report['detailed_results']:
                print(f"\nExtreme prediction performance top 3 indices:")
                sorted_results = sorted(report['detailed_results'].items(), 
                                     key=lambda x: x[1]['roc_auc'], reverse=True)[:3]
                for idx_name, result in sorted_results:
                    print(f"  {idx_name}: ROC-AUC={result['roc_auc']:.3f}, PR-AUC={result['pr_auc']:.3f}")
            
            print("=" * 60)
        except Exception as e:
            print(f"{location} analysis error: {e}")
    
    # Print comprehensive analysis results
    print("=== COMPREHENSIVE EXTREME ANALYSIS RESULTS ===")
    
    for location, report in all_reports.items():
        if 'best_performers' in report and report['best_performers']:
            print(f"\n{location.upper()}:")
            print(f"  Extreme events: {report.get('extreme_event_count', 0)} "
                  f"({report.get('extreme_event_percentage', 0):.2f}%)")
            print("  Best performing indices:")
            for metric_key, result in report['best_performers'].items():
                metric_name = metric_key.replace('best_', '').upper()
                print(f"    {metric_name}: {result['index']} ({result['score']:.3f})")

if __name__ == "__main__":
    main() 