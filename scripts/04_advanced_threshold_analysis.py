#!/usr/bin/env python3
"""
Advanced Threshold Analysis Script
- Set water level thresholds and evaluate prediction performance of each discomfort index
- Analyze prediction accuracy by flood risk level
- Generate ROC curves and precision-recall curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Font settings for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ThresholdAnalyzer:
    def __init__(self, data_path, location_name):
        self.data_path = data_path
        self.location_name = location_name
        self.df = None
        self.threshold_levels = {}
        self.results = {}
        
    def load_data(self):
        """Load data"""
        print(f"Loading data: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        print(f"Loaded data: {len(self.df)} rows")
        
    def define_water_level_thresholds(self):
        """Define water level thresholds"""
        water_level = self.df['water_level'].dropna()
        
        # Set thresholds based on percentiles
        self.threshold_levels = {
            'normal': water_level.quantile(0.7),      # 70th percentile
            'watch': water_level.quantile(0.85),     # 85th percentile  
            'warning': water_level.quantile(0.95),   # 95th percentile
            'emergency': water_level.quantile(0.99)  # 99th percentile
        }
        
        print("Water level thresholds:")
        for level, threshold in self.threshold_levels.items():
            print(f"  {level}: {threshold:.2f}m")
            
    def create_threshold_labels(self):
        """Create threshold-based labels"""
        def classify_water_level(level):
            if pd.isna(level):
                return 'unknown'
            elif level >= self.threshold_levels['emergency']:
                return 'emergency'
            elif level >= self.threshold_levels['warning']:
                return 'warning'
            elif level >= self.threshold_levels['watch']:
                return 'watch'
            else:
                return 'normal'
                
        self.df['water_level_category'] = self.df['water_level'].apply(classify_water_level)
        
        # Check distribution
        category_counts = self.df['water_level_category'].value_counts()
        print("\nWater level category distribution:")
        for category, count in category_counts.items():
            percentage = count / len(self.df) * 100
            print(f"  {category}: {count} times ({percentage:.1f}%)")
            
    def analyze_index_threshold_performance(self):
        """Analyze threshold prediction performance of each discomfort index"""
        
        # Find discomfort index columns
        discomfort_indices = [col for col in self.df.columns 
                            if any(idx in col for idx in [
                                'THI', 'heat_index', 'traditional_DI', 'apparent_temp',
                                'temp_humidity_composite', 'humidex', 'WBGT_simple',
                                'UTCI_simplified', 'effective_temperature', 'feels_like_temp'
                            ])]
        
        print(f"\nDiscomfort indices to analyze: {len(discomfort_indices)}")
        
        self.results = {}
        
        for threshold_type in ['warning', 'emergency']:
            print(f"\n=== {threshold_type.upper()} LEVEL ANALYSIS ===")
            
            # Convert to binary classification problem
            if threshold_type == 'warning':
                y = (self.df['water_level_category'].isin(['warning', 'emergency'])).astype(int)
            else:
                y = (self.df['water_level_category'] == 'emergency').astype(int)
            
            threshold_results = {}
            
            for index_name in discomfort_indices:
                try:
                    # Remove missing values
                    mask = ~(self.df[index_name].isna() | y.isna())
                    X = self.df.loc[mask, [index_name]].values
                    y_clean = y[mask].values
                    
                    if len(X) < 100 or sum(y_clean) < 10:  # Check for sufficient data and positive cases
                        continue
                        
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_clean, test_size=0.3, random_state=42, stratify=y_clean)
                    
                    # Train model
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
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
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    
                    threshold_results[index_name] = {
                        'roc_auc': roc_auc,
                        'pr_auc': pr_auc,
                        'precision': class_report['1']['precision'],
                        'recall': class_report['1']['recall'],
                        'f1_score': class_report['1']['f1-score'],
                        'fpr': fpr,
                        'tpr': tpr,
                        'precision_curve': precision,
                        'recall_curve': recall
                    }
                    
                    print(f"  {index_name}: ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, F1={class_report['1']['f1-score']:.3f}")
                    
                except Exception as e:
                    print(f"  {index_name}: Analysis failed - {str(e)}")
                    continue
            
            self.results[threshold_type] = threshold_results
            
    def plot_threshold_performance(self):
        """Visualize threshold prediction performance"""
        
        # Performance comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.location_name} Water Level Threshold Prediction Performance by Discomfort Index', fontsize=16, fontweight='bold')
        
        metrics = ['roc_auc', 'pr_auc', 'precision', 'f1_score']
        metric_names = ['ROC AUC', 'PR AUC', 'Precision', 'F1 Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            warning_scores = []
            emergency_scores = []
            index_names = []
            
            for index_name in self.results['warning'].keys():
                if index_name in self.results['emergency']:
                    warning_scores.append(self.results['warning'][index_name][metric])
                    emergency_scores.append(self.results['emergency'][index_name][metric])
                    index_names.append(index_name.replace('_', ' ').title())
            
            x = np.arange(len(index_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, warning_scores, width, label='Warning Level', alpha=0.8)
            bars2 = ax.bar(x + width/2, emergency_scores, width, label='Emergency Level', alpha=0.8)
            
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(index_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{self.location_name}_threshold_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self):
        """Plot ROC curves for different threshold levels"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results['warning'])))
        
        # Warning level ROC curves
        for i, (index_name, result) in enumerate(self.results['warning'].items()):
            ax1.plot(result['fpr'], result['tpr'], 
                    color=colors[i], linewidth=2,
                    label=f"{index_name.replace('_', ' ').title()} (AUC={result['roc_auc']:.3f})")
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Warning Level ROC Curves')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Emergency level ROC curves
        for i, (index_name, result) in enumerate(self.results['emergency'].items()):
            ax2.plot(result['fpr'], result['tpr'], 
                    color=colors[i], linewidth=2,
                    label=f"{index_name.replace('_', ' ').title()} (AUC={result['roc_auc']:.3f})")
        
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Emergency Level ROC Curves')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{self.location_name}_roc_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_threshold_report(self):
        """Generate threshold analysis report"""
        
        # Find best performing index for each metric and threshold level
        best_performers = {}
        
        for threshold_type in ['warning', 'emergency']:
            best_performers[threshold_type] = {}
            
            for metric in ['roc_auc', 'pr_auc', 'precision', 'f1_score']:
                best_index = max(self.results[threshold_type].items(), 
                               key=lambda x: x[1][metric])
                best_performers[threshold_type][f'best_{metric}'] = {
                    'index': best_index[0],
                    'score': best_index[1][metric]
                }
        
        # Save report
        report = {
            'location': self.location_name,
            'threshold_levels': self.threshold_levels,
            'best_performers': best_performers,
            'detailed_results': self.results
        }
        
        report_path = f'results/reports/threshold_analysis_{self.location_name}.json'
        pd.Series(report).to_json(report_path, indent=2, force_ascii=False)
        
        # Print summary
        print(f"\n{self.location_name.upper()} Best Performing Indices:")
        for threshold_type in ['warning', 'emergency']:
            print(f"  {threshold_type.title()} Level:")
            for metric_key, result in best_performers[threshold_type].items():
                metric_name = metric_key.replace('best_', '').upper()
                print(f"    {metric_name}: {result['index']} ({result['score']:.3f})")
        
        return report
    
    def run_analysis(self):
        """Run complete threshold analysis"""
        print(f"=== {self.location_name.upper()} THRESHOLD ANALYSIS START ===")
        
        self.load_data()
        self.define_water_level_thresholds()
        self.create_threshold_labels()
        self.analyze_index_threshold_performance()
        self.plot_threshold_performance()
        self.plot_roc_curves()
        report = self.generate_threshold_report()
        
        print(f"=== {self.location_name.upper()} THRESHOLD ANALYSIS COMPLETE ===")
        return report

def main():
    locations = ['cheongju', 'gadeok']
    all_reports = {}
    
    for location in locations:
        data_path = f"data/processed/{location}_processed.csv"
        analyzer = ThresholdAnalyzer(data_path, location)
        
        try:
            report = analyzer.run_analysis()
            all_reports[location] = report
            print("=" * 60)
        except Exception as e:
            print(f"{location} analysis error: {e}")
    
    # Print comprehensive analysis results
    print("=== COMPREHENSIVE THRESHOLD ANALYSIS RESULTS ===")
    
    for location, report in all_reports.items():
        if 'best_performers' in report:
            print(f"\n{location.upper()} Best Performing Indices:")
            for threshold_type in ['warning', 'emergency']:
                if threshold_type in report['best_performers']:
                    print(f"  {threshold_type.title()} Level:")
                    for metric_key, result in report['best_performers'][threshold_type].items():
                        metric_name = metric_key.replace('best_', '').upper()
                        print(f"    {metric_name}: {result['index']} ({result['score']:.3f})")

if __name__ == "__main__":
    main() 