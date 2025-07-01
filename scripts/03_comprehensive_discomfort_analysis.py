#!/usr/bin/env python3
"""
무심천 종합 불쾌지수 분석
- 10개 불쾌지수 계산 및 성능 비교
- 시대별 개발 순서 vs 수위 예측 성능 분석
- 복잡도 vs 성능 관계 분석
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import json
import os

warnings.filterwarnings('ignore')

class ComprehensiveDiscomfortAnalyzer:
    """종합 불쾌지수 분석기"""
    
    # 각 불쾌지수의 메타데이터
    INDEX_METADATA = {
        'effective_temperature': {
            'name': 'Effective Temperature',
            'development_year': 1923,
            'complexity': 'Simple',
            'formula_complexity_score': 1,
            'primary_use': 'Early comfort assessment',
            'variables_used': ['temperature', 'humidity'],
            'era': 'Early (1920s)'
        },
        'THI': {
            'name': 'Temperature-Humidity Index',
            'development_year': 1950,
            'complexity': 'Simple',
            'formula_complexity_score': 1,
            'primary_use': 'General comfort assessment',
            'variables_used': ['temperature', 'humidity'],
            'era': 'Classical (1950s)'
        },
        'traditional_DI': {
            'name': 'Traditional Discomfort Index',
            'development_year': 1950,
            'complexity': 'Simple', 
            'formula_complexity_score': 1,
            'primary_use': 'Basic thermal comfort',
            'variables_used': ['temperature', 'humidity'],
            'era': 'Classical (1950s)'
        },
        'WBGT_simple': {
            'name': 'Wet Bulb Globe Temperature (Simplified)',
            'development_year': 1956,
            'complexity': 'Moderate',
            'formula_complexity_score': 2,
            'primary_use': 'Military and sports safety',
            'variables_used': ['temperature', 'humidity'],
            'era': 'Classical (1950s)'
        },
        'humidex': {
            'name': 'Humidity Index (Canadian)',
            'development_year': 1965,
            'complexity': 'Moderate',
            'formula_complexity_score': 2,
            'primary_use': 'Canadian weather forecasting',
            'variables_used': ['temperature', 'humidity'],
            'era': 'Modern (1960s-1980s)'
        },
        'heat_index': {
            'name': 'Heat Index (Apparent Temperature)',
            'development_year': 1979,
            'complexity': 'Complex',
            'formula_complexity_score': 4,
            'primary_use': 'US weather forecasting',
            'variables_used': ['temperature', 'humidity'],
            'era': 'Modern (1960s-1980s)'
        },
        'apparent_temp': {
            'name': 'Apparent Temperature (Australian)',
            'development_year': 1984,
            'complexity': 'Moderate',
            'formula_complexity_score': 3,
            'primary_use': 'Australian weather services',
            'variables_used': ['temperature', 'humidity', 'wind_speed'],
            'era': 'Modern (1960s-1980s)'
        },
        'feels_like_temp': {
            'name': 'Feels Like Temperature',
            'development_year': 1990,
            'complexity': 'Moderate',
            'formula_complexity_score': 3,
            'primary_use': 'Modern weather apps',
            'variables_used': ['temperature', 'humidity', 'wind_speed'],
            'era': 'Contemporary (1990s-2010s)'
        },
        'UTCI_simplified': {
            'name': 'Universal Thermal Climate Index (Simplified)',
            'development_year': 2012,
            'complexity': 'Very Complex',
            'formula_complexity_score': 5,
            'primary_use': 'International climate assessment',
            'variables_used': ['temperature', 'humidity', 'wind_speed'],
            'era': 'Contemporary (1990s-2010s)'
        },
        'temp_humidity_composite': {
            'name': 'Temperature-Humidity Composite',
            'development_year': 2024,
            'complexity': 'Simple',
            'formula_complexity_score': 1,
            'primary_use': 'Water level prediction research',
            'variables_used': ['temperature', 'humidity'],
            'era': 'Latest (2020s)'
        }
    }
    
    def __init__(self):
        """초기화"""
        self.data_paths = {
            'cheongju_raw': '../../청주금천_기상수위_통합_2014_2023.csv',
            'gadeok_raw': '../../가덕_기상수위_통합_2014_2023.csv'
        }
        self.results = {}
    
    def calculate_effective_temperature(self, temp, humidity):
        """유효온도 - 초기 열환경 지표 (1923년)"""
        try:
            return temp - (100 - humidity) / 4
        except:
            return np.nan
    
    def calculate_THI(self, temp, humidity):
        """Temperature-Humidity Index (1950년)"""
        try:
            return (1.8 * temp + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * temp - 26))
        except:
            return np.nan
    
    def calculate_traditional_DI(self, temp, humidity):
        """전통적 불쾌지수 (1950년)"""
        try:
            return temp + 0.36 * humidity
        except:
            return np.nan
    
    def calculate_WBGT_simple(self, temp, humidity):
        """WBGT 간소화 공식 (1956년)"""
        try:
            # 습구온도 근사 계산
            wet_bulb = temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + \
                      np.arctan(temp + humidity) - np.arctan(humidity - 1.676331) + \
                      0.00391838 * np.power(humidity, 1.5) * np.arctan(0.023101 * humidity) - 4.686035
            
            return 0.7 * wet_bulb + 0.3 * temp
        except:
            return np.nan
    
    def calculate_humidex(self, temp, humidity):
        """Humidex - 캐나다 기상청 공식 (1965년)"""
        try:
            # 이슬점 온도 계산
            a, b = 17.27, 237.7
            alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
            dewpoint = (b * alpha) / (a - alpha)
            
            # Humidex 계산
            e = 6.11 * np.exp(5417.753 * ((1/273.16) - (1/(dewpoint + 273.16))))
            h = 0.5555 * (e - 10.0)
            
            return temp + h
        except:
            return np.nan
    
    def calculate_heat_index(self, temp, humidity):
        """Heat Index - 미국 기상청 공식 (1979년)"""
        try:
            T = temp * 9/5 + 32
            RH = humidity
            
            HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
            
            if HI >= 80:
                HI = (-42.379 + 2.04901523*T + 10.14333127*RH 
                      - 0.22475541*T*RH - 6.83783e-3*T**2 
                      - 5.481717e-2*RH**2 + 1.22874e-3*T**2*RH 
                      + 8.5282e-4*T*RH**2 - 1.99e-6*T**2*RH**2)
            
            return (HI - 32) * 5/9
        except:
            return np.nan
    
    def calculate_apparent_temp(self, temp, humidity, wind_speed=5.0):
        """체감온도 - 호주 기상청 공식 (1984년)"""
        try:
            e = humidity / 100 * 6.105 * np.exp(17.27 * temp / (237.7 + temp))
            return temp + 0.33 * e - 0.7 * wind_speed - 4.0
        except:
            return np.nan
    
    def calculate_feels_like_temp(self, temp, humidity, wind_speed=5.0):
        """체감온도 - 현대적 버전 (1990년대)"""
        try:
            if temp >= 26.7:
                return self.calculate_heat_index(temp, humidity)
            elif temp <= 10.0 and wind_speed > 4.8:
                return 13.12 + 0.6215*temp - 11.37*(wind_speed**0.16) + 0.3965*temp*(wind_speed**0.16)
            else:
                return temp + 0.3 * (humidity - 60) / 40 - 0.7 * wind_speed / 10
        except:
            return np.nan
    
    def calculate_UTCI_simplified(self, temp, humidity, wind_speed=5.0):
        """UTCI 간소화 근사 공식 (2012년)"""
        try:
            RH = humidity / 100.0
            v = max(wind_speed, 0.5)
            
            utci = temp
            if temp > 9:
                utci += (1.8 * temp - 32.0) * RH * 0.1
            utci -= (37 - temp) / (37 - temp + 68) * (1.76 + 1.4 * v**0.75) * 0.1
            
            return utci
        except:
            return np.nan
    
    def calculate_temp_humidity_composite(self, temp, humidity):
        """온습도 복합 지표 - 기존 연구 효과 지표 (2024년)"""
        try:
            return 0.7 * temp + 0.3 * humidity
        except:
            return np.nan
    
    def load_and_calculate_indices(self, location):
        """데이터 로딩 및 모든 불쾌지수 계산"""
        print(f"\n{location} 데이터 처리 중...")
        
        # 데이터 로딩
        file_path = self.data_paths[f'{location}_raw']
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 결측값이 있는 행 제거
        df_clean = df.dropna(subset=['temperature', 'humidity', 'water_level']).copy()
        
        print(f"원본 데이터: {len(df):,}행, 정제 후: {len(df_clean):,}행")
        
        # 모든 불쾌지수 계산
        print("불쾌지수 계산 중...")
        
        df_clean['effective_temperature'] = df_clean.apply(
            lambda row: self.calculate_effective_temperature(row['temperature'], row['humidity']), axis=1)
        
        df_clean['THI'] = df_clean.apply(
            lambda row: self.calculate_THI(row['temperature'], row['humidity']), axis=1)
        
        df_clean['traditional_DI'] = df_clean.apply(
            lambda row: self.calculate_traditional_DI(row['temperature'], row['humidity']), axis=1)
        
        df_clean['WBGT_simple'] = df_clean.apply(
            lambda row: self.calculate_WBGT_simple(row['temperature'], row['humidity']), axis=1)
        
        df_clean['humidex'] = df_clean.apply(
            lambda row: self.calculate_humidex(row['temperature'], row['humidity']), axis=1)
        
        df_clean['heat_index'] = df_clean.apply(
            lambda row: self.calculate_heat_index(row['temperature'], row['humidity']), axis=1)
        
        # 풍속 고려 지수들
        wind_speed_col = df_clean['wind_speed'].fillna(5.0)
        
        df_clean['apparent_temp'] = df_clean.apply(
            lambda row: self.calculate_apparent_temp(
                row['temperature'], row['humidity'], row.get('wind_speed', 5.0)), axis=1)
        
        df_clean['feels_like_temp'] = df_clean.apply(
            lambda row: self.calculate_feels_like_temp(
                row['temperature'], row['humidity'], row.get('wind_speed', 5.0)), axis=1)
        
        df_clean['UTCI_simplified'] = df_clean.apply(
            lambda row: self.calculate_UTCI_simplified(
                row['temperature'], row['humidity'], row.get('wind_speed', 5.0)), axis=1)
        
        df_clean['temp_humidity_composite'] = df_clean.apply(
            lambda row: self.calculate_temp_humidity_composite(row['temperature'], row['humidity']), axis=1)
        
        return df_clean
    
    def analyze_correlations_by_era(self, df, location):
        """시대별 불쾌지수 성능 분석"""
        print(f"\n{location} - 시대별 불쾌지수 성능 분석")
        print("=" * 50)
        
        correlations = {}
        
        for idx_name, metadata in self.INDEX_METADATA.items():
            if idx_name in df.columns:
                corr = df[idx_name].corr(df['water_level'])
                correlations[idx_name] = {
                    'correlation': corr,
                    'development_year': metadata['development_year'],
                    'complexity_score': metadata['formula_complexity_score'],
                    'era': metadata['era'],
                    'name': metadata['name']
                }
        
        # 개발연도 순으로 정렬
        sorted_by_year = sorted(correlations.items(), key=lambda x: x[1]['development_year'])
        
        print(f"{'지수명':<25} {'개발연도':<8} {'시대':<20} {'복잡도':<6} {'상관계수':<8}")
        print("-" * 80)
        
        for idx_name, data in sorted_by_year:
            print(f"{data['name']:<25} {data['development_year']:<8} {data['era']:<20} "
                  f"{data['complexity_score']:<6} {data['correlation']:<8.4f}")
        
        return correlations
    
    def analyze_complexity_vs_performance(self, correlations, location):
        """복잡도 vs 성능 관계 분석"""
        print(f"\n{location} - 복잡도 vs 성능 분석")
        print("=" * 40)
        
        # 복잡도별 그룹화
        complexity_groups = {}
        for idx_name, data in correlations.items():
            complexity = data['complexity_score']
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append({
                'name': idx_name,
                'correlation': abs(data['correlation']),  # 절댓값으로 성능 평가
                'era': data['era']
            })
        
        print(f"{'복잡도':<8} {'평균 성능':<10} {'최고 성능':<10} {'해당 지수들'}")
        print("-" * 60)
        
        for complexity in sorted(complexity_groups.keys()):
            group = complexity_groups[complexity]
            avg_perf = np.mean([item['correlation'] for item in group])
            max_perf = np.max([item['correlation'] for item in group])
            indices = [item['name'] for item in group]
            
            print(f"{complexity:<8} {avg_perf:<10.4f} {max_perf:<10.4f} {', '.join(indices)}")
        
        return complexity_groups
    
    def analyze_era_trends(self, correlations, location):
        """시대별 성능 트렌드 분석"""
        print(f"\n{location} - 시대별 성능 트렌드")
        print("=" * 35)
        
        era_groups = {}
        for idx_name, data in correlations.items():
            era = data['era']
            if era not in era_groups:
                era_groups[era] = []
            era_groups[era].append({
                'name': idx_name,
                'correlation': abs(data['correlation']),
                'year': data['development_year']
            })
        
        # 시대순 정렬을 위한 키
        era_order = ['Early (1920s)', 'Classical (1950s)', 'Modern (1960s-1980s)', 
                     'Contemporary (1990s-2010s)', 'Latest (2020s)']
        
        print(f"{'시대':<25} {'평균 성능':<10} {'최고 성능':<10} {'지수 개수'}")
        print("-" * 60)
        
        for era in era_order:
            if era in era_groups:
                group = era_groups[era]
                avg_perf = np.mean([item['correlation'] for item in group])
                max_perf = np.max([item['correlation'] for item in group])
                count = len(group)
                
                print(f"{era:<25} {avg_perf:<10.4f} {max_perf:<10.4f} {count}")
        
        return era_groups
    
    def save_results(self, location, correlations, complexity_groups, era_groups):
        """분석 결과 저장"""
        results = {
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'correlations': correlations,
            'complexity_analysis': complexity_groups,
            'era_analysis': era_groups,
            'metadata': self.INDEX_METADATA
        }
        
        output_file = f"../results/reports/comprehensive_discomfort_analysis_{location}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과 저장 완료: {output_file}")
        
        return results
    
    def run_comprehensive_analysis(self):
        """종합 분석 실행"""
        print("🎯 무심천 종합 불쾌지수 분석 시작")
        print("=" * 60)
        
        locations = ['cheongju', 'gadeok']
        
        for location in locations:
            # 데이터 로딩 및 불쾌지수 계산
            df = self.load_and_calculate_indices(location)
            
            # 상관관계 분석
            correlations = self.analyze_correlations_by_era(df, location)
            
            # 복잡도 vs 성능 분석
            complexity_groups = self.analyze_complexity_vs_performance(correlations, location)
            
            # 시대별 트렌드 분석
            era_groups = self.analyze_era_trends(correlations, location)
            
            # 결과 저장
            self.results[location] = self.save_results(location, correlations, complexity_groups, era_groups)
        
        return self.results

def main():
    """메인 실행 함수"""
    analyzer = ComprehensiveDiscomfortAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print("\n🎉 종합 분석 완료!")
    print("\n주요 발견점:")
    
    # 지역별 최고 성능 지수 비교
    for location, result in results.items():
        best_corr = max(result['correlations'].items(), key=lambda x: abs(x[1]['correlation']))
        print(f"\n{location.upper()} 최고 성능:")
        print(f"  - {best_corr[1]['name']}: {best_corr[1]['correlation']:.4f}")
        print(f"  - 개발연도: {best_corr[1]['development_year']}")
        print(f"  - 복잡도: {best_corr[1]['complexity_score']}")

if __name__ == "__main__":
    main() 