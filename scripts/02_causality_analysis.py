#!/usr/bin/env python3
"""
무심천 불쾌지수-수위 분석: 그랜저 인과성 검정
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """처리된 데이터 로드"""
    print("처리된 데이터 로딩 중...")
    
    cheongju_df = pd.read_csv('../data/processed/cheongju_processed.csv')
    gadeok_df = pd.read_csv('../data/processed/gadeok_processed.csv')
    
    # datetime 컬럼 변환
    cheongju_df['datetime'] = pd.to_datetime(cheongju_df['datetime'])
    gadeok_df['datetime'] = pd.to_datetime(gadeok_df['datetime'])
    
    print(f"청주금천 데이터: {cheongju_df.shape}")
    print(f"가덕 데이터: {gadeok_df.shape}")
    
    return cheongju_df, gadeok_df

def check_stationarity(series, series_name, max_diff=2):
    """
    시계열 정상성 검정 (ADF 테스트)
    Args:
        series: 시계열 데이터
        series_name: 시계열 이름
        max_diff: 최대 차분 횟수
    Returns:
        정상성을 만족하는 시계열과 차분 횟수
    """
    print(f"\n{series_name} 정상성 검정:")
    
    # 결측값 제거
    clean_series = series.dropna()
    
    if len(clean_series) < 50:
        print(f"  데이터가 부족합니다 (n={len(clean_series)})")
        return None, None
    
    # 원본 시계열 검정
    adf_result = adfuller(clean_series, autolag='AIC')
    print(f"  원본 시계열 ADF p-value: {adf_result[1]:.6f}")
    
    if adf_result[1] <= 0.05:
        print(f"  → {series_name}은 정상시계열입니다")
        return clean_series, 0
    
    # 차분으로 정상성 달성 시도
    diff_series = clean_series.copy()
    for d in range(1, max_diff + 1):
        diff_series = diff_series.diff().dropna()
        
        if len(diff_series) < 50:
            print(f"  차분 후 데이터가 부족합니다")
            return None, None
            
        adf_result = adfuller(diff_series, autolag='AIC')
        print(f"  {d}차 차분 ADF p-value: {adf_result[1]:.6f}")
        
        if adf_result[1] <= 0.05:
            print(f"  → {d}차 차분 후 정상시계열 달성")
            return diff_series, d
    
    print(f"  → {max_diff}차 차분까지 시도했으나 정상성 달성 실패")
    return None, None

def granger_causality_test(cause_series, effect_series, max_lags=24):
    """
    그랜저 인과성 검정 수행
    Args:
        cause_series: 원인 시계열
        effect_series: 결과 시계열
        max_lags: 최대 지연 기간
    Returns:
        최적 지연 기간과 p-value
    """
    # 두 시계열을 결합하여 결측값 동시 제거
    combined_df = pd.DataFrame({
        'cause': cause_series,
        'effect': effect_series
    }).dropna()
    
    if len(combined_df) < max_lags * 3:
        print(f"    데이터가 부족하여 검정 불가 (n={len(combined_df)})")
        return None, None
    
    # 그랜저 인과성 검정
    test_data = combined_df[['effect', 'cause']].values
    
    try:
        gc_results = grangercausalitytests(test_data, maxlag=max_lags, verbose=False)
        
        # 최적 지연 기간 찾기 (가장 낮은 p-value)
        best_lag = None
        best_pvalue = float('inf')
        
        for lag in range(1, max_lags + 1):
            if lag in gc_results:
                # F-test p-value 사용
                pvalue = gc_results[lag][0]['ssr_ftest'][1]
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_lag = lag
        
        return best_lag, best_pvalue
    
    except Exception as e:
        print(f"    그랜저 검정 실행 중 오류: {e}")
        return None, None

def analyze_discomfort_causality(df, region_name):
    """
    불쾌지수의 수위에 대한 인과성 분석
    Args:
        df: 데이터 DataFrame
        region_name: 지역명
    Returns:
        결과 딕셔너리
    """
    print(f"\n=== {region_name} 불쾌지수-수위 인과성 분석 ===")
    
    # 분석할 불쾌지수 지표들
    discomfort_indices = ['THI', 'heat_index', 'traditional_DI', 'apparent_temp', 'temp_humidity_composite']
    
    results = {}
    
    # 수위 시계열 정상성 검정
    water_level_stationary, water_diff_order = check_stationarity(df['water_level'], '수위')
    if water_level_stationary is None:
        print(f"{region_name} 수위 데이터에 문제가 있어 분석 중단")
        return results
    
    # 각 불쾌지수 지표에 대해 인과성 검정
    for idx_name in discomfort_indices:
        if idx_name not in df.columns:
            continue
            
        print(f"\n--- {idx_name} → 수위 인과성 검정 ---")
        
        # 불쾌지수 정상성 검정
        idx_stationary, idx_diff_order = check_stationarity(df[idx_name], idx_name)
        
        if idx_stationary is None:
            print(f"  {idx_name} 정상성 달성 실패")
            continue
        
        # 그랜저 인과성 검정
        print(f"  그랜저 인과성 검정 수행 중...")
        
        # 동일한 차분 차수 적용
        if water_diff_order == idx_diff_order:
            # 같은 차분 차수면 바로 사용
            if water_diff_order == 0:
                cause_data = df[idx_name]
                effect_data = df['water_level']
            else:
                cause_data = df[idx_name].diff(water_diff_order)
                effect_data = df['water_level'].diff(water_diff_order)
        else:
            # 차분 차수가 다르면 높은 차수로 맞춤
            max_diff = max(water_diff_order, idx_diff_order)
            cause_data = df[idx_name].diff(max_diff) if max_diff > 0 else df[idx_name]
            effect_data = df['water_level'].diff(max_diff) if max_diff > 0 else df['water_level']
        
        best_lag, best_pvalue = granger_causality_test(cause_data, effect_data)
        
        if best_lag is not None:
            significance = "***" if best_pvalue < 0.001 else "**" if best_pvalue < 0.01 else "*" if best_pvalue < 0.05 else ""
            print(f"  최적 지연: {best_lag}시간, p-value: {best_pvalue:.6f} {significance}")
            
            results[idx_name] = {
                'best_lag': best_lag,
                'p_value': best_pvalue,
                'significant': best_pvalue < 0.05,
                'diff_order': max(water_diff_order, idx_diff_order) if water_diff_order != idx_diff_order else water_diff_order
            }
        else:
            print(f"  인과성 검정 실패")
            results[idx_name] = None
    
    return results

def analyze_cascade_effects(df, region_name):
    """
    불쾌지수 → 강수량 → 수위 cascade 효과 분석
    Args:
        df: 데이터 DataFrame  
        region_name: 지역명
    Returns:
        cascade 분석 결과
    """
    print(f"\n=== {region_name} Cascade 효과 분석 (불쾌지수 → 강수량 → 수위) ===")
    
    cascade_results = {}
    discomfort_indices = ['THI', 'traditional_DI', 'apparent_temp', 'temp_humidity_composite']
    
    for idx_name in discomfort_indices:
        if idx_name not in df.columns:
            continue
            
        print(f"\n--- {idx_name} cascade 분석 ---")
        
        # 1단계: 불쾌지수 → 강수량
        print(f"  1단계: {idx_name} → 강수량")
        
        # 정상성 검정
        idx_stationary, idx_diff = check_stationarity(df[idx_name], idx_name)
        precip_stationary, precip_diff = check_stationarity(df['precipitation'], '강수량')
        
        if idx_stationary is None or precip_stationary is None:
            print(f"    정상성 달성 실패")
            continue
        
        # 그랜저 인과성 검정
        max_diff = max(idx_diff, precip_diff)
        cause_data = df[idx_name].diff(max_diff) if max_diff > 0 else df[idx_name]
        effect_data = df['precipitation'].diff(max_diff) if max_diff > 0 else df['precipitation']
        
        lag1, pval1 = granger_causality_test(cause_data, effect_data, max_lags=12)
        
        if lag1 is not None:
            sig1 = "***" if pval1 < 0.001 else "**" if pval1 < 0.01 else "*" if pval1 < 0.05 else ""
            print(f"    {idx_name} → 강수량: lag={lag1}, p={pval1:.6f} {sig1}")
        
        # 2단계: 강수량 → 수위 (이미 알려진 관계)
        print(f"  2단계: 강수량 → 수위")
        
        water_stationary, water_diff = check_stationarity(df['water_level'], '수위')
        if water_stationary is None:
            continue
            
        max_diff2 = max(precip_diff, water_diff)
        cause_data2 = df['precipitation'].diff(max_diff2) if max_diff2 > 0 else df['precipitation']
        effect_data2 = df['water_level'].diff(max_diff2) if max_diff2 > 0 else df['water_level']
        
        lag2, pval2 = granger_causality_test(cause_data2, effect_data2, max_lags=6)
        
        if lag2 is not None:
            sig2 = "***" if pval2 < 0.001 else "**" if pval2 < 0.01 else "*" if pval2 < 0.05 else ""
            print(f"    강수량 → 수위: lag={lag2}, p={pval2:.6f} {sig2}")
        
        # cascade 효과 종합
        cascade_results[idx_name] = {
            'step1_lag': lag1,
            'step1_pvalue': pval1,
            'step1_significant': pval1 < 0.05 if pval1 is not None else False,
            'step2_lag': lag2,
            'step2_pvalue': pval2,
            'step2_significant': pval2 < 0.05 if pval2 is not None else False,
            'full_cascade': (pval1 < 0.05 and pval2 < 0.05) if (pval1 is not None and pval2 is not None) else False
        }
    
    return cascade_results

def save_causality_results(cheongju_results, gadeok_results, cheongju_cascade, gadeok_cascade):
    """인과성 분석 결과 저장"""
    print("\n인과성 분석 결과 저장 중...")
    
    # 직접 인과성 결과를 DataFrame으로 변환
    direct_results = []
    
    for region, results in [('청주금천', cheongju_results), ('가덕', gadeok_results)]:
        for idx_name, result in results.items():
            if result is not None:
                direct_results.append({
                    'region': region,
                    'discomfort_index': idx_name,
                    'best_lag': result['best_lag'],
                    'p_value': result['p_value'],
                    'significant': result['significant'],
                    'diff_order': result['diff_order']
                })
    
    direct_df = pd.DataFrame(direct_results)
    direct_df.to_csv('../results/reports/granger_causality_direct.csv', index=False)
    
    # Cascade 결과를 DataFrame으로 변환
    cascade_results = []
    
    for region, results in [('청주금천', cheongju_cascade), ('가덕', gadeok_cascade)]:
        for idx_name, result in results.items():
            cascade_results.append({
                'region': region,
                'discomfort_index': idx_name,
                'step1_lag': result['step1_lag'],
                'step1_pvalue': result['step1_pvalue'],
                'step1_significant': result['step1_significant'],
                'step2_lag': result['step2_lag'],
                'step2_pvalue': result['step2_pvalue'],
                'step2_significant': result['step2_significant'],
                'full_cascade': result['full_cascade']
            })
    
    cascade_df = pd.DataFrame(cascade_results)
    cascade_df.to_csv('../results/reports/granger_causality_cascade.csv', index=False)
    
    print("결과 저장 완료!")

def main():
    """메인 실행 함수"""
    print("=== 무심천 불쾌지수-수위 그랜저 인과성 분석 시작 ===\n")
    
    # 데이터 로드
    cheongju_df, gadeok_df = load_processed_data()
    
    # 청주금천 인과성 분석
    cheongju_results = analyze_discomfort_causality(cheongju_df, "청주금천")
    cheongju_cascade = analyze_cascade_effects(cheongju_df, "청주금천")
    
    # 가덕 인과성 분석
    gadeok_results = analyze_discomfort_causality(gadeok_df, "가덕")
    gadeok_cascade = analyze_cascade_effects(gadeok_df, "가덕")
    
    # 결과 저장
    save_causality_results(cheongju_results, gadeok_results, cheongju_cascade, gadeok_cascade)
    
    # 결과 요약 출력
    print("\n" + "="*80)
    print("=== 인과성 분석 결과 요약 ===")
    print("="*80)
    
    for region, results in [('청주금천', cheongju_results), ('가덕', gadeok_results)]:
        print(f"\n{region} 지역:")
        print("-" * 40)
        
        significant_results = []
        for idx_name, result in results.items():
            if result is not None and result['significant']:
                significant_results.append((idx_name, result['best_lag'], result['p_value']))
        
        if significant_results:
            significant_results.sort(key=lambda x: x[2])  # p-value로 정렬
            for idx_name, lag, pval in significant_results:
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                print(f"  {idx_name} → 수위: {lag}시간 지연, p={pval:.6f} {stars}")
        else:
            print("  유의한 인과관계를 발견하지 못했습니다.")
    
    print("\n=== 그랜저 인과성 분석 완료 ===")

if __name__ == "__main__":
    main() 