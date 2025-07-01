import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DiscomfortIndexCalculator:
    """
    확장된 불쾌지수 계산기 클래스
    - 기존 4개 지수 + 새로운 6개 지수 추가
    - 각 지수의 개발연도, 복잡도, 특성 정보 포함
    """
    
    # 각 불쾌지수의 메타데이터
    INDEX_METADATA = {
        'THI': {
            'name': 'Temperature-Humidity Index',
            'development_year': 1950,
            'complexity': 'Simple',
            'formula_complexity_score': 1,
            'primary_use': 'General comfort assessment',
            'variables_used': ['temperature', 'humidity']
        },
        'traditional_DI': {
            'name': 'Traditional Discomfort Index',
            'development_year': 1950,
            'complexity': 'Simple', 
            'formula_complexity_score': 1,
            'primary_use': 'Basic thermal comfort',
            'variables_used': ['temperature', 'humidity']
        },
        'heat_index': {
            'name': 'Heat Index (Apparent Temperature)',
            'development_year': 1979,
            'complexity': 'Complex',
            'formula_complexity_score': 4,
            'primary_use': 'US weather forecasting',
            'variables_used': ['temperature', 'humidity']
        },
        'humidex': {
            'name': 'Humidity Index (Canadian)',
            'development_year': 1965,
            'complexity': 'Moderate',
            'formula_complexity_score': 2,
            'primary_use': 'Canadian weather forecasting',
            'variables_used': ['temperature', 'humidity']
        },
        'apparent_temp': {
            'name': 'Apparent Temperature (Australian)',
            'development_year': 1984,
            'complexity': 'Moderate',
            'formula_complexity_score': 3,
            'primary_use': 'Australian weather services',
            'variables_used': ['temperature', 'humidity', 'wind_speed']
        },
        'WBGT_simple': {
            'name': 'Wet Bulb Globe Temperature (Simplified)',
            'development_year': 1956,
            'complexity': 'Moderate',
            'formula_complexity_score': 2,
            'primary_use': 'Military and sports safety',
            'variables_used': ['temperature', 'humidity']
        },
        'temp_humidity_composite': {
            'name': 'Temperature-Humidity Composite',
            'development_year': 2024,  # 우리의 커스텀 지표
            'complexity': 'Simple',
            'formula_complexity_score': 1,
            'primary_use': 'Water level prediction research',
            'variables_used': ['temperature', 'humidity']
        },
        'UTCI_simplified': {
            'name': 'Universal Thermal Climate Index (Simplified)',
            'development_year': 2012,
            'complexity': 'Very Complex',
            'formula_complexity_score': 5,
            'primary_use': 'International climate assessment',
            'variables_used': ['temperature', 'humidity', 'wind_speed']
        },
        'effective_temperature': {
            'name': 'Effective Temperature',
            'development_year': 1923,
            'complexity': 'Simple',
            'formula_complexity_score': 1,
            'primary_use': 'Early comfort assessment',
            'variables_used': ['temperature', 'humidity']
        },
        'feels_like_temp': {
            'name': 'Feels Like Temperature',
            'development_year': 1990,
            'complexity': 'Moderate',
            'formula_complexity_score': 3,
            'primary_use': 'Modern weather apps',
            'variables_used': ['temperature', 'humidity', 'wind_speed']
        }
    }
    
    @staticmethod
    def calculate_THI(temp, humidity):
        """Temperature-Humidity Index (1950년)"""
        try:
            return (1.8 * temp + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * temp - 26))
        except:
            return np.nan
    
    @staticmethod
    def calculate_heat_index(temp, humidity):
        """Heat Index - 미국 기상청 공식 (1979년)"""
        try:
            # 섭씨를 화씨로 변환
            T = temp * 9/5 + 32
            RH = humidity
            
            # 간단한 공식 먼저 시도
            HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
            
            # 80°F 이상일 때 정확한 공식 사용
            if HI >= 80:
                HI = (-42.379 + 2.04901523*T + 10.14333127*RH 
                      - 0.22475541*T*RH - 6.83783e-3*T**2 
                      - 5.481717e-2*RH**2 + 1.22874e-3*T**2*RH 
                      + 8.5282e-4*T*RH**2 - 1.99e-6*T**2*RH**2)
            
            # 화씨를 섭씨로 변환
            return (HI - 32) * 5/9
        except:
            return np.nan
    
    @staticmethod
    def calculate_traditional_DI(temp, humidity):
        """전통적 불쾌지수 (1950년)"""
        try:
            return temp + 0.36 * humidity
        except:
            return np.nan
    
    @staticmethod
    def calculate_apparent_temp(temp, humidity, wind_speed=None):
        """체감온도 - 호주 기상청 공식 (1984년)"""
        try:
            if wind_speed is None:
                wind_speed = 5.0  # 기본값
            
            # 수증기압 계산
            e = humidity / 100 * 6.105 * np.exp(17.27 * temp / (237.7 + temp))
            
            AT = temp + 0.33 * e - 0.7 * wind_speed - 4.0
            return AT
        except:
            return np.nan
    
    @staticmethod
    def calculate_temp_humidity_composite(temp, humidity):
        """온습도 복합 지표 - 기존 연구에서 효과적이었던 지표"""
        try:
            return 0.7 * temp + 0.3 * humidity
        except:
            return np.nan
    
    @staticmethod
    def calculate_humidex(temp, humidity):
        """Humidex - 캐나다 기상청 공식 (1965년)"""
        try:
            # 이슬점 온도 계산 (Magnus 공식)
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
            dewpoint = (b * alpha) / (a - alpha)
            
            # Humidex 계산
            e = 6.11 * np.exp(5417.753 * ((1/273.16) - (1/(dewpoint + 273.16))))
            h = 0.5555 * (e - 10.0)
            
            return temp + h
        except:
            return np.nan
    
    @staticmethod
    def calculate_WBGT_simple(temp, humidity):
        """WBGT 간소화 공식 (1956년) - 습구온도 근사"""
        try:
            # 습구온도 근사 계산 (Stull 2011)
            wet_bulb = temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + \
                      np.arctan(temp + humidity) - np.arctan(humidity - 1.676331) + \
                      0.00391838 * np.power(humidity, 1.5) * np.arctan(0.023101 * humidity) - 4.686035
            
            # WBGT 실내 공식 (0.7 * 습구온도 + 0.3 * 건구온도)
            return 0.7 * wet_bulb + 0.3 * temp
        except:
            return np.nan
    
    @staticmethod
    def calculate_UTCI_simplified(temp, humidity, wind_speed=None):
        """UTCI 간소화 근사 공식 (2012년)"""
        try:
            if wind_speed is None:
                wind_speed = 5.0  # 기본값
            
            # 간소화된 UTCI 근사 (실제 UTCI는 6차 다항식)
            # 온도, 습도, 풍속의 조합을 통한 근사
            RH = humidity / 100.0
            v = max(wind_speed, 0.5)  # 최소 풍속
            
            # 기본 온도에서 시작
            utci = temp
            
            # 습도 효과
            if temp > 9:
                utci += (1.8 * temp - 32.0) * RH * 0.1
            
            # 풍속 효과 (냉각)
            utci -= (37 - temp) / (37 - temp + 68) * (1.76 + 1.4 * v**0.75) * 0.1
            
            return utci
        except:
            return np.nan
    
    @staticmethod
    def calculate_effective_temperature(temp, humidity):
        """유효온도 - 초기 열환경 지표 (1923년)"""
        try:
            # Bedford와 Warner의 간소화 공식
            return temp - (100 - humidity) / 4
        except:
            return np.nan
    
    @staticmethod
    def calculate_feels_like_temp(temp, humidity, wind_speed=None):
        """체감온도 - 현대적 버전 (1990년대)"""
        try:
            if wind_speed is None:
                wind_speed = 5.0
            
            # 온도에 따라 다른 공식 적용
            if temp >= 26.7:  # 따뜻할 때는 Heat Index 스타일
                return DiscomfortIndexCalculator.calculate_heat_index(temp, humidity)
            elif temp <= 10.0:  # 추울 때는 Wind Chill 고려
                if wind_speed > 4.8:
                    return 13.12 + 0.6215*temp - 11.37*(wind_speed**0.16) + 0.3965*temp*(wind_speed**0.16)
                else:
                    return temp
            else:  # 중간 온도
                return temp + 0.3 * (humidity - 60) / 40 - 0.7 * wind_speed / 10
        except:
            return np.nan

def calculate_discomfort_indices(df):
    """모든 불쾌지수 계산"""
    calc = DiscomfortIndexCalculator()
    
    print("불쾌지수 계산 중...")
    
    # 기존 지수들
    df['THI'] = df.apply(lambda row: calc.calculate_THI(row['temperature'], row['humidity']), axis=1)
    df['heat_index'] = df.apply(lambda row: calc.calculate_heat_index(row['temperature'], row['humidity']), axis=1)
    df['traditional_DI'] = df.apply(lambda row: calc.calculate_traditional_DI(row['temperature'], row['humidity']), axis=1)
    df['temp_humidity_composite'] = df.apply(lambda row: calc.calculate_temp_humidity_composite(row['temperature'], row['humidity']), axis=1)
    
    # 새로운 지수들
    df['humidex'] = df.apply(lambda row: calc.calculate_humidex(row['temperature'], row['humidity']), axis=1)
    df['WBGT_simple'] = df.apply(lambda row: calc.calculate_WBGT_simple(row['temperature'], row['humidity']), axis=1)
    df['effective_temperature'] = df.apply(lambda row: calc.calculate_effective_temperature(row['temperature'], row['humidity']), axis=1)
    
    # 풍속이 있는 경우
    if 'wind_speed' in df.columns and df['wind_speed'].notna().any():
        df['apparent_temp'] = df.apply(lambda row: calc.calculate_apparent_temp(
            row['temperature'], row['humidity'], row.get('wind_speed', 5.0)), axis=1)
        df['UTCI_simplified'] = df.apply(lambda row: calc.calculate_UTCI_simplified(
            row['temperature'], row['humidity'], row.get('wind_speed', 5.0)), axis=1)
        df['feels_like_temp'] = df.apply(lambda row: calc.calculate_feels_like_temp(
            row['temperature'], row['humidity'], row.get('wind_speed', 5.0)), axis=1)
    else:
        df['apparent_temp'] = df.apply(lambda row: calc.calculate_apparent_temp(row['temperature'], row['humidity']), axis=1)
        df['UTCI_simplified'] = df.apply(lambda row: calc.calculate_UTCI_simplified(row['temperature'], row['humidity']), axis=1)
        df['feels_like_temp'] = df.apply(lambda row: calc.calculate_feels_like_temp(row['temperature'], row['humidity']), axis=1)
    
    # 불쾌지수 목록 업데이트
    discomfort_indices = ['THI', 'heat_index', 'traditional_DI', 'apparent_temp', 'temp_humidity_composite',
                         'humidex', 'WBGT_simple', 'UTCI_simplified', 'effective_temperature', 'feels_like_temp']
    
    # 유효한 데이터 개수 출력
    for idx in discomfort_indices:
        valid_count = df[idx].notna().sum()
        total_count = len(df)
        print(f"{idx}: {valid_count:,}개 유효 데이터 ({valid_count/total_count*100:.1f}%)")
    
    return df, discomfort_indices 