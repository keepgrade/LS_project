import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#pip install nycflights13
from nycflights13 import flights, planes

# 주제 자유 : 
# merge 사용해서 flights 와 planes 병합한 데이터로
# 각 데이터 변수 최소 하나씩 선택한 후 분석할 것. 
# 날짜 & 시간 저ㄴ처리 코드 들어갈 것.
# 문자열 전처리 코드 들어갈것. 
# 시각화 할 것. 
# 시각화 종류 최소 3개 (배우지 않은것도 가능)

flights.info()
planes.info()

# tailnum을 key로 merge 하기 
df = pd.merge(flights, planes, on = 'tailnum', how='left')
df.info()


# 보고서 주제 : 제조사별 기체 결함으로 인한 운항 지연 분석 보고서

# 1. 개요 
# 목적 : 우리는 항공사로서 비행기 운항의 신뢰성과 정시성으 중요하게 고려해야합니다.
# 최근 운항 지연이 빈번하게 발생하고 있습니다.
# 다양한 요인이 있을 수 있지만, 기체 결함 및 부품 문제가 지연의 주요 원인 중 하나라는 내부 보고를 받았습니다.
# 이에 따라 제조사별 기체의 운항 지연 패턴을 분석하고, 특정 제조사의 기체에서 지연이 빈번하게 발생하는지 파악하여 운영 효율성을 개선하고자 합니다.


# 분석 목표 :
# 제조사별 운항 지연 패턴 분석: 어떤 제조사의 기체가 지연을 자주 유발하는지 확인.


# 2. 데이터 분비 및 전처리
# 2.1 데이터 병합 
# 두 데이터를 tailnum(기체 등록번호) 기준으로 병합하여 기체별 제조사 정보를 결합
df = pd.merge(flights, planes, on='tailnum', how='left')

# 2.2 날짜 및 시간 데이터 생성 
# 연(year_x), 월(month), 일(day), 시(hour), 분(minute) 데이터를 조합하여 date_time 생성
# 2013년 1월~6월(상반기) 데이터를 필터링
df['date_time'] = pd.to_datetime(df['year_x'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str) + ' ' + df['hour'].astype(str) + ':' + df['minute'].astype(str))
df_fh = df[(df['date_time'] >= '2013-01-01') & (df['date_time'] <= '2013-06-30')]


# 2.3 지연시간 관련 전처리
# 음수 도착 지연 시간 제거 (빠른 이륙은 지연 원인이 아니므로 분석 제외)
# 제조사별 운항 횟수 확인
# 우리는 운항이 가장 많은 항공사야 우리 항공사를 골라줘 

# 항공사별 운항 횟수 집계
top_airline = df_fh['carrier'].value_counts().idxmax()  # 운항 횟수가 가장 많은 항공사 선택

df_fh_our_airline = df_fh[df_fh['carrier'] == top_airline]  # 우리 항공사의 데이터만 필터링
df_fh_our_airline = df_fh_our_airline[df_fh_our_airline['dep_delay'] >= 0]  # 음수 제거

# 우리 항공사의 상위 5개 제조사 선정
top_5_manufacturers = df_fh_our_airline['manufacturer'].value_counts().head(5).index

# 어차피 3개밖에 안쓰는구나 
set(df_fh_our_airline['manufacturer'])

# 제조사별 운항 횟수 집계
manufacturer_counts = df_fh_our_airline['manufacturer'].value_counts()

# 원형 그래프 (파이 차트) 생성
plt.figure(figsize=(8, 8))
plt.pie(manufacturer_counts, labels=manufacturer_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Aircraft Manufacturer Distribution for Our Airline')
plt.show()

# 3. 제조사별 운항 지연 분석
# 3.1 제조사별 평균 지연 시간

# 이상치 제거를 위한 IQR(Interquartile Range) 계산
Q1 = df_fh_our_airline['dep_delay'].quantile(0.25)
Q3 = df_fh_our_airline['dep_delay'].quantile(0.75)
IQR = Q3 - Q1

# 정상 범위 설정 (IQR 범위 내 데이터만 유지)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 제거
df_fh_our_airline = df_fh_our_airline[(df_fh_our_airline['dep_delay'] >= lower_bound) & (df_fh_our_airline['dep_delay'] <= upper_bound)]


# - 제조사별 평균 지연 시간 및 표준편차 확인
df_fh_our_airline.groupby('manufacturer')['dep_delay'].describe()

# 3.2 제조사별 운항 횟수 대비 지연 비율
# - 단순 평균 지연 시간 비교가 아닌, 운항 횟수 대비 지연 발생 비율 계산
delay_rate = df_fh_our_airline.groupby('manufacturer')['dep_delay'].count() /df_fh_our_airline['manufacturer'].value_counts()

delay_rate.sort_values(ascending=False)


# 3.3 제조사별 이륙 vs. 도착 지연 비교
# - 특정 제조사의 기체에서 이륙 지연이 많은지, 운항 중 추가 지연이 발생하는지 분석
df_fh_our_airline.groupby('manufacturer')[['dep_delay']].mean()


# 3.4 월별 지연 패턴 분석
# - 제조사별 월별 평균 도착 지연 분석
df_fh_our_airline['month'] = df_fh_our_airline['date_time'].dt.month
monthly_delay = df_fh_our_airline.groupby(['manufacturer', 'month'])['arr_delay'].mean().unstack()
monthly_delay


plt.figure(figsize=(12,10))
monthly_delay = df_fh_our_airline.groupby(['manufacturer', 'month'])['dep_delay'].mean().unstack()

sns.lineplot(data=monthly_delay.T, marker="o")
plt.xlabel("Month")
plt.ylabel("Average Delay (minutes)")
plt.title("Monthly Average Delay by Manufacturer")
plt.legend(title="Manufacturer")
plt.grid(True)
plt.show()

# 4. 데이터 시각화
# 4.1 제조사별 도착 지연 시간 분포 

# 제조사별 평균 지연 시간 비교
mean_delays = df_fh_our_airline.groupby('manufacturer')['dep_delay'].mean().sort_values()

plt.figure(figsize=(10,5))
mean_delays.plot(kind='bar', color='skyblue')
plt.xlabel('Manufacturer')
plt.ylabel('Mean Departure Delay (minutes)')
plt.title('Mean Departure Delay by Manufacturer (Jan-Jun 2013)')
plt.grid(True)
plt.show()

# 4.2 제조사별 이륙 vs. 도착 지연 비교 (가로 막대 그래프)
delay_comparison = df_fh_our_airline.groupby('manufacturer')[['dep_delay', 'arr_delay']].mean()

delay_comparison.plot(kind='barh', figsize=(12,6), color=['royalblue', 'tomato'])
plt.xlabel('Mean Delay (minutes)')
plt.ylabel('Manufacturer')
plt.title('Mean Departure vs. Arrival Delay by Manufacturer (Jan-Jun 2013)')
plt.legend(['Departure Delay', 'Arrival Delay'])
plt.grid(True)
plt.show()



# 4.3 3차원 데이터 분석을 통해서 보기 
# 제조사별 평균 이륙 지연, 도착 지연 및 운행 대수 데이터 준비
delay_comparison = df_fh_our_airline.groupby('manufacturer')[['dep_delay', 'arr_delay']].mean()
manufacturer_flight_counts = df_fh_our_airline['manufacturer'].value_counts()

# 데이터 병합
delay_comparison['flight_count'] = manufacturer_flight_counts

# 3D 그래프 생성
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# x, y, z 좌표 설정
x_labels = delay_comparison.index  # 제조사 목록
x = np.arange(len(x_labels))  # x축 인덱스 (제조사별)
y_dep = delay_comparison['dep_delay']  # 이륙 지연
y_arr = delay_comparison['arr_delay']  # 도착 지연
z = delay_comparison['flight_count']  # 운행 대수

# 막대 너비 설정
bar_width = 0.3

# 3D 막대 그래프 형태로 시각화 (조금 더 정교한 스타일 적용)
ax.bar(x - bar_width, y_dep, zs=z, zdir='y', color='royalblue', alpha=0.8, width=bar_width, label='Departure Delay')
ax.bar(x + bar_width, y_arr, zs=z, zdir='y', color='tomato', alpha=0.8, width=bar_width, label='Arrival Delay')

# 축 설정 및 시각적 개선
ax.set_xlabel('Manufacturer', fontsize=12, labelpad=15, color = 'red')
ax.set_ylabel('Total Flights Operated', fontsize=12, labelpad=15)
ax.set_zlabel('Mean Delay (minutes)', fontsize=12, labelpad=15)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, fontsize=10, ha='right')
ax.set_title('3D Visualization of Mean Departure & Arrival Delay by Manufacturer', fontsize=14, pad=20)
ax.view_init(elev=20, azim=120)  # 시점 조정

# 범례 추가
ax.legend()

# 그래프 출력
plt.show()


# 5. 결론 및 운영 개선 방안

