import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px

st.set_page_config(page_title="Аналіз даних NOAA", layout="wide")

@st.cache_data
def load_data():
    province_mapping = {
        1: (22, 'Черкаська'), 2: (24, 'Чернігівська'), 3: (23, 'Чернівецька'), 
        4: (25, 'АР Крим'), 5: (3, 'Дніпропетровська'), 6: (4, 'Донецька'), 
        7: (8, 'Івано-Франківська'), 8: (19, 'Харківська'), 9: (20, 'Херсонська'), 
        10: (21, 'Хмельницька'), 11: (9, 'Київська'), 12: (26, 'м. Київ'), 
        13: (10, 'Кіровоградська'), 14: (11, 'Луганська'), 15: (12, 'Львівська'), 
        16: (13, 'Миколаївська'), 17: (14, 'Одеська'), 18: (15, 'Полтавська'), 
        19: (16, 'Рівненська'), 20: (27, 'м. Севастополь'), 21: (17, 'Сумська'), 
        22: (18, 'Тернопільська'), 23: (6, 'Закарпатська'), 24: (1, 'Вінницька'), 
        25: (2, 'Волинська'), 26: (7, 'Запорізька'), 27: (5, 'Житомирська')
    }
    
    all_files = glob.glob("vhi_data/*.csv")
    if not all_files:
        return pd.DataFrame()
        
    dfs = []
    for file in all_files:
        base_name = os.path.basename(file)
        old_id = int(base_name.split('_')[2])
        
        df = pd.read_csv(file, index_col=False, header=None, skiprows=2, skipfooter=1, engine='python',
                         names=['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'Empty'])
        df = df.drop('Empty', axis=1, errors='ignore')
        df = df[df['VHI'] != -1.0] 
        
        new_id, prov_name = province_mapping.get(old_id, (old_id, "Невідомо"))
        df['Province'] = prov_name
        df['Province_ID'] = new_id
        dfs.append(df)
        
    df_clean = pd.concat(dfs, ignore_index=True)
    df_clean['Year'] = df_clean['Year'].astype(str).str.replace('<tt><pre>', '').astype(int)
    
    df_clean = df_clean.sort_values(by=['Year', 'Week'])
    df_clean['Period'] = df_clean['Year'].astype(str) + "-W" + df_clean['Week'].astype(str).str.zfill(2)
    
    return df_clean

df = load_data()

if df.empty:
    st.error("Дані не знайдено! Переконайтеся, що файли CSV знаходяться у папці 'vhi_data'.")
    st.stop()


min_week, max_week = int(df['Week'].min()), int(df['Week'].max())
min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
provinces = sorted(df['Province'].unique())
indices = ['VCI', 'TCI', 'VHI']

def reset_filters():
    st.session_state['index_sel'] = indices[0]
    st.session_state['prov_sel'] = provinces[0]
    st.session_state['week_sel'] = (min_week, max_week)
    st.session_state['year_sel'] = (min_year, max_year)
    st.session_state['sort_asc'] = False
    st.session_state['sort_desc'] = False


col1, col2 = st.columns([1, 3])

with col1:
    st.header("Налаштування")
    
    st.button('Скинути фільтри', on_click=reset_filters)
    
    selected_index = st.selectbox('Оберіть індекс', indices, key='index_sel')
    
    selected_province = st.selectbox('Оберіть область', provinces, key='prov_sel')
    
    if 'week_sel' not in st.session_state:
        st.session_state['week_sel'] = (min_week, max_week)
    week_range = st.slider('Інтервал тижнів', min_week, max_week, key='week_sel')
    
    if 'year_sel' not in st.session_state:
        st.session_state['year_sel'] = (min_year, max_year)
    year_range = st.slider('Інтервал років', min_year, max_year, key='year_sel')
    
    st.write("Сортування даних у таблиці:")
    sort_asc = st.checkbox("За зростанням обраного індексу", key='sort_asc')
    sort_desc = st.checkbox("За спаданням обраного індексу", key='sort_desc')


mask = (
    (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
    (df['Week'] >= week_range[0]) & (df['Week'] <= week_range[1])
)
df_filtered_all_provinces = df[mask]
df_filtered = df_filtered_all_provinces[df_filtered_all_provinces['Province'] == selected_province]

if sort_asc and sort_desc:
    st.warning("Увага: Увімкнено обидва чекбокси сортування. Сортування скасовано.")
elif sort_asc:
    df_filtered = df_filtered.sort_values(by=selected_index, ascending=True)
elif sort_desc:
    df_filtered = df_filtered.sort_values(by=selected_index, ascending=False)


with col2:
    st.header("Результати аналізу")
    
    tab1, tab2, tab3 = st.tabs(["Таблиця", "Графік області", "Порівняння областей"])
    
    with tab1:
        st.subheader(f"Дані для області: {selected_province}")
        st.dataframe(df_filtered[['Year', 'Week', 'Province', selected_index]], use_container_width=True)
        
    with tab2:
        st.subheader(f"Динаміка {selected_index} ({selected_province})")
        df_plot = df_filtered.sort_values(by=['Year', 'Week'])
        fig_single = px.line(df_plot, x='Period', y=selected_index, 
                             title=f'{selected_index} з {year_range[0]} по {year_range[1]} роки',
                             labels={'Period': 'Рік-Тиждень', selected_index: 'Значення'})
        st.plotly_chart(fig_single, use_container_width=True)
        
    with tab3:
        st.subheader(f"Порівняння {selected_index} по всім областям")
        st.info("Виділено лінію обраної області. Інші відображаються для порівняння.")
        
        df_plot_all = df_filtered_all_provinces.sort_values(by=['Year', 'Week'])
        
        fig_comp = px.line(df_plot_all, x='Period', y=selected_index, color='Province',
                           title=f'Порівняння {selected_index} серед усіх областей',
                           labels={'Period': 'Рік-Тиждень', selected_index: 'Значення'})
        
        for trace in fig_comp.data:
            if trace.name == selected_province:
                trace.line.width = 4
            else:
                trace.line.width = 1
                trace.opacity = 0.3
                
        st.plotly_chart(fig_comp, use_container_width=True)