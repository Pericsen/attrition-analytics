import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Título principal
st.title("📊 Dashboard de Análisis de Recursos Humanos")
st.markdown("---")

df = pd.read_csv("./data/HR_Analytics.csv")

# Crear variables derivadas para los análisis
df['Attrition_Numeric'] = (df['Attrition'] == 'Yes').astype(int)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                        labels=['18-30', '31-40', '41-50', '51+'])
df['Income_Group'] = pd.cut(df['MonthlyIncome'], bins=5, precision=0)

# Sidebar para filtros
st.sidebar.header("🔍 Filtros")

# Filtros dinámicos basados en las columnas del dataset
# Filtros categóricos
st.sidebar.subheader("Variables Categóricas")

departments = st.sidebar.multiselect(
    "Departamentos:",
    options=sorted(df['Department'].unique()),
    default=sorted(df['Department'].unique())
)

job_roles = st.sidebar.multiselect(
    "Roles de Trabajo:",
    options=sorted(df['JobRole'].unique()),
    default=sorted(df['JobRole'].unique())
)

genders = st.sidebar.multiselect(
    "Género:",
    options=sorted(df['Gender'].unique()),
    default=sorted(df['Gender'].unique())
)

marital_status = st.sidebar.multiselect(
    "Estado Civil:",
    options=sorted(df['MaritalStatus'].unique()),
    default=sorted(df['MaritalStatus'].unique())
)

education_fields = st.sidebar.multiselect(
    "Campo de Educación:",
    options=sorted(df['EducationField'].unique()),
    default=sorted(df['EducationField'].unique())
)

business_travel = st.sidebar.multiselect(
    "Viajes de Trabajo:",
    options=sorted(df['BusinessTravel'].unique()),
    default=sorted(df['BusinessTravel'].unique())
)

overtime = st.sidebar.multiselect(
    "Horas Extra:",
    options=sorted(df['OverTime'].unique()),
    default=sorted(df['OverTime'].unique())
)

attrition = st.sidebar.multiselect(
    "Attrition:",
    options=sorted(df['Attrition'].unique()),
    default=sorted(df['Attrition'].unique())
)

# Filtros numéricos
st.sidebar.subheader("Variables Numéricas")

age_range = st.sidebar.slider(
    "Edad:",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

job_levels = st.sidebar.multiselect(
    "Niveles de Trabajo:",
    options=sorted(df['JobLevel'].unique()),
    default=sorted(df['JobLevel'].unique())
)

education_range = st.sidebar.slider(
    "Nivel de Educación:",
    min_value=int(df['Education'].min()),
    max_value=int(df['Education'].max()),
    value=(int(df['Education'].min()), int(df['Education'].max()))
)

income_range = st.sidebar.slider(
    "Ingreso Mensual ($):",
    min_value=int(df['MonthlyIncome'].min()),
    max_value=int(df['MonthlyIncome'].max()),
    value=(int(df['MonthlyIncome'].min()), int(df['MonthlyIncome'].max())),
    step=1000
)

distance_range = st.sidebar.slider(
    "Distancia desde Casa (km):",
    min_value=int(df['DistanceFromHome'].min()),
    max_value=int(df['DistanceFromHome'].max()),
    value=(int(df['DistanceFromHome'].min()), int(df['DistanceFromHome'].max()))
)

years_company_range = st.sidebar.slider(
    "Años en la Empresa:",
    min_value=int(df['YearsAtCompany'].min()),
    max_value=int(df['YearsAtCompany'].max()),
    value=(int(df['YearsAtCompany'].min()), int(df['YearsAtCompany'].max()))
)

total_working_years_range = st.sidebar.slider(
    "Años de Experiencia Total:",
    min_value=int(df['TotalWorkingYears'].min()),
    max_value=int(df['TotalWorkingYears'].max()),
    value=(int(df['TotalWorkingYears'].min()), int(df['TotalWorkingYears'].max()))
)

job_satisfaction_range = st.sidebar.slider(
    "Job Satisfaction:",
    min_value=int(df['JobSatisfaction'].min()),
    max_value=int(df['JobSatisfaction'].max()),
    value=(int(df['JobSatisfaction'].min()), int(df['JobSatisfaction'].max()))
)

environment_satisfaction_range = st.sidebar.slider(
    "Environment Satisfaction:",
    min_value=int(df['EnvironmentSatisfaction'].min()),
    max_value=int(df['EnvironmentSatisfaction'].max()),
    value=(int(df['EnvironmentSatisfaction'].min()), int(df['EnvironmentSatisfaction'].max()))
)

relationship_satisfaction_range = st.sidebar.slider(
    "Relationship Satisfaction:",
    min_value=int(df['RelationshipSatisfaction'].min()),
    max_value=int(df['RelationshipSatisfaction'].max()),
    value=(int(df['RelationshipSatisfaction'].min()), int(df['RelationshipSatisfaction'].max()))
)

work_life_balance_range = st.sidebar.slider(
    "Work Life Balance:",
    min_value=int(df['WorkLifeBalance'].min()),
    max_value=int(df['WorkLifeBalance'].max()),
    value=(int(df['WorkLifeBalance'].min()), int(df['WorkLifeBalance'].max()))
)

performance_rating_range = st.sidebar.slider(
    "Rating de Performance:",
    min_value=int(df['PerformanceRating'].min()),
    max_value=int(df['PerformanceRating'].max()),
    value=(int(df['PerformanceRating'].min()), int(df['PerformanceRating'].max()))
)

# Aplicar todos los filtros
filtered_df = df[
    (df['Department'].isin(departments)) &
    (df['JobRole'].isin(job_roles)) &
    (df['Gender'].isin(genders)) &
    (df['MaritalStatus'].isin(marital_status)) &
    (df['EducationField'].isin(education_fields)) &
    (df['BusinessTravel'].isin(business_travel)) &
    (df['OverTime'].isin(overtime)) &
    (df['Attrition'].isin(attrition)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['JobLevel'].isin(job_levels)) &
    (df['Education'].between(education_range[0], education_range[1])) &
    (df['MonthlyIncome'].between(income_range[0], income_range[1])) &
    (df['DistanceFromHome'].between(distance_range[0], distance_range[1])) &
    (df['YearsAtCompany'].between(years_company_range[0], years_company_range[1])) &
    (df['TotalWorkingYears'].between(total_working_years_range[0], total_working_years_range[1])) &
    (df['JobSatisfaction'].between(job_satisfaction_range[0], job_satisfaction_range[1])) &
    (df['EnvironmentSatisfaction'].between(environment_satisfaction_range[0], environment_satisfaction_range[1])) &
    (df['RelationshipSatisfaction'].between(relationship_satisfaction_range[0], relationship_satisfaction_range[1])) &
    (df['WorkLifeBalance'].between(work_life_balance_range[0], work_life_balance_range[1])) &
    (df['PerformanceRating'].between(performance_rating_range[0], performance_rating_range[1]))
]

# Métricas principales
st.header("📈 Métricas Principales")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_employees = len(filtered_df)
    st.metric("Total Empleados", total_employees)

with col2:
    attrition_rate = (filtered_df['Attrition'] == 'Yes').mean() * 100
    st.metric("Tasa de Rotación", f"{attrition_rate:.1f}%")

with col3:
    avg_satisfaction = filtered_df['JobSatisfaction'].mean()
    st.metric("Satisfacción Promedio", f"{avg_satisfaction:.1f}/5")

with col4:
    avg_income = filtered_df['MonthlyIncome'].mean()
    st.metric("Ingreso Promedio", f"${avg_income:,.0f}")

with col5:
    avg_years = filtered_df['YearsAtCompany'].mean()
    st.metric("Años Promedio", f"{avg_years:.1f}")

st.markdown("---")

# Gráficos principales
st.header("📊 Análisis Visual")

# Fila 1: Attrition y Demographics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Rotación por Departamento")
    attrition_dept = filtered_df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
    attrition_pct = attrition_dept.div(attrition_dept.sum(axis=1), axis=0) * 100
    
    fig_attrition = px.bar(
        x=attrition_pct.index,
        y=attrition_pct['Yes'],
        title="Tasa de Rotación por Departamento",
        labels={'x': 'Departamento', 'y': 'Tasa de Rotación (%)'},
        color=attrition_pct['Yes'],
        color_continuous_scale='Reds'
    )
    fig_attrition.update_layout(showlegend=False)
    st.plotly_chart(fig_attrition, use_container_width=True)

with col2:
    st.subheader("Distribución por Género")
    gender_counts = filtered_df['Gender'].value_counts()
    
    fig_gender = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title="Distribución de Empleados por Género"
    )
    st.plotly_chart(fig_gender, use_container_width=True)

# Fila 2: Satisfacción y Compensación
col1, col2 = st.columns(2)

with col1:
    st.subheader("Satisfacción Laboral")
    satisfaction_counts = filtered_df['JobSatisfaction'].value_counts().sort_index()
    
    fig_satisfaction = px.bar(
        x=satisfaction_counts.index,
        y=satisfaction_counts.values,
        title="Distribución de Satisfacción Laboral",
        labels={'x': 'Nivel de Satisfacción', 'y': 'Número de Empleados'},
        color=satisfaction_counts.values,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_satisfaction, use_container_width=True)

with col2:
    st.subheader("Ingresos por Nivel de Trabajo")
    income_by_level = filtered_df.groupby('JobLevel')['MonthlyIncome'].mean().reset_index()
    
    fig_income = px.bar(
        income_by_level,
        x='JobLevel',
        y='MonthlyIncome',
        title="Ingreso Promedio por Nivel de Trabajo",
        text='MonthlyIncome',
        color='MonthlyIncome',
        color_continuous_scale='Blues'
    )
    fig_income.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_income.update_layout(
        xaxis_title="Nivel de Trabajo",
        yaxis_title="Ingreso Mensual Promedio ($)",
        showlegend=False
    )
    st.plotly_chart(fig_income, use_container_width=True)

# Fila 3: Análisis de Edad y Experiencia
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de Edad")
    fig_age = px.histogram(
        filtered_df,
        x='Age',
        nbins=20,
        title="Distribución de Edad de los Empleados",
        color_discrete_sequence=['lightblue']
    )
    fig_age.update_layout(
        xaxis_title="Edad",
        yaxis_title="Frecuencia"
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.subheader("Experiencia vs Ingreso")
    fig_scatter = px.scatter(
        filtered_df,
        x='TotalWorkingYears',
        y='MonthlyIncome',
        color='JobLevel',
        title="Relación entre Experiencia e Ingreso",
        opacity=0.6
    )
    fig_scatter.update_layout(
        xaxis_title="Años de Experiencia Total",
        yaxis_title="Ingreso Mensual ($)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Análisis de Horas Extra y Work-Life Balance
st.header("⚖️ Análisis de Work-Life Balance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Horas Extra por Departamento")
    overtime_dept = pd.crosstab(filtered_df['Department'], filtered_df['OverTime'], normalize='index') * 100
    
    fig_overtime = px.bar(
        x=overtime_dept.index,
        y=overtime_dept['Yes'],
        title="Porcentaje de Empleados con Horas Extra",
        labels={'x': 'Departamento', 'y': 'Porcentaje con Horas Extra (%)'},
        color=overtime_dept['Yes'],
        color_continuous_scale='Oranges'
    )
    st.plotly_chart(fig_overtime, use_container_width=True)

with col2:
    st.subheader("Work-Life Balance")
    wlb_counts = filtered_df['WorkLifeBalance'].value_counts().sort_index()
    
    fig_wlb = px.bar(
        x=wlb_counts.index,
        y=wlb_counts.values,
        title="Distribución de Work-Life Balance",
        labels={'x': 'Nivel de Work-Life Balance', 'y': 'Número de Empleados'},
        color=wlb_counts.values,
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig_wlb, use_container_width=True)

# Heatmap de Correlaciones
st.header("🔥 Mapa de Correlaciones")

# Seleccionar solo columnas numéricas
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
correlation_matrix = filtered_df[numeric_cols].corr()

fig_heatmap = px.imshow(
    correlation_matrix,
    title="Matriz de Correlación de Variables Numéricas",
    color_continuous_scale='RdBu',
    aspect='auto'
)
fig_heatmap.update_layout(height=600)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Análisis de Rotación Detallado
st.header("🔄 Análisis Detallado de Rotación - Insights Clave")

# Crear pestañas para diferentes análisis
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Fatiga Operativa", "Early-Career & Underpaid", "Pared de los 3 Años", "Equidad Interna", "Clima como Amplificador"])

with tab1:
    st.subheader("🔥 Fatiga Operativa - Eje Gravitacional")
    st.markdown("**Casi dos de cada tres renuncias pertenecen a personas que trabajan horas extra o viajan seguido**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Análisis de OverTime y Travel
        fatigue_analysis = filtered_df.copy()
        fatigue_analysis['Fatigue_Factor'] = 'Normal'
        fatigue_analysis.loc[(fatigue_analysis['OverTime'] == 'Yes') | 
                           (fatigue_analysis['BusinessTravel'] == 'Travel_Frequently'), 'Fatigue_Factor'] = 'Alta Fatiga'
        
        fatigue_attrition = fatigue_analysis.groupby('Fatigue_Factor')['Attrition_Numeric'].agg(['count', 'sum', 'mean']).reset_index()
        fatigue_attrition.columns = ['Fatigue_Factor', 'Total', 'Attrition_Count', 'Attrition_Rate']
        
        fig_fatigue = px.bar(
            fatigue_attrition,
            x='Fatigue_Factor',
            y='Attrition_Rate',
            title="Tasa de Attrition: Normal vs Alta Fatiga",
            text='Attrition_Rate',
            color='Attrition_Rate',
            color_continuous_scale='Reds'
        )
        fig_fatigue.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_fatigue.update_layout(showlegend=False)
        st.plotly_chart(fig_fatigue, use_container_width=True)
        
        # Estadísticas
        st.write("**Estadísticas de Fatiga:**")
        st.dataframe(fatigue_attrition)
    
    with col2:
        # Análisis de combinación OT + Travel por JobRole
        ot_travel_analysis = filtered_df.groupby(['JobRole', 'OverTime', 'BusinessTravel'])['Attrition_Numeric'].agg(['count', 'mean']).reset_index()
        ot_travel_analysis.columns = ['JobRole', 'OverTime', 'BusinessTravel', 'Count', 'Attrition_Rate']
        ot_travel_analysis = ot_travel_analysis[ot_travel_analysis['Count'] >= 3]
        
        # Filtrar casos de alto riesgo (OT + Travel)
        high_risk = ot_travel_analysis[
            (ot_travel_analysis['OverTime'] == 'Yes') & 
            (ot_travel_analysis['BusinessTravel'].isin(['Travel_Frequently', 'Travel_Rarely']))
        ].sort_values('Attrition_Rate', ascending=False)
        
        if not high_risk.empty:
            fig_high_risk = px.bar(
                high_risk.head(8),
                x='Attrition_Rate',
                y='JobRole',
                orientation='h',
                title="Roles con Mayor Riesgo (OT + Travel)",
                text='Attrition_Rate',
                color='Attrition_Rate',
                color_continuous_scale='Reds'
            )
            fig_high_risk.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_high_risk.update_layout(showlegend=False)
            st.plotly_chart(fig_high_risk, use_container_width=True)
        
        st.write("**Top Combinaciones de Riesgo:**")
        if not high_risk.empty:
            st.dataframe(high_risk[['JobRole', 'Count', 'Attrition_Rate']].round(3))

with tab2:
    st.subheader("Early-Career & Underpaid")
    st.markdown("**Tramo 18-30 + Q1 salarial + Single explica más de la mitad de las bajas**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Crear análisis de perfil de alto riesgo
        early_career = filtered_df.copy()
        income_q1 = early_career['MonthlyIncome'].quantile(0.25)
        
        early_career['Risk_Profile'] = 'Bajo Riesgo'
        early_career.loc[
            (early_career['Age'] <= 30) & 
            (early_career['MonthlyIncome'] <= income_q1) & 
            (early_career['MaritalStatus'] == 'Single'), 
            'Risk_Profile'
        ] = 'Alto Riesgo (18-30, Q1, Single)'
        
        risk_analysis = early_career.groupby('Risk_Profile')['Attrition_Numeric'].agg(['count', 'sum', 'mean']).reset_index()
        risk_analysis.columns = ['Risk_Profile', 'Total', 'Attrition_Count', 'Attrition_Rate']
        
        fig_risk = px.bar(
            risk_analysis,
            x='Risk_Profile',
            y='Attrition_Rate',
            title="Perfil de Riesgo: Early-Career & Underpaid",
            text='Attrition_Rate',
            color='Attrition_Rate',
            color_continuous_scale='Oranges'
        )
        fig_risk.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_risk.update_layout(showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.write("**Análisis de Perfil de Riesgo:**")
        st.dataframe(risk_analysis)
    
    with col2:
        # Análisis por quartiles de edad e ingreso
        age_income_analysis = filtered_df.copy()
        age_income_analysis['Age_Quartile'] = pd.qcut(age_income_analysis['Age'], 4, labels=['Q1(18-30)', 'Q2(31-38)', 'Q3(39-47)', 'Q4(48+)'])
        age_income_analysis['Income_Quartile'] = pd.qcut(age_income_analysis['MonthlyIncome'], 4, labels=['Q1(Bajo)', 'Q2(Medio-Bajo)', 'Q3(Medio-Alto)', 'Q4(Alto)'])
        
        heatmap_data = age_income_analysis.pivot_table(
            values='Attrition_Numeric',
            index='Age_Quartile',
            columns='Income_Quartile',
            aggfunc='mean'
        )
        
        fig_heatmap_age_income = px.imshow(
            heatmap_data,
            title="Heatmap: Attrition por Edad vs Ingreso",
            color_continuous_scale='Reds',
            text_auto='.2f'
        )
        st.plotly_chart(fig_heatmap_age_income, use_container_width=True)

with tab3:
    st.subheader("📊 Pared de los 3 Años")
    st.markdown("**La mediana YearsAtCompany para desertores es 3; quienes superan esa barrera estabilizan su riesgo**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Análisis de la "pared de los 3 años"
        years_analysis = filtered_df.copy()
        years_analysis['Years_Group'] = pd.cut(
            years_analysis['YearsAtCompany'], 
            bins=[0, 1, 2, 3, 5, 10, 50], 
            labels=['0-1', '1-2', '2-3', '3-5', '5-10', '10+']
        )
        
        years_attrition = years_analysis.groupby('Years_Group')['Attrition_Numeric'].agg(['count', 'sum', 'mean']).reset_index()
        years_attrition.columns = ['Years_Group', 'Total', 'Attrition_Count', 'Attrition_Rate']
        
        fig_years = px.bar(
            years_attrition,
            x='Years_Group',
            y='Attrition_Rate',
            title="Tasa de Attrition por Años en la Empresa",
            text='Attrition_Rate',
            color='Attrition_Rate',
            color_continuous_scale='Blues'
        )
        fig_years.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_years.update_layout(showlegend=False)
        st.plotly_chart(fig_years, use_container_width=True)
        
        st.write("**Análisis por Años en la Empresa:**")
        st.dataframe(years_attrition)
    
    with col2:
        # Stagnation Index (YearsInCurrentRole / YearsAtCompany)
        stagnation_analysis = filtered_df.copy()
        stagnation_analysis['Stagnation_Index'] = stagnation_analysis['YearsInCurrentRole'] / (stagnation_analysis['YearsAtCompany'] + 0.1)  # +0.1 para evitar división por 0
        stagnation_analysis['Stagnation_Level'] = pd.cut(
            stagnation_analysis['Stagnation_Index'],
            bins=[0, 0.4, 0.7, 1.0],
            labels=['Bajo (<0.4)', 'Medio (0.4-0.7)', 'Alto (≥0.7)']
        )
        
        stagnation_attrition = stagnation_analysis.groupby('Stagnation_Level')['Attrition_Numeric'].agg(['count', 'mean']).reset_index()
        stagnation_attrition.columns = ['Stagnation_Level', 'Count', 'Attrition_Rate']
        
        fig_stagnation = px.bar(
            stagnation_attrition,
            x='Stagnation_Level',
            y='Attrition_Rate',
            title="Stagnation Index vs Attrition",
            text='Attrition_Rate',
            color='Attrition_Rate',
            color_continuous_scale='Purples'
        )
        fig_stagnation.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_stagnation.update_layout(showlegend=False)
        st.plotly_chart(fig_stagnation, use_container_width=True)
        
        st.write("**Análisis de Stagnation Index:**")
        st.dataframe(stagnation_attrition)

with tab4:
    st.subheader("⚖️ Equidad Interna > Dinero Absoluto")
    st.markdown("**Estar en el cuartil inferior de tu banda eleva el riesgo casi 3×**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Análisis de equidad por JobLevel
        equity_analysis = filtered_df.copy()
        
        # Calcular quartiles por JobLevel
        equity_data = []
        for level in equity_analysis['JobLevel'].unique():
            level_data = equity_analysis[equity_analysis['JobLevel'] == level]
            if len(level_data) > 4:  # Mínimo para calcular quartiles
                q25 = level_data['MonthlyIncome'].quantile(0.25)
                level_data_copy = level_data.copy()
                level_data_copy['Income_Quartile_Within_Level'] = 'Q2-Q4'
                level_data_copy.loc[level_data_copy['MonthlyIncome'] <= q25, 'Income_Quartile_Within_Level'] = 'Q1 (Inferior)'
                equity_data.append(level_data_copy)
        
        if equity_data:
            equity_df = pd.concat(equity_data)
            equity_summary = equity_df.groupby(['JobLevel', 'Income_Quartile_Within_Level'])['Attrition_Numeric'].agg(['count', 'mean']).reset_index()
            equity_summary.columns = ['JobLevel', 'Income_Quartile_Within_Level', 'Count', 'Attrition_Rate']
            
            fig_equity = px.bar(
                equity_summary,
                x='JobLevel',
                y='Attrition_Rate',
                color='Income_Quartile_Within_Level',
                title="Equidad Interna: Attrition por Quartil de Ingreso dentro del Nivel",
                barmode='group'
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            st.write("**Análisis de Equidad Interna:**")
            st.dataframe(equity_summary[equity_summary['Count'] >= 3])
    
    with col2:
        # Análisis de percentiles salariales
        percentile_analysis = filtered_df.copy()
        percentile_analysis['Salary_Percentile'] = percentile_analysis['MonthlyIncome'].rank(pct=True)
        percentile_analysis['Percentile_Group'] = pd.cut(
            percentile_analysis['Salary_Percentile'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['P0-25', 'P25-50', 'P50-75', 'P75-100']
        )
        
        percentile_attrition = percentile_analysis.groupby('Percentile_Group')['Attrition_Numeric'].agg(['count', 'mean']).reset_index()
        percentile_attrition.columns = ['Percentile_Group', 'Count', 'Attrition_Rate']
        
        fig_percentile = px.bar(
            percentile_attrition,
            x='Percentile_Group',
            y='Attrition_Rate',
            title="Attrition por Percentil Salarial Global",
            text='Attrition_Rate',
            color='Attrition_Rate',
            color_continuous_scale='Greens'
        )
        fig_percentile.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_percentile.update_layout(showlegend=False)
        st.plotly_chart(fig_percentile, use_container_width=True)

with tab5:
    st.subheader("🌡️ Clima como Amplificador")
    st.markdown("**Job/Environment Satisfaction niveles 1-2 añaden +8 p.p. de riesgo a cualquier otro driver**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Análisis de satisfacción como amplificador
        climate_analysis = filtered_df.copy()
        climate_analysis['Low_Satisfaction'] = (
            (climate_analysis['JobSatisfaction'] <= 2) | 
            (climate_analysis['EnvironmentSatisfaction'] <= 2)
        )
        climate_analysis['Has_Other_Risk'] = (
            (climate_analysis['OverTime'] == 'Yes') | 
            (climate_analysis['BusinessTravel'] == 'Travel_Frequently') |
            (climate_analysis['Age'] <= 30)
        )
        
        # Crear combinaciones
        climate_analysis['Risk_Combination'] = 'Sin Factores de Riesgo'
        climate_analysis.loc[
            climate_analysis['Has_Other_Risk'] & ~climate_analysis['Low_Satisfaction'], 
            'Risk_Combination'
        ] = 'Solo Otros Factores'
        climate_analysis.loc[
            ~climate_analysis['Has_Other_Risk'] & climate_analysis['Low_Satisfaction'], 
            'Risk_Combination'
        ] = 'Solo Baja Satisfacción'
        climate_analysis.loc[
            climate_analysis['Has_Other_Risk'] & climate_analysis['Low_Satisfaction'], 
            'Risk_Combination'
        ] = 'Otros Factores + Baja Satisfacción'
        
        climate_summary = climate_analysis.groupby('Risk_Combination')['Attrition_Numeric'].agg(['count', 'mean']).reset_index()
        climate_summary.columns = ['Risk_Combination', 'Count', 'Attrition_Rate']
        
        fig_climate = px.bar(
            climate_summary,
            x='Risk_Combination',
            y='Attrition_Rate',
            title="Efecto Amplificador del Clima Laboral",
            text='Attrition_Rate',
            color='Attrition_Rate',
            color_continuous_scale='Reds'
        )
        fig_climate.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_climate.update_layout(showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig_climate, use_container_width=True)
        
        st.write("**Análisis del Efecto Amplificador:**")
        st.dataframe(climate_summary)
    
    with col2:
        # Heatmap de satisfacción combinada
        satisfaction_heatmap = filtered_df.pivot_table(
            values='Attrition_Numeric',
            index='JobSatisfaction',
            columns='EnvironmentSatisfaction',
            aggfunc='mean'
        )
        
        fig_satisfaction_heatmap = px.imshow(
            satisfaction_heatmap,
            title="Heatmap: Job vs Environment Satisfaction",
            color_continuous_scale='Reds',
            text_auto='.2f'
        )
        fig_satisfaction_heatmap.update_layout(
            xaxis_title="Environment Satisfaction",
            yaxis_title="Job Satisfaction"
        )
        st.plotly_chart(fig_satisfaction_heatmap, use_container_width=True)
        
        # Análisis detallado de satisfacción
        low_sat_analysis = filtered_df.copy()
        low_sat_analysis['Satisfaction_Level'] = 'Alta (3-4)'
        low_sat_analysis.loc[
            (low_sat_analysis['JobSatisfaction'] <= 2) | 
            (low_sat_analysis['EnvironmentSatisfaction'] <= 2),
            'Satisfaction_Level'
        ] = 'Baja (1-2)'
        
        sat_summary = low_sat_analysis.groupby('Satisfaction_Level')['Attrition_Numeric'].agg(['count', 'mean']).reset_index()
        sat_summary.columns = ['Satisfaction_Level', 'Count', 'Attrition_Rate']
        
        st.write("**Impacto de Baja Satisfacción:**")
        st.dataframe(sat_summary)

# ===================
# GRÁFICOS ADICIONALES
# ====================

# Análisis por Grupos de Edad e Ingresos
st.header("📊 Análisis por Grupos Demográficos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Attrition por Grupo de Edad")
    age_analysis = filtered_df.groupby('Age_Group')['Attrition_Numeric'].agg(['count', 'sum', 'mean']).reset_index()
    age_analysis.columns = ['Age_Group', 'Total', 'Attrition_Count', 'Attrition_Rate']
    
    fig_age_groups = px.bar(
        age_analysis,
        x='Age_Group',
        y='Attrition_Rate',
        title="Tasa de Attrition por Grupo de Edad",
        labels={'Age_Group': 'Grupo de Edad', 'Attrition_Rate': 'Tasa de Attrition'},
        text='Attrition_Rate',
        color='Attrition_Rate',
        color_continuous_scale='Blues'
    )
    fig_age_groups.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_age_groups.update_layout(showlegend=False)
    st.plotly_chart(fig_age_groups, use_container_width=True)
    
    # Mostrar tabla
    st.write("**Estadísticas por Grupo de Edad:**")
    st.dataframe(age_analysis.round(3))

with col2:
    st.subheader("Attrition por Grupo de Ingresos")
    income_analysis = filtered_df.groupby('Income_Group')['Attrition_Numeric'].agg(['count', 'sum', 'mean']).reset_index()
    income_analysis.columns = ['Income_Group', 'Total', 'Attrition_Count', 'Attrition_Rate']
    
    # Convertir Income_Group a string para mejor visualización
    income_analysis['Income_Label'] = income_analysis['Income_Group'].apply(
        lambda x: f'${int(x.left/1000)}K-{int(x.right/1000)}K' if pd.notna(x) else 'N/A'
    )
    
    fig_income_groups = px.bar(
        income_analysis,
        x='Income_Label',
        y='Attrition_Rate',
        title="Tasa de Attrition por Grupo de Ingresos",
        labels={'Income_Label': 'Grupo de Ingresos', 'Attrition_Rate': 'Tasa de Attrition'},
        text='Attrition_Rate',
        color='Attrition_Rate',
        color_continuous_scale='Reds'
    )
    fig_income_groups.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_income_groups.update_layout(showlegend=False, xaxis_tickangle=45)
    st.plotly_chart(fig_income_groups, use_container_width=True)
    
    # Mostrar tabla
    st.write("**Estadísticas por Grupo de Ingresos:**")
    display_income = income_analysis[['Income_Label', 'Total', 'Attrition_Count', 'Attrition_Rate']].round(3)
    st.dataframe(display_income)

# Análisis Multivariado
st.header("🔄 Análisis Multivariado de Attrition")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Heatmap: Departamento vs OverTime")
    pivot_data = filtered_df.pivot_table(values='Attrition_Numeric', 
                                       index='Department', 
                                       columns='OverTime', 
                                       aggfunc='mean')
    
    fig_heatmap_multi = px.imshow(
        pivot_data,
        title="Tasa de Attrition: Departamento vs OverTime",
        color_continuous_scale='Reds',
        aspect='auto',
        text_auto='.3f'
    )
    st.plotly_chart(fig_heatmap_multi, use_container_width=True)

with col2:
    st.subheader("Top Combinaciones de Alto Riesgo")
    multi_analysis = filtered_df.groupby(['Department', 'Gender', 'OverTime'])['Attrition_Numeric'].agg(['count', 'mean']).reset_index()
    multi_analysis.columns = ['Department', 'Gender', 'OverTime', 'Count', 'Attrition_Rate']
    multi_analysis = multi_analysis[multi_analysis['Count'] >= 5]  # Filtrar grupos pequeños
    multi_analysis = multi_analysis.sort_values('Attrition_Rate', ascending=False).head(10)
    
    # Crear combinación para mejor visualización
    multi_analysis['Combination'] = multi_analysis['Department'] + ' + ' + multi_analysis['Gender'] + ' + ' + multi_analysis['OverTime']
    
    fig_multi = px.bar(
        multi_analysis.head(8),  # Top 8 para mejor visualización
        x='Attrition_Rate',
        y='Combination',
        orientation='h',
        title="Top 8 Combinaciones con Mayor Attrition",
        text='Attrition_Rate',
        color='Attrition_Rate',
        color_continuous_scale='Reds'
    )
    fig_multi.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_multi.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_multi, use_container_width=True)
    
    st.write("**Tabla Detallada:**")
    st.dataframe(multi_analysis[['Department', 'Gender', 'OverTime', 'Count', 'Attrition_Rate']].round(3))

# Análisis por Variables Categóricas
st.header("📈 Análisis por Variables Categóricas")

categorical_cols = ['Department', 'JobRole', 'Gender', 'MaritalStatus', 'EducationField', 'BusinessTravel', 'OverTime']
available_categorical = [col for col in categorical_cols if col in filtered_df.columns]

if available_categorical:
    # Selector de variable
    selected_var = st.selectbox("Selecciona variable para análisis detallado:", available_categorical)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Distribución de {selected_var}")
        value_counts = filtered_df[selected_var].value_counts()
        
        fig_dist = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribución de {selected_var}",
            labels={'x': selected_var, 'y': 'Número de Empleados'},
            text=value_counts.values,
            color=value_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_dist.update_traces(textposition='outside')
        fig_dist.update_layout(showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader(f"Attrition por {selected_var}")
        ct = pd.crosstab(filtered_df[selected_var], filtered_df['Attrition'], normalize='index') * 100
        
        fig_attrition_cat = px.bar(
            x=ct.index,
            y=ct['Yes'],
            title=f"Tasa de Attrition por {selected_var}",
            labels={'x': selected_var, 'y': 'Tasa de Attrition (%)'},
            text=ct['Yes'],
            color=ct['Yes'],
            color_continuous_scale='Reds'
        )
        fig_attrition_cat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_attrition_cat.update_layout(showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig_attrition_cat, use_container_width=True)
        
        # Mostrar tabla de contingencia
        st.write("**Tabla de Contingencia:**")
        ct_abs = pd.crosstab(filtered_df[selected_var], filtered_df['Attrition'])
        st.dataframe(ct_abs)

# Vista Comparativa - Todas las Variables
st.header("📊 Vista Comparativa - Todas las Variables")

st.subheader("Resumen de Attrition por Todas las Variables")

# Crear un resumen de todas las variables
summary_data = []

for col in available_categorical:
    ct = pd.crosstab(filtered_df[col], filtered_df['Attrition'], normalize='index') * 100
    for category in ct.index:
        summary_data.append({
            'Variable': col,
            'Categoria': category,
            'Tasa_Attrition': ct.loc[category, 'Yes'],
            'Count': len(filtered_df[filtered_df[col] == category])
        })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df[summary_df['Count'] >= 5]  # Filtrar categorías pequeñas
summary_df = summary_df.sort_values('Tasa_Attrition', ascending=False)

# Gráfico de resumen
fig_summary = px.scatter(
    summary_df,
    x='Count',
    y='Tasa_Attrition',
    color='Variable',
    size='Count',
    hover_data=['Categoria'],
    title="Attrition Rate vs Tamaño de Grupo por Variable",
    labels={'Count': 'Número de Empleados', 'Tasa_Attrition': 'Tasa de Attrition (%)'}
)
fig_summary.update_layout(height=500)
st.plotly_chart(fig_summary, use_container_width=True)

# Top 10 categorías con mayor attrition
st.subheader("Top 10 Categorías con Mayor Attrition")
top_attrition = summary_df.head(10)

fig_top = px.bar(
    top_attrition,
    x='Tasa_Attrition',
    y='Categoria',
    color='Variable',
    orientation='h',
    title="Top 10 Categorías con Mayor Tasa de Attrition",
    text='Tasa_Attrition'
)
fig_top.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig_top.update_layout(height=500)
st.plotly_chart(fig_top, use_container_width=True)

# Tabla resumen
st.write("**Resumen Completo:**")
display_summary = summary_df[['Variable', 'Categoria', 'Tasa_Attrition', 'Count']].round(2)
st.dataframe(display_summary, height=300)

# Tabla de datos resumida
st.header("📋 Resumen de Datos")

summary_stats = filtered_df.describe()
st.dataframe(summary_stats, use_container_width=True)

# Sección de exportación
st.header("💾 Exportar Datos")

col1, col2 = st.columns(2)

with col1:
    if st.button("Descargar Datos Filtrados (CSV)"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name="hr_analytics_filtered.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Descargar Estadísticas Resumidas"):
        stats_csv = summary_stats.to_csv()
        st.download_button(
            label="Descargar Estadísticas CSV",
            data=stats_csv,
            file_name="hr_analytics_summary.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("Dashboard creado con Streamlit 🚀 | Datos de HR Analytics")