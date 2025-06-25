import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio


class TargetEncoder:
    """
    Target Encoder con suavizado para evitar overfitting
    """
    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encodings = {}
        self.global_mean = None
        
    def fit(self, X, y):
        """
        Ajusta el encoder con los datos de entrenamiento
        """
        self.global_mean = y.mean()
        
        # Crear un DataFrame temporal combinando X e y
        temp_df = X.copy()
        temp_df['target'] = y.values
        
        for column in X.columns:
            # Calcular estadísticas por categoría
            stats = temp_df.groupby(column)['target'].agg(['count', 'mean']).reset_index()
            stats.columns = [column, 'count', 'mean']
            
            # Aplicar suavizado
            # Formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            stats['smoothed_mean'] = (
                (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / 
                (stats['count'] + self.smoothing)
            )
            
            # Filtrar categorías con pocas muestras
            stats = stats[stats['count'] >= self.min_samples_leaf]
            
            # Guardar encoding
            encoding_dict = dict(zip(stats[column], stats['smoothed_mean']))
            self.encodings[column] = encoding_dict
            
        return self
    
    def transform(self, X):
        """
        Transforma las variables categóricas usando el encoding ajustado
        """
        X_encoded = X.copy()
        
        for column in X.columns:
            if column in self.encodings:
                # Mapear valores conocidos, usar global_mean para valores nuevos
                X_encoded[column] = X[column].map(self.encodings[column]).fillna(self.global_mean)
            
        return X_encoded
    
    def fit_transform(self, X, y):
        """
        Ajusta y transforma en un solo paso
        """
        return self.fit(X, y).transform(X)
# Configurar tema global para todos los plots
# pio.templates["anthropic"] = {
#     'layout': {
#         'colorway': ['#D4A574', '#5A8FC8', '#5AC86F', '#C85A5A', '#C8905A', '#8F5AC8'],
#         'paper_bgcolor': '#F5F2E8',  # Fondo coincide con tu tema
#         'plot_bgcolor': '#E8E2D5',   # Fondo del área de ploteo
#         'font': {'color': '#4A453E'}, # Color del texto
#         'title': {'font': {'color': '#4A453E', 'size': 16}, 'x': 0.5},
#         'xaxis': {'gridcolor': '#E8E2D5', 'linecolor': '#4A453E'},
#         'yaxis': {'gridcolor': '#E8E2D5', 'linecolor': '#4A453E'},
#     }
# }
# pio.templates.default = "anthropic"

# Configuración de la página
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("Dashboard de Análisis de Recursos Humanos - Attrition")
st.markdown("---")

df = pd.read_csv("./data/HR_Analytics.csv")

# Crear variables derivadas para los análisis
df['Attrition_Numeric'] = (df['Attrition'] == 'Yes').astype(int)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                        labels=['18-30', '31-40', '41-50', '51+'])
df['Income_Group'] = pd.cut(df['MonthlyIncome'], bins=5, precision=0)

# Sidebar para filtros
st.sidebar.header("Filtros")

# Filtros categóricos con dropdown
st.sidebar.subheader("Variables Categóricas")

departments = st.sidebar.multiselect(
    "Departamentos:",
    options=sorted(df['Department'].unique()),
    default=sorted(df['Department'].unique()),
    help="Selecciona uno o más departamentos"
)

job_roles = st.sidebar.multiselect(
    "Roles de Trabajo:",
    options=sorted(df['JobRole'].unique()),
    default=sorted(df['JobRole'].unique()),
    help="Selecciona uno o más roles de trabajo"
)

genders = st.sidebar.multiselect(
    "Género:",
    options=sorted(df['Gender'].unique()),
    default=sorted(df['Gender'].unique()),
    help="Selecciona género"
)

marital_status = st.sidebar.multiselect(
    "Estado Civil:",
    options=sorted(df['MaritalStatus'].unique()),
    default=sorted(df['MaritalStatus'].unique()),
    help="Selecciona estado civil"
)

education_fields = st.sidebar.multiselect(
    "Campo de Educación:",
    options=sorted(df['EducationField'].unique()),
    default=sorted(df['EducationField'].unique()),
    help="Selecciona campo de educación"
)

business_travel = st.sidebar.multiselect(
    "Viajes de Trabajo:",
    options=sorted(df['BusinessTravel'].unique()),
    default=sorted(df['BusinessTravel'].unique()),
    help="Selecciona frecuencia de viajes"
)

overtime = st.sidebar.multiselect(
    "Horas Extra:",
    options=sorted(df['OverTime'].unique()),
    default=sorted(df['OverTime'].unique()),
    help="Selecciona si trabajan horas extra"
)

attrition = st.sidebar.multiselect(
    "Attrition:",
    options=sorted(df['Attrition'].unique()),
    default=sorted(df['Attrition'].unique()),
    help="Selecciona estatus de attrition"
)

# Filtros numéricos
st.sidebar.subheader("Variables Numéricas")

age_range = st.sidebar.slider(
    "Edad:",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max())),
    help="Rango de edad"
)

job_levels = st.sidebar.multiselect(
    "Niveles de Trabajo:",
    options=sorted(df['JobLevel'].unique()),
    default=sorted(df['JobLevel'].unique()),
    help="Selecciona niveles de trabajo"
)

education_levels = st.sidebar.multiselect(
    "Nivel de Educación:",
    options=sorted(df['Education'].unique()),
    default=sorted(df['Education'].unique()),
    help="Selecciona niveles de educación (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor)"
)

income_range = st.sidebar.slider(
    "Ingreso Mensual ($):",
    min_value=int(df['MonthlyIncome'].min()),
    max_value=int(df['MonthlyIncome'].max()),
    value=(int(df['MonthlyIncome'].min()), int(df['MonthlyIncome'].max())),
    step=1000,
    help="Rango de ingreso mensual"
)

distance_range = st.sidebar.slider(
    "Distancia desde Casa (km):",
    min_value=int(df['DistanceFromHome'].min()),
    max_value=int(df['DistanceFromHome'].max()),
    value=(int(df['DistanceFromHome'].min()), int(df['DistanceFromHome'].max())),
    help="Distancia desde casa al trabajo"
)

years_company_range = st.sidebar.slider(
    "Años en la Empresa:",
    min_value=int(df['YearsAtCompany'].min()),
    max_value=int(df['YearsAtCompany'].max()),
    value=(int(df['YearsAtCompany'].min()), int(df['YearsAtCompany'].max())),
    help="Años trabajando en la empresa"
)

total_working_years_range = st.sidebar.slider(
    "Años de Experiencia Total:",
    min_value=int(df['TotalWorkingYears'].min()),
    max_value=int(df['TotalWorkingYears'].max()),
    value=(int(df['TotalWorkingYears'].min()), int(df['TotalWorkingYears'].max())),
    help="Total de años de experiencia laboral"
)

job_satisfaction_selected = st.sidebar.multiselect(
    "Job Satisfaction:",
    options=sorted(df['JobSatisfaction'].unique()),
    default=sorted(df['JobSatisfaction'].unique()),
    help="Nivel de satisfacción laboral (1=Low, 2=Medium, 3=High, 4=Very High)"
)

environment_satisfaction_selected = st.sidebar.multiselect(
    "Environment Satisfaction:",
    options=sorted(df['EnvironmentSatisfaction'].unique()),
    default=sorted(df['EnvironmentSatisfaction'].unique()),
    help="Satisfacción con el ambiente de trabajo"
)

relationship_satisfaction_selected = st.sidebar.multiselect(
    "Relationship Satisfaction:",
    options=sorted(df['RelationshipSatisfaction'].unique()),
    default=sorted(df['RelationshipSatisfaction'].unique()),
    help="Satisfacción con las relaciones en el trabajo"
)

work_life_balance_selected = st.sidebar.multiselect(
    "Work Life Balance:",
    options=sorted(df['WorkLifeBalance'].unique()),
    default=sorted(df['WorkLifeBalance'].unique()),
    help="Nivel de balance vida-trabajo"
)

performance_rating_selected = st.sidebar.multiselect(
    "Rating de Performance:",
    options=sorted(df['PerformanceRating'].unique()),
    default=sorted(df['PerformanceRating'].unique()),
    help="Rating de performance del empleado"
)

# Botón para limpiar filtros
if st.sidebar.button("Limpiar Todos los Filtros"):
    st.experimental_rerun()

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
    (df['Education'].isin(education_levels)) &
    (df['MonthlyIncome'].between(income_range[0], income_range[1])) &
    (df['DistanceFromHome'].between(distance_range[0], distance_range[1])) &
    (df['YearsAtCompany'].between(years_company_range[0], years_company_range[1])) &
    (df['TotalWorkingYears'].between(total_working_years_range[0], total_working_years_range[1])) &
    (df['JobSatisfaction'].isin(job_satisfaction_selected)) &
    (df['EnvironmentSatisfaction'].isin(environment_satisfaction_selected)) &
    (df['RelationshipSatisfaction'].isin(relationship_satisfaction_selected)) &
    (df['WorkLifeBalance'].isin(work_life_balance_selected)) &
    (df['PerformanceRating'].isin(performance_rating_selected))
]

# Mostrar información sobre filtros aplicados
if len(filtered_df) < len(df):
    st.sidebar.info(f"Mostrando {len(filtered_df):,} de {len(df):,} empleados ({len(filtered_df)/len(df)*100:.1f}%)")
else:
    st.sidebar.success(f"Mostrando todos los {len(df):,} empleados")

# Métricas principales
st.header("Métricas Principales")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_employees = len(filtered_df)
    st.metric("Total Empleados", f"{total_employees:,}")

with col2:
    if len(filtered_df) > 0:
        attrition_rate = (filtered_df['Attrition'] == 'Yes').mean() * 100
        st.metric("Tasa de Rotación", f"{attrition_rate:.1f}%")
    else:
        st.metric("Tasa de Rotación", "N/A")

with col3:
    if len(filtered_df) > 0:
        avg_satisfaction = filtered_df['JobSatisfaction'].mean()
        st.metric("Satisfacción Promedio", f"{avg_satisfaction:.1f}/4")
    else:
        st.metric("Satisfacción Promedio", "N/A")

with col4:
    if len(filtered_df) > 0:
        avg_income = filtered_df['MonthlyIncome'].mean()
        st.metric("Ingreso Promedio", f"${avg_income:,.0f}")
    else:
        st.metric("Ingreso Promedio", "N/A")

with col5:
    if len(filtered_df) > 0:
        avg_years = filtered_df['YearsAtCompany'].mean()
        st.metric("Años Promedio", f"{avg_years:.1f}")
    else:
        st.metric("Años Promedio", "N/A")

st.markdown("---")

# Verificar si hay datos para mostrar
if len(filtered_df) == 0:
    st.warning("No hay datos que coincidan con los filtros seleccionados. Por favor, ajusta los filtros.")
    st.stop()

# Gráficos principales
st.header("Análisis Visual")

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
        text=attrition_pct['Yes'],
        color_discrete_sequence=['#D4A574']
    )
    fig_attrition.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
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
        text = satisfaction_counts.values,
        color_discrete_sequence=['#5A8FC8']
    )
    fig_satisfaction.update_traces(textposition='outside')
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
        color_discrete_sequence=['#5AC86F']
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
        color_discrete_sequence=['#C85A5A']
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
st.header("Análisis de Work-Life Balance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Horas Extra por Departamento")
    overtime_dept = pd.crosstab(filtered_df['Department'], filtered_df['OverTime'], normalize='index') * 100
    
    fig_overtime = px.bar(
        x=overtime_dept.index,
        y=overtime_dept['Yes'],
        title="Porcentaje de Empleados con Horas Extra",
        labels={'x': 'Departamento', 'y': 'Porcentaje con Horas Extra (%)'},
        text = overtime_dept['Yes'],
        color_discrete_sequence=['#C8905A']
    )
    fig_overtime.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_overtime, use_container_width=True)

with col2:
    st.subheader("Work-Life Balance")
    wlb_counts = filtered_df['WorkLifeBalance'].value_counts().sort_index()
    
    fig_wlb = px.bar(
        x=wlb_counts.index,
        y=wlb_counts.values,
        title="Distribución de Work-Life Balance",
        labels={'x': 'Nivel de Work-Life Balance', 'y': 'Número de Empleados'},
        text=wlb_counts.values,
        color_discrete_sequence=['#8F5AC8']
    )
    fig_wlb.update_traces(textposition='outside')
    st.plotly_chart(fig_wlb, use_container_width=True)

# Análisis de attrition por monthly rate
st.header("Análisis de Attrition por Monthly Rate")

st.subheader("Rotación por Monthly Rate")

# Crear el gráfico base
fig_rate = px.scatter(
    filtered_df,
    x='TotalWorkingYears',
    y='MonthlyIncome',
    color='Attrition',
    labels={'x': 'Total Working Years', 'y': 'Monthly Income'},
    color_discrete_map={'No': "#78B3F1", 'Yes': "#BD4242"},
)

# Actualizar las trazas para diferentes opacidades
for trace in fig_rate.data:
    if trace.name == 'No':
        trace.marker.opacity = 0.2  # Opacidad completa para No
    else:  # Yes
        trace.marker.opacity = 0.8  # Opacidad reducida para Yes

fig_rate.update_layout(
    xaxis_title="Total Working Years",
    yaxis_title="Monthly Income"
)
st.plotly_chart(fig_rate, use_container_width=True)


# Heatmap de Correlaciones
st.header("Mapa de Correlaciones")

# Seleccionar solo columnas numéricas
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
correlation_matrix = filtered_df[numeric_cols].corr()

fig_heatmap = px.imshow(
    correlation_matrix,
    title="Matriz de Correlación de Variables Numéricas",
    color_continuous_scale=['#F5F2E8', '#E8DCC6', '#D4A574', '#A67C52', '#4A453E'],
    aspect='auto'
)
fig_heatmap.update_layout(height=600)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Análisis de Rotación Detallado
st.header("Análisis Detallado de Rotación - Insights Clave")

# Crear pestañas para diferentes análisis
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Fatiga Operativa", "Early-Career & Underpaid", "Pared de los 3 Años", "Equidad Interna", "Clima como Amplificador"])

with tab1:
    st.subheader("Fatiga Operativa - Eje Gravitacional")
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
            color_discrete_sequence=['#C85A5A']
        )
        fig_fatigue.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_fatigue.update_layout(
            showlegend=False, 
            yaxis = dict(range=[0, (fatigue_attrition['Attrition_Rate'] if 'Attrition_Rate' in fatigue_attrition.columns else pd.Series([0])).max() * 1.1]))
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

        # Crear campo combinado para evitar apilamiento
        high_risk['JobRole_Detail'] = high_risk['JobRole'] + ' (' + high_risk['BusinessTravel'] + ')'

        if not high_risk.empty:
            fig_high_risk = px.bar(
                high_risk.head(8),
                x='Attrition_Rate',
                y='JobRole_Detail',  # Usar el campo combinado
                orientation='h',
                title="Roles con Mayor Riesgo (OT + Travel)",
                text='Attrition_Rate',
                color_discrete_sequence=['#C85A5A']
            )
            fig_high_risk.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_high_risk.update_layout(showlegend=False)
            st.plotly_chart(fig_high_risk, use_container_width=True)

        st.write("**Top Combinaciones de Riesgo:**")
        if not high_risk.empty:
            st.dataframe(high_risk[['JobRole', 'BusinessTravel', 'Count', 'Attrition_Rate']].round(3))

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
            color_discrete_sequence=['#C8905A']
        )
        fig_risk.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_risk.update_layout(
            showlegend=False, 
            xaxis_tickangle=45,
            yaxis = dict(range=[0, (risk_analysis['Attrition_Rate'] if 'Attrition_Rate' in risk_analysis.columns else pd.Series([0])).max() * 1.1]))
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
            color_continuous_scale=['#F5F2E8', '#E8DCC6', '#D4A574', '#C85A5A', '#4A453E'],
            text_auto='.2f'
        )
        st.plotly_chart(fig_heatmap_age_income, use_container_width=True)

with tab3:
    st.subheader("Pared de los 3 Años")
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
            color_discrete_sequence=['#5A8FC8']
        )
        fig_years.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_years.update_layout(
            showlegend=False,
            yaxis = dict(range=[0, (years_attrition['Attrition_Rate'] if 'Attrition_Rate' in years_attrition.columns else pd.Series([0])).max() * 1.1]))
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
            color_discrete_sequence=['#8F5AC8']
        )
        fig_stagnation.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_stagnation.update_layout(showlegend=False)
        st.plotly_chart(fig_stagnation, use_container_width=True)
        
        st.write("**Análisis de Stagnation Index:**")
        st.dataframe(stagnation_attrition)

with tab4:
    st.subheader("Equidad Interna > Dinero Absoluto")
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
            color_discrete_sequence=['#5AC86F']
        )
        fig_percentile.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_percentile.update_layout(showlegend=False)
        st.plotly_chart(fig_percentile, use_container_width=True)

with tab5:
    st.subheader("Clima como Amplificador")
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
            color_discrete_sequence=['#C85A5A']
        )
        fig_climate.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_climate.update_layout(
            showlegend=False, 
            xaxis_tickangle=45,
            yaxis=dict(range=[0, (climate_summary['Attrition_Rate'] if 'Attrition_Rate' in climate_summary.columns else pd.Series([0])).max() * 1.1]))
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
            color_continuous_scale=['#F5F2E8', '#E8DCC6', '#D4A574', '#C85A5A', '#4A453E'],
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

# Análisis por Grupos de Edad e Ingresos
st.header("Análisis por Grupos Demográficos")

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
        color_discrete_sequence=['#5A8FC8']
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
        color_discrete_sequence=['#C85A5A']
    )
    fig_income_groups.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_income_groups.update_layout(
        showlegend=False,  
        yaxis = dict(range=[0, (income_analysis['Attrition_Rate'] if 'Attrition_Rate' in income_analysis.columns else pd.Series([0])).max() * 1.1]))
    st.plotly_chart(fig_income_groups, use_container_width=True)
    
    # Mostrar tabla
    st.write("**Estadísticas por Grupo de Ingresos:**")
    display_income = income_analysis[['Income_Label', 'Total', 'Attrition_Count', 'Attrition_Rate']].round(3)
    st.dataframe(display_income)

# Análisis Multivariado
st.header("Análisis Multivariado de Attrition")

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
        color_continuous_scale=['#F5F2E8', '#E8DCC6', '#D4A574', '#C85A5A', '#4A453E'],
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
        color_discrete_sequence=['#C85A5A']
    )
    fig_multi.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_multi.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_multi, use_container_width=True)
    
    st.write("**Tabla Detallada:**")
    st.dataframe(multi_analysis[['Department', 'Gender', 'OverTime', 'Count', 'Attrition_Rate']].round(3))

# Análisis por Variables Categóricas
st.header("Análisis por Variables Categóricas")

categorical_cols = ['Department', 'JobRole', 'Gender', 'MaritalStatus', 'EducationField', 'BusinessTravel', 'OverTime']
available_categorical = [col for col in categorical_cols if col in filtered_df.columns]

if available_categorical:
    # Selector de variable
    selected_var = st.selectbox("Selecciona variable para análisis detallado:", available_categorical)
    
    col1, col2 = st.columns(2)
    
    # Obtener todas las categorías únicas y ordenarlas
    all_categories = sorted(filtered_df[selected_var].unique())
    
    with col1:
        st.subheader(f"Distribución de {selected_var}")
        value_counts = filtered_df[selected_var].value_counts()
        
        # Reindexar para incluir todas las categorías (rellenar con 0 si no existen)
        value_counts = value_counts.reindex(all_categories, fill_value=0)
        
        fig_dist = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribución de {selected_var}",
            labels={'x': selected_var, 'y': 'Número de Empleados'},
            text=value_counts.values,
            color_discrete_sequence=['#D4A574']
        )
        fig_dist.update_traces(textposition='outside')
        fig_dist.update_layout(
            showlegend=False, 
            xaxis_tickangle=45,
            xaxis=dict(categoryorder='array', categoryarray=all_categories),
            yaxis=dict(range=[0, value_counts.max() * 1.1])  # Rango Y con 10% extra
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader(f"Attrition por {selected_var}")
        ct = pd.crosstab(filtered_df[selected_var], filtered_df['Attrition'], normalize='index') * 100
        
        # Reindexar para incluir todas las categorías
        ct = ct.reindex(all_categories, fill_value=0)
        
        fig_attrition_cat = px.bar(
            x=ct.index,
            y=ct['Yes'] if 'Yes' in ct.columns else [0] * len(ct.index),
            title=f"Tasa de Attrition por {selected_var}",
            labels={'x': selected_var, 'y': 'Tasa de Attrition (%)'},
            text=ct['Yes'] if 'Yes' in ct.columns else [0] * len(ct.index),
            color_discrete_sequence=['#C85A5A']
        )
        fig_attrition_cat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_attrition_cat.update_layout(
            showlegend=False, 
            xaxis_tickangle=45,
            xaxis=dict(categoryorder='array', categoryarray=all_categories),
            yaxis=dict(range=[0, (ct['Yes'] if 'Yes' in ct.columns else pd.Series([0])).max() * 1.1])  # Rango Y con 10% extra
        )
        st.plotly_chart(fig_attrition_cat, use_container_width=True)
        
        # Mostrar tabla de contingencia
        st.write("**Tabla de Contingencia:**")
        ct_abs = pd.crosstab(filtered_df[selected_var], filtered_df['Attrition'])
        # Reindexar la tabla absoluta también para consistencia
        ct_abs = ct_abs.reindex(all_categories, fill_value=0)
        st.dataframe(ct_abs)

# Vista Comparativa - Todas las Variables
st.header("Vista Comparativa - Todas las Variables")

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
    labels={'Count': 'Número de Empleados', 'Tasa_Attrition': 'Tasa de Attrition (%)'},
    opacity=0.8
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
    text='Tasa_Attrition',
    opacity=0.9
)
fig_top.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig_top.update_layout(height=500)
st.plotly_chart(fig_top, use_container_width=True)

# Tabla resumen
st.write("**Resumen Completo:**")
display_summary = summary_df[['Variable', 'Categoria', 'Tasa_Attrition', 'Count']].round(2)
st.dataframe(display_summary, height=300)

# Tabla de datos resumida
st.header("Resumen de Datos")

summary_stats = filtered_df.describe()
st.dataframe(summary_stats, use_container_width=True)

# Sección de exportación
st.header("Exportar Datos")

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

# ====== SECCIÓN DE PREDICCIÓN DE ATTRITION ======
import joblib 
import pickle

st.header("Predicción de Attrition")
st.write("Utiliza nuestro modelo de machine learning para predecir la probabilidad de que un empleado deje la empresa.")

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        # Intentar cargar el modelo XGBoost optimizado primero
        model_components = joblib.load('attrition_model_xgboost_optimized_f1.pkl')
        return model_components
    except FileNotFoundError:
        try:
            # Intentar cargar modelo XGBoost básico
            model_components = joblib.load('attrition_model_xgboost_optimized.pkl')
            return model_components
        except FileNotFoundError:
            try:
                # Intentar cargar modelo LightGBM
                model_components = joblib.load('attrition_model_lightgbm_target_encoded.pkl')
                return model_components
            except FileNotFoundError:
                st.error("❌ Modelo no encontrado. Asegúrate de haber entrenado y guardado el modelo.")
                return None

model_components = load_model()

if model_components is not None:
    # Función de predicción universal
    def predict_attrition_streamlit(employee_data):
        try:
            model = model_components['model']
            scaler = model_components['scaler']
            feature_names = model_components['feature_names']
            categorical_features = model_components['categorical_features']
            numerical_features = model_components['numerical_features']
            model_type = model_components.get('model_type', 'Unknown')
            
            # Crear dataframe
            emp_df = pd.DataFrame([employee_data])
            
            # Manejar encoding según el tipo de modelo
            if 'TargetEncoded' in model_type:
                # Modelo con Target Encoding
                target_encoder = model_components['target_encoder']
                
                # Separar features categóricas y numéricas
                if categorical_features:
                    emp_categorical = emp_df[categorical_features]
                    emp_categorical_encoded = target_encoder.transform(emp_categorical)
                else:
                    emp_categorical_encoded = pd.DataFrame()
                
                if numerical_features:
                    emp_numerical = emp_df[numerical_features]
                    emp_numerical_scaled = pd.DataFrame(
                        scaler.transform(emp_numerical),
                        columns=numerical_features
                    )
                else:
                    emp_numerical_scaled = pd.DataFrame()
                
                # Combinar features
                emp_features = pd.concat([emp_categorical_encoded, emp_numerical_scaled], axis=1)
                
                # Asegurar orden correcto
                emp_features = emp_features[feature_names]
                
            else:
                # Modelo con Label Encoding
                label_encoders = model_components['label_encoders']
                
                # Label encoding para categóricas
                for feature in categorical_features:
                    if feature in emp_df.columns and feature in label_encoders:
                        le = label_encoders[feature]
                        try:
                            emp_df[f'{feature}_encoded'] = le.transform(emp_df[feature])
                        except ValueError:
                            # Valor no visto, usar 0
                            emp_df[f'{feature}_encoded'] = 0
                
                # Preparar features
                emp_features = emp_df[feature_names].fillna(0)
                
                # Escalar numéricas
                emp_features[numerical_features] = scaler.transform(emp_features[numerical_features])
            
            # Predecir
            probability = model.predict_proba(emp_features)[0]
            prediction = model.predict(emp_features)[0]
            
            # Usar threshold optimizado si está disponible
            threshold = model_components.get('best_threshold', 0.5)
            prediction_optimized = 1 if probability[1] >= threshold else 0
            
            return {
                'prediction': 'Sí' if prediction_optimized == 1 else 'No',
                'probability_no': probability[0],
                'probability_yes': probability[1],
                'threshold_used': threshold,
                'model_type': model_type
            }
        except Exception as e:
            st.error(f"Error en predicción: {str(e)}")
            return None
    
    # Crear formulario de entrada
    st.subheader("Ingresa los datos del empleado:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Información Laboral**")
        years_at_company = st.number_input(
            "Años en la Empresa", 
            min_value=0, 
            max_value=40, 
            value=3,
            help="Número de años que el empleado ha trabajado en la empresa"
        )
        
        monthly_income = st.number_input(
            "Ingreso Mensual ($)", 
            min_value=1000, 
            max_value=25000, 
            value=5000, 
            step=100,
            help="Salario mensual del empleado en dólares"
        )
    
    with col2:
        st.write("**Información Personal**")
        age = st.number_input(
            "Edad", 
            min_value=18, 
            max_value=70, 
            value=30,
            help="Edad del empleado"
        )
        
        overtime = st.selectbox(
            "Horas Extra",
            options=["No", "Yes"],
            index=0,
            help="¿El empleado trabaja horas extra regularmente?"
        )
    
    with col3:
        st.write("**Viajes de Trabajo**")
        business_travel = st.selectbox(
            "Viajes de Negocio",
            options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
            index=0,
            help="Frecuencia de viajes de trabajo del empleado"
        )
        
        # Campo adicional opcional para completar el modelo
        job_level = st.selectbox(
            "Nivel de Trabajo",
            options=[1, 2, 3, 4, 5],
            index=1,
            help="Nivel jerárquico del empleado (1=Junior, 5=Senior)"
        )
    
    # Botón de predicción
    if st.button("Predecir Riesgo de Attrition", type="primary"):
        # Recopilar datos básicos
        employee_data = {
            'YearsAtCompany': years_at_company,
            'MonthlyIncome': monthly_income,
            'Age': age,
            'OverTime': overtime,
            'BusinessTravel': business_travel,
            'JobLevel': job_level,
            # Valores por defecto para otros campos requeridos
            'TotalWorkingYears': max(age - 18, years_at_company),
            'YearsInCurrentRole': min(years_at_company, 2),
            'YearsSinceLastPromotion': min(years_at_company, 1),
            'Education': 3,
            'DistanceFromHome': 10,
            'WorkLifeBalance': 3,
            'JobSatisfaction': 3,
            'EnvironmentSatisfaction': 3,
            'JobInvolvement': 3,
            'Department': 'Sales',  # Valor por defecto
            'JobRole': 'Sales Executive',  # Valor por defecto
            'Gender': 'Male',  # Valor por defecto
            'MaritalStatus': 'Married',  # Valor por defecto
            'EducationField': 'Technical Degree',  # Valor por defecto
            'DailyRate': 800,
            'HourlyRate': 50,
            'MonthlyRate': 15000,
            'NumCompaniesWorked': 2,
            'PercentSalaryHike': 15,
            'PerformanceRating': 3,
            'RelationshipSatisfaction': 3,
            'StockOptionLevel': 1,
            'TrainingTimesLastYear': 3,
            'YearsWithCurrManager': min(years_at_company, 2)
        }
        
        # Hacer predicción
        result = predict_attrition_streamlit(employee_data)
        
        if result is not None:
            # Mostrar resultados
            st.subheader("📊 Resultados de la Predicción")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                # Predicción principal con colores
                if result['prediction'] == 'Sí':
                    st.error(f"⚠️ **Predicción: ALTO RIESGO de Attrition**")
                    st.error(f"🔴 Probabilidad de Attrition: **{result['probability_yes']:.1%}**")
                else:
                    st.success(f"✅ **Predicción: BAJO RIESGO de Attrition**")
                    st.success(f"🟢 Probabilidad de Retención: **{result['probability_no']:.1%}**")
                
                # Información técnica
                st.info(f"🎯 Threshold usado: {result['threshold_used']:.3f}")
                st.info(f"🤖 Modelo: {result['model_type']}")
            
            with col_result2:
                # Gráfico de probabilidades
                prob_data = pd.DataFrame({
                    'Resultado': ['Se Queda', 'Se Va'],
                    'Probabilidad': [result['probability_no'], result['probability_yes']],
                    'Color': ['#5A8FC8', '#C85A5A']
                })
                
                fig_prob = px.bar(
                    prob_data,
                    x='Resultado',
                    y='Probabilidad',
                    title="Probabilidades de Attrition",
                    color='Color',
                    color_discrete_map={'#5A8FC8': '#5A8FC8', '#C85A5A': '#C85A5A'},
                    text='Probabilidad'
                )
                fig_prob.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig_prob.update_layout(
                    showlegend=False, 
                    yaxis=dict(range=[0, 1]),
                    height=300
                )
                st.plotly_chart(fig_prob, use_container_width=True)
            
            # Recomendaciones basadas en el riesgo
            st.subheader("Recomendaciones")
            
            risk_level = result['probability_yes']
            
            if risk_level > 0.7:
                st.error("""
                **RIESGO CRÍTICO (>70%)**
                - **Acción inmediata requerida**
                - Revisar compensación y beneficios urgentemente
                - Evaluar carga de trabajo y balance vida-trabajo
                - Programar reunión one-on-one inmediata con el manager
                - Considerar promoción o cambio de rol
                - Implementar plan de retención personalizado
                """)
            elif risk_level > 0.5:
                st.warning("""
                **RIESGO ALTO (50-70%)**
                - Monitorear de cerca y tomar medidas preventivas
                - Evaluar oportunidades de desarrollo profesional
                - Revisar satisfacción laboral y environment
                - Considerar ajustes en responsabilidades o team
                - Planificar career path claro
                """)
            elif risk_level > 0.3:
                st.info("""
                **RIESGO MODERADO (30-50%)**
                - Monitorear satisfacción laboral regularmente
                - Mantener comunicación abierta con el empleado
                - Evaluar oportunidades de crecimiento a mediano plazo
                - Considerar para proyectos de desarrollo
                """)
            else:
                st.success("""
                **RIESGO BAJO (<30%)**
                - Empleado estable, continuar con seguimiento regular
                - Considerar como mentor para otros empleados
                - Evaluar para roles de mayor responsabilidad
                - Potencial candidato para leadership development
                """)
                
            # Factores de riesgo identificados
            st.subheader("🎯 Factores de Riesgo Detectados")
            risk_factors = []
            
            if overtime == "Yes":
                risk_factors.append("🔴 Trabajo con horas extra")
            if business_travel in ["Travel_Frequently"]:
                risk_factors.append("🔴 Viajes frecuentes")
            elif business_travel == "Travel_Rarely":
                risk_factors.append("🟡 Viajes ocasionales")
            if years_at_company < 2:
                risk_factors.append("🟡 Empleado relativamente nuevo")
            if monthly_income < 3000:
                risk_factors.append("🟡 Salario por debajo del promedio")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("• ✅ No se detectaron factores de riesgo importantes")

else:
    st.warning("⚠️ No se pudo cargar el modelo de predicción. Verifica que el archivo del modelo esté disponible.")




# Footer
st.markdown("---")
st.markdown("Dashboard creado con Streamlit | Datos de HR Analytics")